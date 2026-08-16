from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from fd_psc.config import minimal_fsd_v2_config
from fd_psc.encoder_adapters import get_encoder_adapter
from fd_psc.low_rank_merge import LowRankFactors, as_factors, factor_svd, factors_from_svd
from fd_psc.v2.system import (
    FSDV2IntegrationError,
    _factor_difference_frobenius_sq_float64,
)
from fd_psc.v2.checkpoint import FSDV2CheckpointError
from planning.adajepa import AdaJEPATrainer
from fd_psc.lora_layers import DualLoRALinear


BASE_HASH = "a" * 64
PREPROCESS_HASH = "b" * 64


class _FrozenBackbone(nn.Module):
    def forward_features(self, value: torch.Tensor):
        return {"tokens": value.flatten(2).transpose(1, 2).contiguous()}


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _FrozenBackbone()
        self.feature_key = "tokens"
        self.projector_name = "channel"
        self.projector = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        self.emb_dim = 2
        with torch.no_grad():
            self.projector.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tokens = self.base_model.forward_features(value)[self.feature_key]
        feature_map = tokens.reshape(value.shape[0], 2, 2, 2).permute(0, 3, 1, 2)
        return self.projector(feature_map).flatten(2).transpose(1, 2).contiguous()


class _Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(2, 2, bias=False)
        self.last_dropout_training = False
        with torch.no_grad():
            self.proj.weight.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        normalized = self.norm(value.reshape(-1, shape[-1])).reshape(shape)
        self.last_dropout_training = bool(self.dropout.training)
        normalized = self.dropout(normalized)
        return self.proj(normalized)


class _WorldModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.predictor = _Predictor()
        self.proprio_encoder = nn.Identity()
        self.action_encoder = nn.Identity()
        self.encoder_transform = nn.Identity()
        self.num_hist = 1
        self.concat_dim = 0
        self.action_dim = 2
        self.stop_grad = True
        self._base_checkpoint_hash = BASE_HASH
        self._fd_psc_preprocess_hash = PREPROCESS_HASH
        # Real modules commonly carry scalar counters (for example
        # BatchNorm.num_batches_tracked); base hashing must support them.
        self.register_buffer("audit_counter", torch.tensor(0, dtype=torch.int64))

    def encode(self, obs, actions):
        visual = obs["visual"]
        batch, time = visual.shape[:2]
        projected = self.encoder(visual.reshape(batch * time, *visual.shape[2:]))
        projected = projected.reshape(batch, time, projected.shape[1], projected.shape[2])
        proprio = self.proprio_encoder(obs["proprio"])
        action = self.action_encoder(actions)
        return torch.cat((projected, proprio.unsqueeze(2), action.unsqueeze(2)), dim=2)

    def predict(self, value):
        batch, time, patches, dimension = value.shape
        output = self.predictor(value.reshape(batch, time * patches, dimension))
        return output.reshape(batch, time, patches, dimension)

    def extract_frozen_visual_latent(self, obs):
        return get_encoder_adapter(self).extract_frozen_visual_latent(obs)

    def project_visual_latent(self, latent):
        return get_encoder_adapter(self).project_visual_latent(latent)


def _support_segment(scale: float = 1.0):
    visual = torch.zeros(1, 2, 2, 2, 2)
    visual[:, 0, 0].fill_(float(scale))
    visual[:, 1, 1].fill_(float(scale))
    obs = {
        "visual": visual,
        "proprio": torch.zeros(1, 2, 2),
    }
    actions = torch.zeros(1, 1, 2)
    return obs, actions


def _theta0_snapshot(model: nn.Module):
    values = []
    for parameter in model.parameters():
        values.append((parameter, parameter.detach().clone(), True))
    for buffer in model.buffers():
        values.append((buffer, buffer.detach().clone(), False))
    return values


def _config(*, checkpoint: bool = False, resume_path: str | None = None):
    return minimal_fsd_v2_config(
        seed=17,
        episodic_lora={"rank": 1, "alpha": 1.0, "dropout": 0.0},
        slow_lora={
            "initial_rank": 1,
            "allowed_ranks": [1, 2],
            "maximum_rank": 2,
            "persistent_rank": 2,
        },
        raw_replay={
            "historical_windows": 4,
            "maximum_context_clusters": 4,
            "minimum_windows_per_cluster": 1,
        },
        rtrc={
            "budget_fraction_initial": 0.2,
            "geometry_windows": 4,
            "geometry_maximum_rank": 2,
            "geometry_energy_threshold": 1.0,
        },
        checkpoint={
            "enabled": checkpoint,
            "state_directory": "fsd_v2_state",
            "latest_pointer_path": "fsd_v2_state_latest.json",
            "resume_path": resume_path,
            "save_every_episodes": 1,
            "retention_versions": 3,
        },
    )


def _trainer(root: Path, config, *, seed: int = 3):
    torch.manual_seed(seed)
    model = _WorldModel()
    theta0 = _theta0_snapshot(model)
    trainer = AdaJEPATrainer(
        wm=model,
        lr=0.25,
        steps=1,
        optimizer_name="sgd",
        finetune_encoder=True,
        last_layer_only=False,
        encoder_lr=0.25,
        encoder_last_layer_only=False,
        fd_psc=config,
        runtime_output_dir=str(root),
    )
    return model, trainer, theta0


class FSDV2NoExternalDataTests(unittest.TestCase):
    def assert_theta0_unchanged(self, snapshot) -> None:
        for tensor, expected, parameter in snapshot:
            self.assertTrue(torch.equal(tensor.detach(), expected))
            if parameter:
                self.assertFalse(tensor.requires_grad)

    def _run_episode(self, trainer, episode: int, scale: float):
        obs, actions = _support_segment(scale)
        trainer.begin_fd_psc_episode(
            f"episode-{episode}",
            f"context-{episode}",
            metadata={"sample_idx": episode},
        )
        losses = trainer.finetune([obs], [actions])
        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(losses[0]))
        return obs, actions, trainer.end_fd_psc_episode([obs], [actions])

    def test_compression_error_includes_low_precision_factor_roundoff(self):
        generator = torch.Generator().manual_seed(91)
        candidate = LowRankFactors(
            torch.randn(8, 4, generator=generator, dtype=torch.float16),
            torch.randn(4, 8, generator=generator, dtype=torch.float16),
        )
        reconstructed = factors_from_svd(
            factor_svd(candidate),
            rank=4,
            dtype=torch.float16,
        )
        measured_sq = _factor_difference_frobenius_sq_float64(
            candidate,
            reconstructed,
        )
        dense_error_sq = float(
            torch.sum(
                (
                    candidate.b.to(torch.float64) @ candidate.a.to(torch.float64)
                    - reconstructed.b.to(torch.float64)
                    @ reconstructed.a.to(torch.float64)
                ).square()
            )
        )
        self.assertGreater(measured_sq, 0.0)
        self.assertAlmostEqual(measured_sq, dense_error_sq, places=10)

    def test_optimizer_construction_failure_restores_online_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, trainer, theta0 = _trainer(Path(temporary), _config())
            system = trainer.fd_psc_system
            obs, actions = _support_segment(1.0)
            trainer.begin_fd_psc_episode("episode-0", "context-0")
            trainer.optimizer_name = "definitely-invalid"
            with self.assertRaises(ValueError):
                trainer.finetune([obs], [actions])
            self.assertFalse(system._online_mode)
            self.assertFalse(model.predictor.training)
            self.assertFalse(
                any(parameter.requires_grad for parameter in model.parameters())
            )
            self.assert_theta0_unchanged(theta0)
            trainer.abort_fd_psc_episode("expected-test-failure")

    def test_failed_resume_restores_original_model_for_safe_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fsd_v2_state_latest.json").write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            torch.manual_seed(37)
            model = _WorldModel()
            original_projection = model.predictor.proj
            original_requires_grad = {
                id(parameter): bool(parameter.requires_grad)
                for parameter in model.parameters()
            }
            with self.assertRaisesRegex(
                FSDV2CheckpointError,
                "V1 sidecar cannot be loaded as FSD V2",
            ):
                AdaJEPATrainer(
                    wm=model,
                    lr=0.25,
                    steps=1,
                    optimizer_name="sgd",
                    finetune_encoder=True,
                    last_layer_only=False,
                    encoder_lr=0.25,
                    encoder_last_layer_only=False,
                    fd_psc=_config(
                        checkpoint=True,
                        resume_path="fsd_v2_state_latest.json",
                    ),
                    runtime_output_dir=str(root),
                )
            self.assertIs(model.predictor.proj, original_projection)
            self.assertIsInstance(model.predictor.proj, nn.Linear)
            self.assertFalse(hasattr(model, "_fd_psc_encoder_adapter"))
            for parameter in model.parameters():
                self.assertEqual(
                    bool(parameter.requires_grad),
                    original_requires_grad[id(parameter)],
                )

            retried = AdaJEPATrainer(
                wm=model,
                lr=0.25,
                steps=1,
                optimizer_name="sgd",
                finetune_encoder=True,
                last_layer_only=False,
                encoder_lr=0.25,
                encoder_last_layer_only=False,
                fd_psc=_config(checkpoint=False),
                runtime_output_dir=str(root),
            )
            self.assertIsInstance(model.predictor.proj, DualLoRALinear)
            retried.fd_psc_system.close()

    def test_resume_rejects_changed_theta0_despite_same_declared_base_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, first, _ = _trainer(root, _config(checkpoint=True), seed=43)
            self._run_episode(first, 0, 1.0)

            torch.manual_seed(43)
            changed = _WorldModel()
            original_projection = changed.encoder.projector
            with torch.no_grad():
                changed.encoder.projector.weight.add_(123.0)
            with self.assertRaisesRegex(
                FSDV2CheckpointError,
                "runtime_base_state_hash mismatch",
            ):
                AdaJEPATrainer(
                    wm=changed,
                    lr=0.25,
                    steps=1,
                    optimizer_name="sgd",
                    finetune_encoder=True,
                    last_layer_only=False,
                    encoder_lr=0.25,
                    encoder_last_layer_only=False,
                    fd_psc=_config(
                        checkpoint=True,
                        resume_path="fsd_v2_state_latest.json",
                    ),
                    runtime_output_dir=str(root),
                )
            self.assertIs(changed.encoder.projector, original_projection)
            self.assertIsInstance(changed.encoder.projector, nn.Conv2d)

    def test_two_episodes_run_without_any_external_algorithm_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "fd_psc.external_data.ExternalDataRegistry",
                side_effect=AssertionError("V2 constructed ExternalDataRegistry"),
            ) as registry:
                model, trainer, theta0 = _trainer(root, _config())
                system = trainer.fd_psc_system
                self.assertEqual(system.config.run_mode, "fsd_v2")
                self.assertIsNone(system.external)

                obs, actions = _support_segment(1.0)
                trainer.begin_fd_psc_episode("episode-0", "context-0")
                trainer.finetune([obs], [actions])
                # Normal JEPA wake semantics keep predictor Dropout active,
                # while BatchNorm's immutable theta_0 buffers stay in eval.
                self.assertTrue(model.predictor.last_dropout_training)
                self.assertFalse(model.predictor.norm.training)
                full_tasks = {
                    key: as_factors(value).b @ as_factors(value).a
                    for key, value in system._episode_tasks().items()
                }
                first = trainer.end_fd_psc_episode([obs], [actions])
                self.assertEqual(first["status"], "committed")
                self.assertEqual(first["geometry_replay_window_count"], 0)
                self.assertEqual(first["rtrc_eta"], 0.0)
                self.assertGreater(first["raw_replay_window_count"], 0)
                for logical_id, adapter in system.injection.adapters.items():
                    slow = as_factors(adapter.get_slow_factors())
                    self.assertTrue(
                        torch.allclose(
                            slow.b @ slow.a,
                            full_tasks[logical_id],
                            atol=2.0e-5,
                            rtol=2.0e-5,
                        )
                    )

                _, _, second = self._run_episode(trainer, 1, 1.5)
                self.assertEqual(second["status"], "committed")
                self.assertGreater(second["geometry_replay_window_count"], 0)
                self.assertGreater(second["rtrc_geometry_rank"], 0)
                self.assertTrue(math.isfinite(second["rtrc_eta"]))
                self.assertLessEqual(
                    second["rtrc_accepted_drift"],
                    second["rtrc_delta"]
                    + max(1.0e-8, abs(second["rtrc_delta"]) * 1.0e-6),
                )
                self.assertAlmostEqual(
                    second["final_commit_drift_upper_bound"],
                    second["rtrc_budget_norm"]
                    + second["compression_additive_bound"],
                    places=12,
                )
                metric_names = {event.name for event in system.metrics.events()}
                self.assertTrue(
                    {
                        "rtrc_layer_weight",
                        "rank_compression_error",
                        "rank_compression_relative_error",
                        "compression_additive_bound",
                        "final_commit_drift_upper_bound",
                    }.issubset(metric_names)
                )
                self.assertEqual(system.persistent_model_version, 2)
                self.assertEqual(system._replay_version, 2)
                self.assertEqual(len(system.replay), 2)
                self.assertFalse(
                    any(parameter.requires_grad for parameter in system.wm.parameters())
                )
                self.assertFalse(hasattr(system, "anchor_records"))
                self.assertFalse(hasattr(system, "plasticity_probe"))
                self.assertFalse(hasattr(system, "commit_query"))
                registry.assert_not_called()
                self.assert_theta0_unchanged(theta0)

                runtime_manifest = json.loads(
                    (root / "fsd_v2_runtime_manifest.json").read_text(encoding="utf-8")
                )
                self.assertFalse(runtime_manifest["algorithm_external_data_dependency"])
                self.assertFalse(runtime_manifest["report_test_used_for_algorithm"])

    def test_schema2_checkpoint_roundtrip_restores_raw_replay_and_rejects_v1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, trainer, _ = _trainer(root, _config(checkpoint=True))
            _, _, report = self._run_episode(trainer, 0, 1.0)
            self.assertEqual(report["commit_sequence"], 1)
            pointer = root / "fsd_v2_state_latest.json"
            self.assertTrue(pointer.is_file())
            stored, _ = trainer.fd_psc_system.checkpoint_store.load_pointer()
            live = trainer.fd_psc_system.state_dict()
            self.assertGreater(live["metrics"]["sequence"], 0)
            self.assertEqual(
                stored["metrics"]["sequence"],
                live["metrics"]["sequence"],
            )
            mismatched = copy.deepcopy(stored)
            mismatched["commit_sequence"] = 0
            with self.assertRaisesRegex(
                FSDV2CheckpointError,
                "envelope/state commit_sequence mismatch",
            ):
                trainer.fd_psc_system.checkpoint_store.save(
                    mismatched,
                    commit_sequence=2,
                )
            pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
            version_path = root / "fsd_v2_state" / pointer_value["version_file"]
            envelope = torch.load(
                version_path,
                map_location="cpu",
                weights_only=False,
            )
            envelope["runtime_base_state_hash"] = "c" * 64
            tampered_path = root / "fsd_v2_state" / "tampered-runtime-hash.pt"
            torch.save(envelope, tampered_path)
            with self.assertRaisesRegex(
                FSDV2CheckpointError,
                "runtime_base_state_hash mismatch",
            ):
                trainer.fd_psc_system.checkpoint_store.load_resume(tampered_path)

            _, resumed, resumed_theta0 = _trainer(
                root,
                _config(
                    checkpoint=True,
                    resume_path="fsd_v2_state_latest.json",
                ),
            )
            restored = resumed.fd_psc_system.state_dict()
            self.assertEqual(restored["schema_version"], 2)
            self.assertEqual(restored["algorithm_version"], "fsd_v2")
            self.assertEqual(restored["commit_sequence"], stored["commit_sequence"])
            self.assertEqual(restored["episode_sequence"], stored["episode_sequence"])
            self.assertEqual(
                restored["persistent_model_version"],
                stored["persistent_model_version"],
            )
            restored_windows = resumed.fd_psc_system.replay.windows()
            stored_windows = trainer.fd_psc_system.replay.windows()
            self.assertEqual(len(restored_windows), len(stored_windows))
            for left, right in zip(restored_windows, stored_windows):
                self.assertEqual(left.content_hash, right.content_hash)
                self.assertTrue(torch.equal(left.actions, right.actions))
                self.assertEqual(set(left.obs), set(right.obs))
                for key in left.obs:
                    if torch.is_tensor(left.obs[key]):
                        self.assertTrue(torch.equal(left.obs[key], right.obs[key]))
            self.assert_theta0_unchanged(resumed_theta0)

            with self.assertRaisesRegex(
                FSDV2IntegrationError,
                "V1 sidecar cannot be loaded as FSD V2",
            ):
                resumed.fd_psc_system.load_state_dict({"schema_version": 1})

            system = resumed.fd_psc_system
            before = system.state_dict()
            before_slow = {
                key: as_factors(adapter.get_slow_factors()).b
                @ as_factors(adapter.get_slow_factors()).a
                for key, adapter in system.injection.adapters.items()
            }
            before_replay_hashes = [
                window.content_hash for window in system.replay.windows()
            ]
            corrupt = copy.deepcopy(before)
            corrupt["state_machine"]["sequence"] = 999999
            corrupt["metrics"] = {"schema_version": -1}
            with self.assertRaisesRegex(Exception, "metrics schema"):
                system.load_state_dict(corrupt)
            after = system.state_dict()
            for key in (
                "episode_sequence",
                "commit_sequence",
                "persistent_model_version",
                "replay_version",
                "metrics",
                "state_machine",
            ):
                self.assertEqual(after[key], before[key])
            self.assertEqual(
                [window.content_hash for window in system.replay.windows()],
                before_replay_hashes,
            )
            for key, adapter in system.injection.adapters.items():
                slow = as_factors(adapter.get_slow_factors())
                self.assertTrue(torch.equal(slow.b @ slow.a, before_slow[key]))

    def test_checkpoint_resume_matches_uninterrupted_stochastic_wake(self):
        with tempfile.TemporaryDirectory() as continuous_dir, tempfile.TemporaryDirectory() as resumed_dir:
            continuous_root = Path(continuous_dir)
            resumed_root = Path(resumed_dir)

            _, continuous, _ = _trainer(
                continuous_root,
                _config(checkpoint=True),
                seed=29,
            )
            self._run_episode(continuous, 0, 1.0)
            _, _, continuous_second = self._run_episode(continuous, 1, 1.5)
            continuous_slow = {
                key: (
                    as_factors(adapter.get_slow_factors()).b
                    @ as_factors(adapter.get_slow_factors()).a
                ).detach().clone()
                for key, adapter in continuous.fd_psc_system.injection.adapters.items()
            }

            _, first_process, _ = _trainer(
                resumed_root,
                _config(checkpoint=True),
                seed=29,
            )
            self._run_episode(first_process, 0, 1.0)
            _, resumed, _ = _trainer(
                resumed_root,
                _config(
                    checkpoint=True,
                    resume_path="fsd_v2_state_latest.json",
                ),
                seed=29,
            )
            _, _, resumed_second = self._run_episode(resumed, 1, 1.5)

            for name in (
                "rtrc_beta",
                "rtrc_eta",
                "rtrc_delta",
                "rtrc_raw_drift",
                "rtrc_accepted_drift",
                "rank_compression_error",
                "compression_additive_bound",
                "final_commit_drift_upper_bound",
            ):
                with self.subTest(metric=name):
                    self.assertEqual(resumed_second[name], continuous_second[name])
            for key, adapter in resumed.fd_psc_system.injection.adapters.items():
                slow = as_factors(adapter.get_slow_factors())
                self.assertTrue(torch.equal(slow.b @ slow.a, continuous_slow[key]))
            metric_rows = [
                json.loads(line)
                for line in (resumed_root / "fsd_v2_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["sequence"] for row in metric_rows],
                list(range(resumed.fd_psc_system.metrics.state_dict()["sequence"])),
            )
            self.assertEqual(
                {row["episode_id"] for row in metric_rows},
                {"episode-0", "episode-1"},
            )

    def test_checkpoint_failure_rolls_back_slow_replay_versions_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, trainer, theta0 = _trainer(root, _config(checkpoint=True))
            system = trainer.fd_psc_system
            budget_before = copy.deepcopy(system.budget_controller.state_dict())
            obs, actions = _support_segment(1.0)
            trainer.begin_fd_psc_episode("episode-0", "context-0")
            trainer.finetune([obs], [actions])

            with mock.patch.object(
                system.checkpoint_store,
                "save",
                side_effect=OSError("injected checkpoint failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected checkpoint failure"):
                    trainer.end_fd_psc_episode([obs], [actions])

            self.assertEqual(system.state_machine.state.value, "idle")
            self.assertIsNone(system._active_episode_id)
            self.assertEqual(system._commit_sequence, 0)
            self.assertEqual(system.persistent_model_version, 0)
            self.assertEqual(system._replay_version, 0)
            self.assertEqual(len(system.replay), 0)
            self.assertEqual(system.metrics.state_dict()["sequence"], 0)
            self.assertEqual(
                system.budget_controller.state_dict()["u"],
                budget_before["u"],
            )
            self.assertEqual(
                system.budget_controller.state_dict()["update_count"],
                budget_before["update_count"],
            )
            for adapter in system.injection.adapters.values():
                self.assertEqual(as_factors(adapter.get_slow_factors()).rank, 0)
            self.assertFalse((root / "fsd_v2_state_latest.json").exists())
            self.assert_theta0_unchanged(theta0)

    def test_pointer_publication_failure_removes_uncommitted_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, trainer, theta0 = _trainer(root, _config(checkpoint=True))
            system = trainer.fd_psc_system
            obs, actions = _support_segment(1.0)
            trainer.begin_fd_psc_episode("episode-0", "context-0")
            trainer.finetune([obs], [actions])
            with mock.patch(
                "fd_psc.v2.checkpoint._atomic_json",
                side_effect=OSError("injected pointer publication failure"),
            ):
                with self.assertRaisesRegex(
                    FSDV2CheckpointError,
                    "checkpoint save failed",
                ):
                    trainer.end_fd_psc_episode([obs], [actions])
            self.assertEqual(system._commit_sequence, 0)
            self.assertEqual(len(system.replay), 0)
            self.assertFalse((root / "fsd_v2_state_latest.json").exists())
            state_dir = root / "fsd_v2_state"
            self.assertEqual(list(state_dir.glob("state-commit-*.pt")), [])
            self.assert_theta0_unchanged(theta0)

    def test_post_publish_prune_failure_cannot_rollback_published_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, trainer, theta0 = _trainer(root, _config(checkpoint=True))
            system = trainer.fd_psc_system
            with mock.patch.object(
                system.checkpoint_store,
                "_prune",
                side_effect=OSError("injected best-effort prune failure"),
            ):
                _, _, report = self._run_episode(trainer, 0, 1.0)
            self.assertEqual(report["status"], "committed")
            self.assertEqual(system._commit_sequence, 1)
            self.assertEqual(system.persistent_model_version, 1)
            self.assertEqual(system._replay_version, 1)
            self.assertEqual(len(system.replay), 1)
            stored, reference = system.checkpoint_store.load_pointer()
            self.assertEqual(reference.commit_sequence, 1)
            self.assertEqual(stored["commit_sequence"], 1)
            self.assertEqual(len(stored["raw_replay"]["clusters"]), 1)
            self.assert_theta0_unchanged(theta0)

    def test_reporting_failure_does_not_fail_or_leak_committed_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, trainer, theta0 = _trainer(root, _config(checkpoint=True))
            system = trainer.fd_psc_system
            with mock.patch.object(
                system,
                "_export_metrics",
                side_effect=RuntimeError("injected reporting failure"),
            ):
                _, _, report = self._run_episode(trainer, 0, 1.0)
            self.assertEqual(report["status"], "committed")
            self.assertIn("RuntimeError", report["metrics_export_error"])
            self.assertEqual(system.state_machine.state.value, "idle")
            self.assertIsNone(system._active_episode_id)
            self.assertFalse(system._online_mode)
            self.assertFalse(
                any(parameter.requires_grad for parameter in system.wm.parameters())
            )
            stored, _ = system.checkpoint_store.load_pointer()
            self.assertEqual(stored["commit_sequence"], 1)
            self.assertEqual(
                len(stored["metrics_audit"]),
                stored["metrics"]["sequence"],
            )
            self.assert_theta0_unchanged(theta0)

            _, resumed, _ = _trainer(
                root,
                _config(
                    checkpoint=True,
                    resume_path="fsd_v2_state_latest.json",
                ),
            )
            _, _, second = self._run_episode(resumed, 1, 1.5)
            self.assertEqual(second["status"], "committed")
            rows = [
                json.loads(line)
                for line in (root / "fsd_v2_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["sequence"] for row in rows],
                list(range(resumed.fd_psc_system.metrics.state_dict()["sequence"])),
            )


if __name__ == "__main__":
    unittest.main()
