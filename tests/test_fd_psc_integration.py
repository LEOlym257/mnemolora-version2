from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from fd_psc.activation_subspace import ActivationSubspace
from fd_psc.checkpoint import CheckpointValidationError
from fd_psc.config import minimal_test_config
from fd_psc.diagnostics import bitwise_state_equal
from fd_psc.encoder_adapters import LATENT_SCHEMA_VERSION, get_encoder_adapter
from fd_psc.external_data import (
    CommitQueryAccessError,
    DataLeakageError,
    ExternalDataError,
    canonical_json_hash,
)
from fd_psc.low_rank_merge import LowRankFactors
from fd_psc.repair import RepairEngine, ScreeningResult
from fd_psc.replay_memory import ReplayWindow
from fd_psc.state_machine import FDPSCState, ProposalType
from fd_psc.trainer import FDPSCIntegrationError
from fd_psc.transaction import RNGSnapshot
from planning.adajepa import AdaJEPATrainer
from planning.adajepa_mpc import AdaJEPAMPCPlanner


BASE_HASH = "a" * 64
PREPROCESS_HASH = "b" * 64


class _ToyFrozenBackbone(nn.Module):
    """Two-channel pixels are the four frozen visual tokens."""

    def forward_features(self, value: torch.Tensor):
        return {"tokens": value.flatten(2).transpose(1, 2).contiguous()}


class _ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _ToyFrozenBackbone()
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


class _ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class _ToyWorldModel(nn.Module):
    """Minimal supported DINO-style world model with AdaJEPA tensor semantics."""

    def __init__(self):
        super().__init__()
        self.encoder = _ToyEncoder()
        self.predictor = _ToyPredictor()
        self.proprio_encoder = nn.Identity()
        self.action_encoder = nn.Identity()
        self.encoder_transform = nn.Identity()
        self.num_hist = 1
        self.concat_dim = 0
        self.action_dim = 2
        self.stop_grad = True
        self._base_checkpoint_hash = BASE_HASH
        self._fd_psc_preprocess_hash = PREPROCESS_HASH
        self._fd_psc_preprocess_hash = PREPROCESS_HASH

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
        result = self.predictor(value.reshape(batch, time * patches, dimension))
        return result.reshape(batch, time, patches, dimension)

    def extract_frozen_visual_latent(self, obs):
        return get_encoder_adapter(self).extract_frozen_visual_latent(obs)

    def project_visual_latent(self, latent):
        return get_encoder_adapter(self).project_visual_latent(latent)

    def encode_from_frozen_visual_latent(self, latent, proprio, actions):
        visual = self.project_visual_latent(latent)
        return torch.cat(
            (
                visual,
                self.proprio_encoder(proprio).unsqueeze(2),
                self.action_encoder(actions).unsqueeze(2),
            ),
            dim=2,
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_payload(split_name: str) -> dict:
    # Store the contractually stable cut before the projection head. Replaying
    # it traverses both injected projection and predictor targets.
    source = [[1.0, 0.0]] * 4
    target = [[0.0, 1.0]] * 4
    return {
        "frozen_visual_latent": {
            "tensor": [source, target],
            "layout": "tokens",
            "encoder_type": f"{_ToyEncoder.__module__}.{_ToyEncoder.__qualname__}",
            "cut_path": "encoder.base_model.forward_features['tokens']",
            "schema_version": LATENT_SCHEMA_VERSION,
            "metadata": {
                "leading_shape": [1, 2],
                "feature_key": "tokens",
                "projector_name": "channel",
                "token_grid": [2, 2],
            },
        },
        "proprio": [[[0.0, 0.0], [0.0, 0.0]]],
        "actions": [[[0.0, 0.0]]],
        "fixed_split_nonce": split_name,
    }


def _write_fixed_manifest(root: Path) -> tuple[Path, dict[str, Path]]:
    split_paths: dict[str, Path] = {}
    split_specs = {}
    contexts = {"ctx": {}}
    for index, split_name in enumerate(
        (
            "calibration",
            "commit_query",
            "plasticity_support",
            "plasticity_query",
            "report_test",
            "anchor",
        )
    ):
        payload = _external_payload(split_name)
        record = {
            "record_id": f"{split_name}-record",
            "context_identifier": "ctx",
            "trajectory_id": f"trajectory-{split_name}-{index}",
            "transition_ids": [f"transition-{split_name}-{index}"],
            "frame_ids": [
                f"frame-{split_name}-{index}-0",
                f"frame-{split_name}-{index}-1",
            ],
            "content_hash": canonical_json_hash(payload),
            "payload": payload,
        }
        split_path = root / f"{split_name}.json"
        split_path.write_text(
            json.dumps({"schema_version": 1, "records": [record]}, sort_keys=True),
            encoding="utf-8",
        )
        split_paths[split_name] = split_path
        split_specs[split_name] = {
            "path": split_path.name,
            "sha256": _sha256_file(split_path),
        }
        contexts["ctx"][split_name] = [record["record_id"]]

    manifest = root / "external_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_checkpoint_hash": BASE_HASH,
                "preprocess_hash": PREPROCESS_HASH,
                "latent_adapter_schema": LATENT_SCHEMA_VERSION,
                "splits": split_specs,
                "contexts": contexts,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest, split_paths


def _write_canary_manifest(root: Path) -> Path:
    payload = {"task": "fixed-canary", "reset_state": 7}
    path = root / "canary_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_checkpoint_hash": BASE_HASH,
                "preprocess_hash": PREPROCESS_HASH,
                "environment_id": "mock-resettable-env/v1",
                "deterministic_reset": True,
                "scenarios": [
                    {
                        "scenario_id": "fixed-0",
                        "context_identifier": "ctx",
                        "seed": 101,
                        "payload": payload,
                        "content_hash": canonical_json_hash(payload),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _config(manifest: Path, split_paths: dict[str, Path], **overrides):
    settings = {
        "seed": 17,
        "target_modules": {
            "predictor_linear": True,
            "post_backbone_projection_linear": True,
            "action_encoder_linear": False,
            "proprio_encoder_linear": False,
            "exclude_frozen_backbone": True,
            "require_active_forward_path": True,
            "fail_on_empty_predictor_targets": True,
            "fail_on_empty_projection_targets": False,
            "require_projection_targets_if_head_exists": True,
        },
        "episodic_lora": {"rank": 1, "alpha": 1.0, "dropout": 0.0},
        "slow_lora": {
            "initial_rank": 1,
            "allowed_ranks": [1, 2],
            "maximum_rank": 2,
            "spectral_energy_threshold": 0.99,
            "functional_error_threshold": 1.0,
        },
        "replay": {
            "historical_windows": 4,
            "maximum_context_clusters": 4,
            "minimum_windows_per_cluster": 1,
        },
        "external_eval_data": {
            "manifest_path": str(manifest),
            "calibration_path": str(split_paths["calibration"]),
            "commit_query_path": str(split_paths["commit_query"]),
            "verify_checksums": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    return minimal_test_config(**settings)


def _support_segment():
    visual = torch.zeros(1, 2, 2, 2, 2)
    visual[:, 0, 0].fill_(1.0)
    visual[:, 1, 1].fill_(1.0)
    obs = {
        "visual": visual,
        "proprio": torch.zeros(1, 2, 2),
    }
    actions = torch.zeros(1, 1, 2)
    return obs, actions


def _one_step_support_chain(
    count: int = 3,
    *,
    identity_gap_before=None,
    corrupt_tensor_boundary_before=None,
):
    """Return audited one-action segments with explicit trajectory identities."""

    def visual_frame(index: int) -> torch.Tensor:
        value = torch.zeros(1, 2, 2, 2)
        value[:, 0].fill_(float(index) / 10.0)
        value[:, 1].fill_(1.0 + float(index) / 20.0)
        return value

    def proprio_frame(index: int) -> torch.Tensor:
        return torch.tensor([[float(index), -float(index)]])

    segments = []
    identities = {}
    for index in range(count):
        first_visual = visual_frame(index)
        first_proprio = proprio_frame(index)
        if corrupt_tensor_boundary_before == index:
            first_visual = first_visual.clone()
            first_visual[:, 0].add_(0.25)
        obs = {
            "visual": torch.stack((first_visual, visual_frame(index + 1)), dim=1),
            "proprio": torch.stack(
                (first_proprio, proprio_frame(index + 1)), dim=1
            ),
        }
        actions = torch.tensor([[[float(index), 1.0]]])
        first_frame_id = (
            f"gap-frame-{index}"
            if identity_gap_before == index
            else f"chain-frame-{index}"
        )
        identities[str(index)] = {
            "record_id": f"chain-support-{index}",
            "context_identifier": "ctx",
            "trajectory_id": "chain-trajectory",
            "transition_ids": [f"chain-transition-{index}"],
            "frame_ids": [first_frame_id, f"chain-frame-{index + 1}"],
        }
        segments.append((obs, actions))
    return segments, identities


def _theta0_snapshot(model: nn.Module):
    values = []
    seen = set()
    for tensor in model.parameters():
        if id(tensor) not in seen:
            values.append((tensor, tensor.detach().clone()))
            seen.add(id(tensor))
    for module in model.modules():
        for name, tensor in module._buffers.items():
            if (
                tensor is not None
                and name not in module._non_persistent_buffers_set
                and id(tensor) not in seen
            ):
                values.append((tensor, tensor.detach().clone()))
                seen.add(id(tensor))
    return values


def _replay_window(index: int, context: str) -> ReplayWindow:
    return ReplayWindow(
        window_id=f"repair-window-{context}-{index}",
        trajectory_id=f"repair-trajectory-{context}-{index}",
        transition_ids=(f"repair-transition-{context}-{index}",),
        frame_ids=(
            f"repair-frame-{context}-{index}-0",
            f"repair-frame-{context}-{index}-1",
        ),
        timesteps=(0, 1),
        content_hash=hashlib.sha256(
            f"repair-content-{context}-{index}".encode("utf-8")
        ).hexdigest(),
        context_identifier=context,
        context_embedding=(1.0, 0.0),
        visual_latent=torch.tensor([float(index)]),
        proprio=torch.zeros(1, 2, 2),
        actions=torch.zeros(1, 1, 2),
        source_episode="repair-fixture",
        committed=True,
    )


class FDPSCIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.split_paths = _write_fixed_manifest(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _trainer(
        self,
        *,
        learning_rate: float,
        steps: int = 1,
        config_overrides=None,
        canary_evaluator=None,
    ):
        model = _ToyWorldModel()
        theta0 = _theta0_snapshot(model)
        trainer = AdaJEPATrainer(
            wm=model,
            lr=learning_rate,
            steps=steps,
            optimizer_name="sgd",
            finetune_encoder=True,
            last_layer_only=False,
            encoder_lr=learning_rate,
            encoder_last_layer_only=False,
            fd_psc=_config(
                self.manifest,
                self.split_paths,
                **(config_overrides or {}),
            ),
            runtime_output_dir=str(self.root),
            fd_psc_canary_evaluator=canary_evaluator,
        )
        return model, trainer, theta0

    def assertTheta0Bitwise(self, snapshot):
        for tensor, expected in snapshot:
            self.assertTrue(torch.equal(tensor.detach(), expected), msg=str(tensor.shape))
            if isinstance(tensor, nn.Parameter):
                self.assertFalse(tensor.requires_grad)

    def _install_zero_exception(self, system, obs) -> str:
        descriptor = system._context_descriptor(obs, "ctx")
        self.assertIsNotNone(descriptor)
        layers = {}
        for logical_id, adapter in sorted(system.injection.adapters.items()):
            factors = adapter.get_slow_factors()
            layers[logical_id] = {
                "B": torch.zeros(
                    factors.out_features,
                    0,
                    dtype=factors.B.dtype,
                ),
                "A": torch.zeros(
                    0,
                    factors.in_features,
                    dtype=factors.A.dtype,
                ),
            }
        adapter_id, evicted = system.exception_router.commit_new(
            adapter_state={
                "schema_version": 1,
                "target_manifest_hash": system.target_manifest.hash,
                "layers": layers,
            },
            context_descriptors=[descriptor],
            validation_gain=0.1,
            metadata={"fixture": "zero_exception"},
        )
        self.assertIsNone(evicted)
        record = system.exception_router.get(adapter_id)
        self.assertEqual(record.usage_count, 1)
        self.assertEqual(record.last_used_clock, 1)
        return adapter_id

    def test_two_episode_gradient_match_and_conflict_choose_safe_coefficients(self):
        """V2 section 26 scenarios 1/2 and shared-vs-safe merge behavior."""

        algorithm = {
            "activation_subspace": {
                "enabled": True,
                "maximum_rank": 1,
                "spectral_energy_threshold": 0.99,
                "minimum_energy": 1.0e-8,
            },
            # Geometry stays out of the online loop in this test.  These batch
            # sizes activate the calibration/history similarity signal during
            # sleep, where exact effective-weight gradients are injected below.
            "gradient_geometry": {
                "enabled": False,
                "current_batches": 1,
                "history_batches": 1,
                "anchor_batches": 1,
                "windows_per_batch": 1,
            },
            "merge": {
                "soft_ness_enabled": True,
                "shared_coefficients": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                "safe_coefficients": [0.5, 1.0],
                "use_context_similarity": False,
                "use_gradient_similarity": True,
                "use_residual_similarity": False,
                "gradient_conflict_threshold": -0.1,
                "gradient_match_threshold": 0.1,
            },
        }

        def run_pair(history_sign: float):
            _, trainer, theta0 = self._trainer(
                learning_rate=0.25,
                steps=2,
                config_overrides=algorithm,
            )
            system = trainer.fd_psc_system
            obs, actions = _support_segment()
            selected_candidates = []

            def select_largest_safe_candidate(_trainer, candidates, **kwargs):
                selected = max(
                    candidates,
                    key=lambda item: (item.alpha_shared, item.alpha_safe),
                )
                selected.calibration_loss = float(kwargs["calibration_fast"])
                selected.calibration_gain = float(
                    kwargs["calibration_before"] - kwargs["calibration_fast"]
                )
                selected.screening_reason = "mock_fixed_calibration_selection"
                selected_candidates.append(selected)
                return selected, [candidate.summary() for candidate in candidates]

            # Episode one creates committed historical replay.  Giving every
            # episode a distinct audited sample identity avoids conflating a
            # repeated tensor fixture with a duplicate trajectory window.
            trainer.begin_fd_psc_episode(
                system.next_episode_id,
                "ctx",
                initial_obs=obs,
                metadata={"sample_idx": 0, "seed": 17},
            )
            system.register_support_segment(obs, actions, iteration=0)
            trainer.finetune([obs], [actions])
            with mock.patch.object(
                system,
                "_screen_candidates",
                side_effect=select_largest_safe_candidate,
            ):
                first_report = trainer.end_fd_psc_episode([obs], [actions])
            self.assertEqual(first_report["fd_psc_outcome"], FDPSCState.COMMIT_SLOW.value)
            self.assertEqual(len(system.replay), 1)

            # Make the historical direction and its orthogonal safe direction
            # explicit for every logical Linear/Conv group.  The second task
            # has components in both directions.
            for logical_id, adapter in sorted(system.injection.adapters.items()):
                self.assertEqual(adapter.in_features, 2)
                system.subspaces.set(
                    logical_id,
                    ActivationSubspace(
                        torch.tensor([[1.0], [0.0]]),
                        torch.tensor([1.0]),
                    ),
                )

            trainer.begin_fd_psc_episode(
                system.next_episode_id,
                "ctx",
                initial_obs=obs,
                metadata={"sample_idx": 1, "seed": 17},
            )
            system.register_support_segment(obs, actions, iteration=0)
            trainer.finetune([obs], [actions])
            for adapter in system.injection.adapters.values():
                with torch.no_grad():
                    adapter.pilot_A.zero_()
                    adapter.pilot_A[0, 0] = 1.0
                    adapter.pilot_A[0, 1] = 1.0
                    adapter.pilot_B.zero_()
                    adapter.pilot_B[0, 0] = 0.2

            current_gradient = {
                logical_id: torch.ones(adapter.out_features, adapter.in_features)
                for logical_id, adapter in sorted(system.injection.adapters.items())
            }
            gradient_results = [
                current_gradient,
                {
                    key: float(history_sign) * value
                    for key, value in current_gradient.items()
                },
            ]

            def fixed_gradient_pair(*_args, **_kwargs):
                return gradient_results.pop(0)

            with mock.patch.object(
                system,
                "_collect_effective_gradients",
                side_effect=fixed_gradient_pair,
            ), mock.patch.object(
                system,
                "_screen_candidates",
                side_effect=select_largest_safe_candidate,
            ):
                second_report = trainer.end_fd_psc_episode([obs], [actions])

            self.assertFalse(gradient_results)
            self.assertEqual(second_report["fd_psc_outcome"], FDPSCState.COMMIT_SLOW.value)
            self.assertEqual(len(system.replay), 2)
            self.assertTheta0Bitwise(theta0)
            return selected_candidates[-1]

        matched = run_pair(1.0)
        conflicted = run_pair(-1.0)

        self.assertEqual(matched.alpha_shared, 1.0)
        self.assertEqual(matched.alpha_safe, 1.0)
        self.assertEqual(conflicted.alpha_shared, 0.25)
        self.assertEqual(conflicted.alpha_safe, 1.0)
        self.assertLess(conflicted.alpha_shared, matched.alpha_shared)
        self.assertGreater(conflicted.alpha_safe, conflicted.alpha_shared)

        # With Q=e1 and p=1/2, conflict attenuation changes the shared e1
        # coefficient from 1 to .625 while the orthogonal e2 safe coefficient
        # remains exactly 1.  This asserts the actual factor transform, not
        # merely the scalar grid labels.
        for task in conflicted.task_factors_by_layer.values():
            self.assertTrue(torch.allclose(task.A[:, 0], torch.tensor([0.625])))
            self.assertTrue(torch.equal(task.A[:, 1], torch.tensor([1.0])))

    def test_near_null_activation_episode_preserves_safe_task_and_commits(self):
        """A task in Q's activation near-null space is retained by soft-NESS."""

        _, trainer, theta0 = self._trainer(
            learning_rate=0.1,
            config_overrides={
                "activation_subspace": {"enabled": True, "maximum_rank": 1},
                "merge": {
                    "soft_ness_enabled": True,
                    "shared_coefficients": [0.0],
                    "safe_coefficients": [1.0],
                    "use_context_similarity": False,
                    "use_gradient_similarity": False,
                    "use_residual_similarity": False,
                },
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        for logical_id, adapter in sorted(system.injection.adapters.items()):
            system.subspaces.set(
                logical_id,
                ActivationSubspace(
                    torch.tensor([[1.0], [0.0]]),
                    torch.tensor([1.0]),
                ),
            )
            self.assertEqual(adapter.in_features, 2)

        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
            metadata={"sample_idx": 0, "seed": 17},
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        for adapter in system.injection.adapters.values():
            with torch.no_grad():
                adapter.pilot_A.zero_()
                adapter.pilot_A[0, 1] = 1.0
                adapter.pilot_B.zero_()
                adapter.pilot_B[0, 0] = 0.2

        selected_candidates = []

        def select_only_candidate(_trainer, candidates, **kwargs):
            self.assertEqual(len(candidates), 1)
            selected = candidates[0]
            selected.calibration_loss = float(kwargs["calibration_fast"])
            selected.calibration_gain = float(
                kwargs["calibration_before"] - kwargs["calibration_fast"]
            )
            selected.screening_reason = "mock_near_null_selection"
            selected_candidates.append(selected)
            return selected, [selected.summary()]

        with mock.patch.object(
            system,
            "_screen_candidates",
            side_effect=select_only_candidate,
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertEqual(report["fd_psc_outcome"], FDPSCState.COMMIT_SLOW.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertEqual(len(selected_candidates), 1)
        for logical_id, task in selected_candidates[0].task_factors_by_layer.items():
            q = system.subspaces.get(logical_id, 2, task.A.device).q
            self.assertTrue(torch.equal(task.A, torch.tensor([[0.0, 1.0]])))
            self.assertEqual(float((task.A @ q).abs().max()), 0.0)
        metric_events = system.metrics.events()
        spectral = [event for event in metric_events if event.name == "spectral_energy"]
        self.assertTrue(spectral)
        self.assertTrue(all(event.tags["status"] == "available" for event in spectral))
        self.assertTrue(all(0.0 <= float(event.value) <= 1.0 for event in spectral))
        self.assertTrue(
            all(
                event.tags["definition"]
                == "retained_factor_spectral_energy_fraction"
                for event in spectral
            )
        )
        self.assertTrue(
            any(event.name == "activation_energy_total" for event in metric_events)
        )
        self.assertTrue(any(event.name == "lambda_distribution" for event in metric_events))
        self.assertTrue(any(event.name == "p_distribution" for event in metric_events))
        self.assertTheta0Bitwise(theta0)

    def test_anchor_conflict_projects_gradient_and_triggers_centered_episode(self):
        """A real update event detects an immutable-anchor conflict end to end."""

        _, trainer, theta0 = self._trainer(
            learning_rate=0.1,
            config_overrides={
                "gradient_geometry": {
                    "enabled": True,
                    "ema_beta": 0.0,
                    "conflict_threshold": -0.1,
                    "minimum_transitions": 1,
                    "consecutive_conflicts": 1,
                    "current_batches": 1,
                    "history_batches": 1,
                    "anchor_batches": 1,
                    "windows_per_batch": 1,
                    "projection_method": "dual_constraint",
                },
                "slice": {
                    "enabled": True,
                    "rank": 1,
                    "randomized_svd_oversampling": 1,
                    "power_iterations": 0,
                    "maximum_scale": 10.0,
                },
                "anchor_data": {
                    "manifest_path": str(self.manifest),
                    "data_path": str(self.split_paths["anchor"]),
                    "windows": 1,
                },
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
            metadata={"sample_idx": 0, "seed": 17},
        )
        system.register_support_segment(obs, actions, iteration=0)

        current_gradients = {}
        anchor_gradients = {}
        for logical_id, adapter in sorted(system.injection.adapters.items()):
            current = torch.zeros(adapter.out_features, adapter.in_features)
            anchor = torch.zeros_like(current)
            current[0, 0] = 1.0
            current[0, 1] = 1.0
            anchor[0, 0] = -1.0
            current_gradients[logical_id] = current
            anchor_gradients[logical_id] = anchor
        fixed_collections = [current_gradients, anchor_gradients]

        def fixed_current_then_anchor(*_args, **_kwargs):
            return fixed_collections.pop(0)

        with mock.patch.object(
            system,
            "_collect_effective_gradients",
            side_effect=fixed_current_then_anchor,
        ):
            losses = trainer.finetune([obs], [actions])

        self.assertEqual(len(losses), 1)
        self.assertFalse(fixed_collections)
        self.assertEqual(system.state_machine.state, FDPSCState.EPISODE_CENTERED)
        self.assertTrue(
            all(adapter.centered_active for adapter in system.injection.adapters.values())
        )
        for logical_id, cosine in system._latest_gradient_cosines.items():
            self.assertLess(cosine["anchor"], -0.1)
            corrected = system._latest_corrected_gradients[logical_id]
            self.assertGreaterEqual(
                float(torch.sum(corrected * anchor_gradients[logical_id])),
                -1.0e-7,
            )

        system.finish_episode_without_sleep("anchor_conflict_verified")
        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertTheta0Bitwise(theta0)

    def test_slow_rank_saturation_rejects_capped_candidate_without_query(self):
        """A rank-two persistent merge cannot silently truncate at rank one."""

        _, trainer, theta0 = self._trainer(
            learning_rate=0.1,
            config_overrides={
                "slow_lora": {
                    "initial_rank": 1,
                    "allowed_ranks": [1],
                    "maximum_rank": 1,
                    "spectral_energy_threshold": 1.0,
                    "functional_error_threshold": 0.01,
                }
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        slow_before = {}
        for logical_id, adapter in sorted(system.injection.adapters.items()):
            slow_b = torch.zeros(adapter.out_features, 1)
            slow_a = torch.zeros(1, adapter.in_features)
            slow_b[0, 0] = 1.0
            slow_a[0, 0] = 1.0
            adapter.replace_slow_adapter(slow_b, slow_a)
            slow_before[logical_id] = (
                adapter.slow_B.detach().clone(),
                adapter.slow_A.detach().clone(),
            )

        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
            metadata={"sample_idx": 0, "seed": 17},
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        for adapter in system.injection.adapters.values():
            with torch.no_grad():
                adapter.pilot_B.zero_()
                adapter.pilot_B[1, 0] = 1.0
                adapter.pilot_A.zero_()
                adapter.pilot_A[0, 1] = 1.0

        candidates_seen = []
        original_make_candidates = system._make_candidates
        original_evaluate = system._evaluate_state

        def capture_candidates(*args, **kwargs):
            candidates = original_make_candidates(*args, **kwargs)
            candidates_seen.extend(candidates)
            return candidates

        def identity_activations(*_args, **_kwargs):
            return {
                logical_id: torch.eye(adapter.in_features)
                for logical_id, adapter in sorted(system.injection.adapters.items())
            }

        def positive_calibration_gain(
            trainer_arg,
            records,
            *,
            state,
            candidate=None,
        ):
            if records and getattr(records[0], "split_name", "") == "calibration":
                if state == "before":
                    return 2.0
                if state == "fast":
                    return 1.0
            return original_evaluate(
                trainer_arg,
                records,
                state=state,
                candidate=candidate,
            )

        with mock.patch.object(
            system,
            "_make_candidates",
            side_effect=capture_candidates,
        ), mock.patch.object(
            system,
            "_collect_activations",
            side_effect=identity_activations,
        ), mock.patch.object(
            system,
            "_evaluate_state",
            side_effect=positive_calibration_gain,
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertEqual(report["fd_psc_outcome"], FDPSCState.REJECT_NO_PROPOSAL.value)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["commit_query_access_count"], 0)
        self.assertEqual(len(candidates_seen), 1)
        candidate = candidates_seen[0]
        self.assertTrue(all(rank == 1 for rank in candidate.selected_rank_by_layer.values()))
        self.assertIn("rank_cap_failed", candidate.screening_reason)
        for logical_id, adapter in sorted(system.injection.adapters.items()):
            expected_b, expected_a = slow_before[logical_id]
            self.assertTrue(torch.equal(adapter.slow_B, expected_b))
            self.assertTrue(torch.equal(adapter.slow_A, expected_a))
        self.assertTheta0Bitwise(theta0)

    def test_planner_normal_return_sleeps_exactly_once_without_cleanup_abort(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.25)
        system = trainer.fd_psc_system
        support_obs, actions = _support_segment()
        initial_obs = {
            "visual": support_obs["visual"][:, 0],
            "proprio": support_obs["proprio"][:, 0],
        }
        goal_obs = {
            "visual": support_obs["visual"][:, 1],
            "proprio": support_obs["proprio"][:, 1],
        }
        planner = object.__new__(AdaJEPAMPCPlanner)
        planner.fd_psc_system = system
        planner.adajepa_trainer = trainer
        planner.evaluator = SimpleNamespace(
            seed=[23],
            state_0=torch.zeros(1, 1),
            state_g=torch.ones(1, 1),
            context_identifiers=["ctx"],
        )
        planner.env = object()
        planner._adajepa_loss_records = []
        planner._records = []
        planner.iter = 0
        planner.is_success = torch.tensor([False])
        planner._dump_per_iter_logs = mock.Mock()
        planner._dump_adajepa_loss_csv = mock.Mock()

        def successful_plan(**_kwargs):
            system.register_support_segment(support_obs, actions, iteration=0)
            trainer.finetune([support_obs], [actions])
            planner._obs_buffer.append(support_obs)
            planner._act_buffer.append(actions)
            return actions, np.asarray([1], dtype=np.int64)

        planner._plan_single = successful_plan
        with mock.patch(
            "planning.adajepa_mpc._EnvWorkerProxy",
            return_value=object(),
        ), mock.patch.object(
            trainer,
            "end_fd_psc_episode",
            wraps=trainer.end_fd_psc_episode,
        ) as sleep, mock.patch.object(
            trainer,
            "abort_fd_psc_episode",
            wraps=trainer.abort_fd_psc_episode,
        ) as abort:
            planned_actions, action_lengths = planner.plan(initial_obs, goal_obs)

        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(abort.call_count, 0)
        self.assertTrue(torch.equal(planned_actions, actions))
        self.assertTrue(np.array_equal(action_lengths, np.asarray([1])))
        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertEqual(
            [record["event"] for record in planner._records],
            ["fd_psc_sleep"],
        )
        self.assertTheta0Bitwise(theta0)

    def test_section_28_metrics_emit_values_and_explicit_availability(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.0,
            steps=2,
            config_overrides={
                "gates": {"plasticity_enabled": True},
                "external_eval_data": {
                    "plasticity_support_path": str(
                        self.split_paths["plasticity_support"]
                    ),
                    "plasticity_query_path": str(
                        self.split_paths["plasticity_query"]
                    ),
                },
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
            metadata={
                "sample_idx": 0,
                "seed": 17,
                "jepa_loss_threshold": 1.0e9,
            },
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        system._emit_context_retention_metrics(
            {"ctx-a": 2.0, "ctx-b": 1.0},
            {"ctx-a": 3.0, "ctx-b": 0.5},
        )
        system._emit_plasticity_gate_ratio(2.0, 1.0)

        events = system.metrics.events()
        route_id = next(event for event in events if event.name == "routed_exception_id")
        route_similarity = next(
            event for event in events if event.name == "routed_exception_similarity"
        )
        self.assertIsNone(route_id.value)
        self.assertEqual(route_id.tags["status"], "not_applicable")
        self.assertIsNone(route_similarity.value)
        self.assertEqual(route_similarity.tags["status"], "unavailable")

        early = [
            event
            for event in events
            if event.name == "next_episode_early_loss_decline"
            and event.tags["status"] == "available"
        ]
        threshold = [
            event
            for event in events
            if event.name == "time_to_threshold_replans"
            and event.tags["status"] == "available"
        ]
        self.assertTrue(early)
        self.assertEqual(threshold[-1].value, 1)

        context_losses = [event for event in events if event.name == "per_context_loss"]
        self.assertEqual(len(context_losses), 4)
        forgetting = [event for event in events if event.name == "forgetting"][-1]
        backward_transfer = [
            event for event in events if event.name == "backward_transfer"
        ][-1]
        ratio = [event for event in events if event.name == "plasticity_gate_ratio"][-1]
        self.assertEqual(forgetting.value, 1.0)
        self.assertEqual(backward_transfer.value, -0.25)
        self.assertEqual(ratio.value, 0.5)
        self.assertEqual(ratio.tags["status"], "available")

        system.finish_episode_without_sleep("section_28_metrics_verified")
        self.assertTheta0Bitwise(theta0)

    def test_context_is_never_inferred_from_a_single_calibration_context(self):
        _, trainer, _ = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        with self.assertRaisesRegex(FDPSCIntegrationError, "episode context is missing"):
            system.resolve_context_identifier({"sample_idx": 0, "seed": 17})

        self.assertEqual(
            system.resolve_context_identifier({"context_identifier": "ctx"}),
            "ctx",
        )

    def test_missing_required_plasticity_context_fails_before_episode_activation(self):
        split_path = self.split_paths["plasticity_query"]
        split = json.loads(split_path.read_text(encoding="utf-8"))
        record = split["records"][0]
        record["context_identifier"] = "other"
        split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["splits"]["plasticity_query"]["sha256"] = _sha256_file(split_path)
        manifest["contexts"]["ctx"].pop("plasticity_query")
        manifest["contexts"]["other"] = {
            "plasticity_query": [record["record_id"]]
        }
        self.manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        _, trainer, _ = self._trainer(
            learning_rate=0.0,
            config_overrides={
                "gates": {"plasticity_enabled": True},
                "external_eval_data": {
                    "plasticity_support_path": str(
                        self.split_paths["plasticity_support"]
                    ),
                    "plasticity_query_path": str(
                        self.split_paths["plasticity_query"]
                    ),
                },
            },
        )
        system = trainer.fd_psc_system
        obs, _ = _support_segment()
        with self.assertRaisesRegex(
            ExternalDataError,
            "required external splits: .*plasticity_query",
        ):
            trainer.begin_fd_psc_episode(
                system.next_episode_id,
                "ctx",
                initial_obs=obs,
            )
        self.assertIsNone(system._active_episode_id)
        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)

    def test_zero_update_sleep_never_opens_commit_query(self):
        model, trainer, theta0 = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        losses = trainer.finetune([obs], [actions])
        self.assertEqual(len(losses), 1)
        self.assertEqual(system.state_machine.online_update_count, 1)

        report = trainer.end_fd_psc_episode([obs], [actions])
        self.assertEqual(report["commit_query_access_count"], 0)
        self.assertEqual(report["candidate_count"], 0)
        self.assertIn("REJECT_NO_PROPOSAL", report["fd_psc_outcome"])
        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertTheta0Bitwise(theta0)
        self.assertTrue(model.training is False or not model.training)

    def test_preserve_runtime_keeps_optimizer_parameters_live_and_trainable(self):
        model, trainer, theta0 = self._trainer(learning_rate=0.1)
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)

        entries = system.injection.manifest.by_logical_id()
        predictor_id = next(
            logical_id
            for logical_id, entry in sorted(entries.items())
            if entry.module_group == "predictor"
        )
        system.injection.adapters[predictor_id].activate_centered_branch(rank=1)
        system.state_machine.activate_centered("preserve-runtime-test")

        system.prepare_online_mode(predictor_train=True, encoder_train=True)
        try:
            optimizer = trainer._make_optimizer()
            owned = tuple(
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            owned_ids = {id(parameter) for parameter in owned}
            owned_slots = {
                (logical_id, name): parameter
                for logical_id, adapter in sorted(system.injection.adapters.items())
                for name in ("pilot_A", "pilot_B", "center_A", "center_B")
                for parameter in (getattr(adapter, name),)
                if isinstance(parameter, nn.Parameter) and id(parameter) in owned_ids
            }
            self.assertTrue(owned_slots)
            self.assertTrue(all(parameter.requires_grad for parameter in owned))

            with system._preserve_adapter_runtime():
                # Fixed-split gradient collection does this while it installs
                # cloned adapter states for measurement.
                for parameter in model.parameters():
                    parameter.requires_grad_(False)

            for (logical_id, name), parameter in owned_slots.items():
                self.assertIs(
                    getattr(system.injection.adapters[logical_id], name),
                    parameter,
                )
                self.assertTrue(parameter.requires_grad)
            self.assertEqual(
                owned_ids,
                {
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                },
            )
            for parameter in model.parameters():
                if id(parameter) not in owned_ids:
                    self.assertFalse(parameter.requires_grad)
            system.assert_base_frozen()
        finally:
            system.finish_online_mode()
            system.abort_episode("preserve-runtime-test-cleanup")
        self.assertTheta0Bitwise(theta0)

    def test_first_multistep_trigger_rebuilds_for_real_centered_parameters(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.1, steps=2)
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)

        entries = system.injection.manifest.by_logical_id()
        predictor_id = next(
            logical_id
            for logical_id, entry in sorted(entries.items())
            if entry.module_group == "predictor"
        )
        adapter = system.injection.adapters[predictor_id]
        optimizers = []
        conflict_calls = []
        centered = {}
        original_make_optimizer = trainer._make_optimizer

        def tracked_make_optimizer():
            optimizer = original_make_optimizer()
            optimizers.append(optimizer)
            return optimizer

        def first_step_conflict(*_args, **_kwargs):
            conflict_calls.append(len(conflict_calls) + 1)
            if len(conflict_calls) != 1:
                return False
            reference = adapter._reference()
            b0 = torch.zeros(
                adapter.out_features,
                1,
                device=reference.device,
                dtype=reference.dtype,
            )
            a0 = torch.ones(
                1,
                adapter.in_features,
                device=reference.device,
                dtype=reference.dtype,
            )
            adapter.activate_centered_branch(b0, a0)
            system.state_machine.activate_centered("multistep-lifecycle-test")
            centered["B"] = adapter.center_B
            centered["B0"] = adapter.center_B.detach().clone()
            return True

        try:
            with mock.patch.object(
                trainer,
                "_make_optimizer",
                side_effect=tracked_make_optimizer,
            ), mock.patch.object(
                system,
                "_evaluate_conflict_trigger",
                side_effect=first_step_conflict,
            ):
                losses = trainer.finetune([obs], [actions])

            self.assertEqual(len(losses), 2)
            self.assertEqual(conflict_calls, [1, 2])
            self.assertEqual(len(optimizers), 2)
            self.assertIs(adapter.center_B, centered["B"])
            second_optimizer_ids = {
                id(parameter)
                for group in optimizers[1].param_groups
                for parameter in group["params"]
            }
            self.assertIn(id(adapter.center_A), second_optimizer_ids)
            self.assertIn(id(adapter.center_B), second_optimizer_ids)
            self.assertFalse(torch.equal(adapter.center_B.detach(), centered["B0"]))
            self.assertEqual(system._replan_index, 1)
        finally:
            system.abort_episode("multistep-lifecycle-test-cleanup")
        self.assertTheta0Bitwise(theta0)

    def test_short_contiguous_replans_form_one_complete_replay_window(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        system.wm.num_hist = 3
        segments, identities = _one_step_support_chain(3)
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=segments[0][0],
            metadata={"support_identities": identities},
        )
        try:
            for iteration, (obs, actions) in enumerate(segments):
                system.register_support_segment(obs, actions, iteration=iteration)
            eligible = system._eligible_replay_segments()
            self.assertEqual(len(eligible), 1)
            composed = eligible[0]
            self.assertEqual(tuple(composed.actions.shape[:2]), (1, 3))
            self.assertEqual(
                composed.identity.transition_ids,
                tuple(f"chain-transition-{index}" for index in range(3)),
            )
            self.assertEqual(
                composed.identity.frame_ids,
                tuple(f"chain-frame-{index}" for index in range(4)),
            )
            windows = system._build_replay_windows(trainer)
            self.assertEqual(len(windows), 1)
            self.assertEqual(tuple(windows[0].actions.shape[:2]), (1, 3))
            self.assertEqual(
                windows[0].metadata["source_support_record_ids"],
                [f"chain-support-{index}" for index in range(3)],
            )
            self.assertTrue(windows[0].metadata["composed_across_replans"])
        finally:
            system.abort_episode("short-contiguous-replay-test")
        self.assertTheta0Bitwise(theta0)

    def test_short_noncontiguous_replans_are_never_joined(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        system.wm.num_hist = 3
        segments, identities = _one_step_support_chain(
            3,
            identity_gap_before=2,
        )
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=segments[0][0],
            metadata={"support_identities": identities},
        )
        try:
            for iteration, (obs, actions) in enumerate(segments):
                system.register_support_segment(obs, actions, iteration=iteration)
            self.assertEqual(system._eligible_replay_segments(), ())
            self.assertEqual(system._build_replay_windows(trainer), [])
        finally:
            system.abort_episode("short-noncontiguous-replay-test")
        self.assertTheta0Bitwise(theta0)

    def test_claimed_contiguous_boundary_with_different_tensor_fails_closed(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        system.wm.num_hist = 3
        segments, identities = _one_step_support_chain(
            3,
            corrupt_tensor_boundary_before=1,
        )
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=segments[0][0],
            metadata={"support_identities": identities},
        )
        try:
            for iteration, (obs, actions) in enumerate(segments):
                system.register_support_segment(obs, actions, iteration=iteration)
            with self.assertRaisesRegex(
                FDPSCIntegrationError,
                "boundary frame content mismatch",
            ):
                system._eligible_replay_segments()
        finally:
            system.abort_episode("corrupt-contiguous-boundary-test")
        self.assertTheta0Bitwise(theta0)

    def test_composed_support_content_hash_is_audited_against_external_splits(self):
        segments, identities = _one_step_support_chain(3)
        composed_obs = {
            key: torch.cat(
                [
                    segments[0][0][key],
                    segments[1][0][key][:, 1:],
                    segments[2][0][key][:, 1:],
                ],
                dim=1,
            )
            for key in segments[0][0]
        }
        composed_actions = torch.cat([item[1] for item in segments], dim=1)
        colliding_payload = {
            "obs": {key: value.tolist() for key, value in composed_obs.items()},
            "actions": composed_actions.tolist(),
        }
        report_path = self.split_paths["report_test"]
        report_split = json.loads(report_path.read_text(encoding="utf-8"))
        report_record = report_split["records"][0]
        report_record["payload"] = colliding_payload
        report_record["content_hash"] = canonical_json_hash(colliding_payload)
        report_path.write_text(json.dumps(report_split, sort_keys=True), encoding="utf-8")
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["splits"]["report_test"]["sha256"] = _sha256_file(report_path)
        self.manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        _, trainer, theta0 = self._trainer(learning_rate=0.0)
        system = trainer.fd_psc_system
        system.wm.num_hist = 3
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=segments[0][0],
            metadata={"support_identities": identities},
        )
        try:
            for iteration, (obs, actions) in enumerate(segments):
                system.register_support_segment(obs, actions, iteration=iteration)
            with self.assertRaisesRegex(DataLeakageError, "content_hash"):
                system._eligible_replay_segments()
        finally:
            system.abort_episode("composed-support-leakage-test")
        self.assertTheta0Bitwise(theta0)

    def test_nonpositive_fast_gain_writes_nullable_candidate_report_and_rejects(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.25)
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        original_evaluate = system._evaluate_state

        def force_zero_fast_gain(
            trainer_arg,
            records,
            *,
            state,
            candidate=None,
        ):
            effective_state = "before" if state == "fast" else state
            return original_evaluate(
                trainer_arg,
                records,
                state=effective_state,
                candidate=candidate,
            )

        with mock.patch.object(
            system,
            "_evaluate_state",
            side_effect=force_zero_fast_gain,
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertEqual(report["fd_psc_outcome"], FDPSCState.REJECT_NO_PROPOSAL.value)
        self.assertEqual(report["commit_query_access_count"], 0)
        candidate_report = json.loads(
            (self.root / "fd_psc_candidates" / f"{episode_id}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreater(candidate_report["candidate_count"], 0)
        self.assertIsNone(candidate_report["candidates"][0]["calibration_loss"])
        self.assertIsNone(candidate_report["candidates"][0]["calibration_gain"])
        self.assertFalse(candidate_report["candidates"][0]["calibration_evaluated"])
        self.assertEqual(
            candidate_report["candidates"][0]["screening_reason"],
            "calibration_fast_gain_not_positive",
        )
        self.assertTheta0Bitwise(theta0)

    def test_proposal_sleep_consumes_exactly_one_query_and_keeps_theta0(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.25, steps=3)
        system = trainer.fd_psc_system
        episode_id = system.next_episode_id
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        before = trainer.evaluate_external_records(system.external.calibration("ctx"))
        trainer.finetune([obs], [actions])
        fast = trainer.evaluate_external_records(system.external.calibration("ctx"))
        self.assertLess(fast, before)

        report = trainer.end_fd_psc_episode([obs], [actions])
        self.assertGreater(report["candidate_count"], 0)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertEqual(system.external.commit_query_access_count(episode_id), 1)
        transitions = [
            item.new_state
            for item in system.state_machine.transition_log
            if item.episode_id == episode_id
        ]
        self.assertEqual(transitions.count(FDPSCState.FINAL_PROPOSAL_READY), 1)
        self.assertEqual(transitions.count(FDPSCState.FINAL_GATE), 1)
        with self.assertRaises(CommitQueryAccessError):
            system.external.issue_commit_query_token(episode_id, "second-proposal")
        self.assertTheta0Bitwise(theta0)

    def test_cold_start_spectral_drift_gate_treats_zero_before_as_zero(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides={
                "gates": {
                    "allow_unsafe_ablation": True,
                    "spectral_drift_enabled": True,
                    "drift_tolerance": 1.0,
                }
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        report = trainer.end_fd_psc_episode([obs], [actions])
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertEqual(report["gates"]["spectral_drift"]["status"], "pass")
        self.assertNotIn("ROLLBACK", report["fd_psc_outcome"])
        self.assertTheta0Bitwise(theta0)

    def test_planner_exception_abort_discards_episode_and_never_queries(self):
        _, trainer, theta0 = self._trainer(learning_rate=0.25)
        system = trainer.fd_psc_system
        episode_id = system.next_episode_id
        slow_before = {
            key: copy.deepcopy(adapter.adapter_state_dict())
            for key, adapter in system.injection.adapters.items()
        }
        support_obs, actions = _support_segment()
        initial_obs = {
            "visual": support_obs["visual"][:, 0],
            "proprio": support_obs["proprio"][:, 0],
        }
        goal_obs = {
            "visual": support_obs["visual"][:, 1],
            "proprio": support_obs["proprio"][:, 1],
        }
        planner = object.__new__(AdaJEPAMPCPlanner)
        planner.fd_psc_system = system
        planner.adajepa_trainer = trainer
        planner.evaluator = SimpleNamespace(
            seed=[23],
            state_0=torch.zeros(1, 1),
            state_g=torch.ones(1, 1),
            context_identifiers=["ctx"],
        )
        planner.env = object()
        planner._adajepa_loss_records = []
        planner._records = []

        def fail_after_update(**_kwargs):
            system.register_support_segment(support_obs, actions, iteration=0)
            trainer.finetune([support_obs], [actions])
            raise RuntimeError("synthetic planner exception")

        planner._plan_single = fail_after_update
        with mock.patch("planning.adajepa_mpc._EnvWorkerProxy", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "synthetic planner exception"):
                planner.plan(initial_obs, goal_obs)

        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertEqual(system.state_machine.rollback_count, 1)
        self.assertEqual(system.external.commit_query_access_count(episode_id), 0)
        for key, adapter in system.injection.adapters.items():
            expected = slow_before[key]
            current = adapter.adapter_state_dict()
            self.assertTrue(torch.equal(current["slow_B"], expected["slow_B"]))
            self.assertTrue(torch.equal(current["slow_A"], expected["slow_A"]))
            self.assertEqual(current["active_exception_id"], expected["active_exception_id"])
        self.assertTheta0Bitwise(theta0)

    def test_slow_commit_sidecar_round_trip_restores_memory_and_version(self):
        state_directory = self.root / "sidecar"
        latest_pointer = self.root / "sidecar-latest.json"
        checkpoint = {
            "enabled": True,
            "state_directory": str(state_directory),
            "latest_pointer_path": str(latest_pointer),
            "save_every_episodes": 1,
            "retention_versions": 3,
        }
        algorithm = {
            "checkpoint": checkpoint,
            "activation_subspace": {
                "enabled": True,
                "maximum_rank": 2,
                "spectral_energy_threshold": 0.99,
                "minimum_energy": 1.0e-8,
            },
            "merge": {
                "soft_ness_enabled": True,
                "shared_coefficients": [0.0, 1.0],
                "safe_coefficients": [1.0],
            },
        }
        _, first_trainer, first_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides=algorithm,
        )
        first = first_trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = first.next_episode_id
        first_trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        first.register_support_segment(obs, actions, iteration=0)
        first_trainer.finetune([obs], [actions])
        report = first_trainer.end_fd_psc_episode([obs], [actions])

        self.assertTrue(report["committed"])
        self.assertEqual(report["fd_psc_outcome"], FDPSCState.COMMIT_SLOW.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertTrue(latest_pointer.is_file())
        saved_state, saved_reference = first.checkpoints.load_latest()
        self.assertEqual(saved_reference.commit_id, "commit-00000001")
        self.assertEqual(saved_reference.commit_sequence, 1)
        self.assertEqual(saved_state["commit_sequence"], 1)
        self.assertEqual(
            saved_state["lifecycle"]["successful_slow_commit_count"], 1
        )
        self.assertEqual(saved_state["base_checkpoint_hash"], BASE_HASH)
        self.assertEqual(saved_state["target_manifest_hash"], first.target_manifest.hash)
        self.assertEqual(
            saved_state["lifecycle"]["transition_index"],
            first.state_machine._transition_index,
        )
        self.assertEqual(
            saved_state["metrics"]["sequence"], first.metrics.state_dict()["sequence"]
        )
        self.assertTrue(
            bitwise_state_equal(saved_state["external_data"], first.external.state_dict())
        )

        slow_before = {
            key: {
                "B": adapter.get_slow_factors().B.detach().clone(),
                "A": adapter.get_slow_factors().A.detach().clone(),
            }
            for key, adapter in first.injection.adapters.items()
        }
        replay_before = copy.deepcopy(first.replay.state_dict())
        subspaces_before = copy.deepcopy(first.subspaces.state_dict())
        self.assertEqual(len(first.replay), 1)
        self.assertEqual(set(subspaces_before), set(first.injection.adapters))
        self.assertTrue(
            all(value["q"].shape[1] > 0 for value in subspaces_before.values())
        )
        base_state_hash = first._compute_base_state_hash()
        self.assertTheta0Bitwise(first_theta0)

        resumed_checkpoint = {**checkpoint, "resume_path": str(latest_pointer)}
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides={**algorithm, "checkpoint": resumed_checkpoint},
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._commit_sequence, 1)
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed.state_machine.persistent_commit_count, 1)
        self.assertEqual(resumed._successful_slow_commit_count, 1)
        self.assertEqual(resumed.base_checkpoint_hash, first.base_checkpoint_hash)
        self.assertEqual(resumed._compute_base_state_hash(), base_state_hash)
        self.assertEqual(resumed.target_manifest.hash, first.target_manifest.hash)
        self.assertEqual(resumed.external.manifest_hash, first.external.manifest_hash)
        for key, adapter in resumed.injection.adapters.items():
            self.assertTrue(
                torch.equal(adapter.get_slow_factors().B, slow_before[key]["B"])
            )
            self.assertTrue(
                torch.equal(adapter.get_slow_factors().A, slow_before[key]["A"])
            )
        self.assertTrue(bitwise_state_equal(resumed.replay.state_dict(), replay_before))
        self.assertTrue(
            bitwise_state_equal(resumed.subspaces.state_dict(), subspaces_before)
        )
        resumed_state, resumed_reference = resumed.checkpoints.load_latest()
        self.assertEqual(resumed_reference, saved_reference)
        self.assertEqual(resumed_state["commit_sequence"], resumed._commit_sequence)
        self.assertTheta0Bitwise(resumed_theta0)

    def test_noncommit_episode_snapshots_restore_route_usage_ledger_and_rng(self):
        state_directory = self.root / "episode-snapshot-state"
        latest_pointer = self.root / "episode-snapshot-latest.json"
        checkpoint = {
            "enabled": True,
            "state_directory": str(state_directory),
            "latest_pointer_path": str(latest_pointer),
            "save_every_episodes": 1,
            "retention_versions": 6,
        }
        algorithm = {
            "checkpoint": checkpoint,
            "exception": {
                "enabled": True,
                "maximum_adapters": 2,
                "maximum_rank": 2,
                "local_replay_windows": 2,
            },
        }
        _, trainer, theta0 = self._trainer(
            learning_rate=0.0,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        adapter_id = self._install_zero_exception(system, obs)

        reject_episode = system.next_episode_id
        begin = trainer.begin_fd_psc_episode(
            reject_episode,
            "ctx",
            initial_obs=obs,
        )
        self.assertEqual(begin["route_adapter_id"], adapter_id)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        reject_report = trainer.end_fd_psc_episode([obs], [actions])
        self.assertEqual(
            reject_report["fd_psc_outcome"],
            FDPSCState.REJECT_NO_PROPOSAL.value,
        )

        reject_state, reject_reference = system.checkpoints.load_latest()
        self.assertEqual(
            reject_reference.commit_id,
            "snapshot-episode-00000001",
        )
        self.assertEqual(reject_reference.commit_sequence, 0)
        self.assertEqual(reject_state["episode_sequence"], 1)
        self.assertEqual(reject_state["commit_sequence"], 0)
        self.assertEqual(
            reject_state["lifecycle"]["persistent_commit_count"],
            0,
        )
        self.assertEqual(reject_state["exception_router"]["usage_clock"], 2)
        self.assertEqual(
            reject_state["exception_router"]["records"][adapter_id]["usage_count"],
            2,
        )
        self.assertEqual(reject_state["external_data"]["query_consumed"], {})
        self.assertTrue(
            bitwise_state_equal(
                reject_state["exception_router"],
                system.exception_router.state_dict(),
            )
        )
        self.assertTrue(
            bitwise_state_equal(
                reject_state["external_data"],
                system.external.state_dict(),
            )
        )
        self.assertEqual(
            reject_state["metrics"],
            system.metrics.state_dict(),
        )
        self.assertTrue(
            bitwise_state_equal(reject_state["rng"], RNGSnapshot.capture())
        )

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.0,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed._commit_sequence, 0)
        self.assertEqual(resumed.state_machine.persistent_commit_count, 0)
        self.assertTrue(
            bitwise_state_equal(
                resumed.exception_router.state_dict(),
                reject_state["exception_router"],
            )
        )
        self.assertTrue(
            bitwise_state_equal(
                resumed.external.state_dict(),
                reject_state["external_data"],
            )
        )
        self.assertTrue(
            bitwise_state_equal(reject_state["rng"], RNGSnapshot.capture())
        )

        abort_episode = resumed.next_episode_id
        begin = resumed_trainer.begin_fd_psc_episode(
            abort_episode,
            "ctx",
            initial_obs=obs,
        )
        self.assertEqual(begin["route_adapter_id"], adapter_id)
        resumed.abort_episode("snapshot_abort_test")
        abort_state, abort_reference = resumed.checkpoints.load_latest()
        self.assertEqual(
            abort_reference.commit_id,
            "snapshot-episode-00000002",
        )
        self.assertEqual(abort_state["episode_sequence"], 2)
        self.assertEqual(abort_state["exception_router"]["usage_clock"], 3)
        self.assertEqual(
            abort_state["exception_router"]["records"][adapter_id]["usage_count"],
            3,
        )
        self.assertEqual(abort_state["lifecycle"]["rollback_count"], 1)
        self.assertEqual(
            abort_state["lifecycle"]["persistent_commit_count"],
            0,
        )

        no_sleep_episode = resumed.next_episode_id
        begin = resumed_trainer.begin_fd_psc_episode(
            no_sleep_episode,
            "ctx",
            initial_obs=obs,
        )
        self.assertEqual(begin["route_adapter_id"], adapter_id)
        no_sleep = resumed.finish_episode_without_sleep("snapshot_no_sleep_test")
        self.assertEqual(no_sleep["fd_psc_outcome"], "NO_SLEEP")
        no_sleep_state, no_sleep_reference = resumed.checkpoints.load_latest()
        self.assertEqual(
            no_sleep_reference.commit_id,
            "snapshot-episode-00000003",
        )
        self.assertEqual(no_sleep_state["episode_sequence"], 3)
        self.assertEqual(no_sleep_state["exception_router"]["usage_clock"], 4)
        self.assertEqual(
            no_sleep_state["exception_router"]["records"][adapter_id]["usage_count"],
            4,
        )
        self.assertEqual(
            no_sleep_state["lifecycle"]["persistent_commit_count"],
            0,
        )
        stale_algorithm = copy.deepcopy(algorithm)
        stale_algorithm["checkpoint"]["resume_path"] = str(
            state_directory / reject_reference.version_file
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "explicit resume version is stale",
        ):
            self._trainer(
                learning_rate=0.0,
                config_overrides=stale_algorithm,
            )
        latest_version_algorithm = copy.deepcopy(algorithm)
        latest_version_algorithm["checkpoint"]["resume_path"] = str(
            state_directory / no_sleep_reference.version_file
        )
        _, latest_version_trainer, latest_version_theta0 = self._trainer(
            learning_rate=0.0,
            config_overrides=latest_version_algorithm,
        )
        self.assertEqual(
            latest_version_trainer.fd_psc_system._episode_sequence,
            3,
        )
        self.assertTheta0Bitwise(theta0)
        self.assertTheta0Bitwise(resumed_theta0)
        self.assertTheta0Bitwise(latest_version_theta0)

    def test_begin_route_apply_failure_is_snapshotted_and_abort_preserves_primary_error(self):
        state_directory = self.root / "begin-failure-state"
        latest_pointer = self.root / "begin-failure-latest.json"
        algorithm = {
            "exception": {
                "enabled": True,
                "maximum_adapters": 2,
                "maximum_rank": 2,
                "local_replay_windows": 2,
            },
            "checkpoint": {
                "enabled": True,
                "state_directory": str(state_directory),
                "latest_pointer_path": str(latest_pointer),
                "save_every_episodes": 1,
                "retention_versions": 4,
            },
        }
        _, trainer, theta0 = self._trainer(
            learning_rate=0.0,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, _ = _support_segment()
        adapter_id = self._install_zero_exception(system, obs)

        with mock.patch.object(
            system,
            "_apply_exception_state",
            side_effect=RuntimeError("fault-injected routed apply failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "routed apply failure",
            ):
                trainer.begin_fd_psc_episode(
                    system.next_episode_id,
                    "ctx",
                    initial_obs=obs,
                )

        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertEqual(system._episode_sequence, 1)
        self.assertEqual(system.state_machine.rollback_count, 1)
        saved_state, saved_reference = system.checkpoints.load_latest()
        self.assertEqual(
            saved_reference.commit_id,
            "snapshot-episode-00000001",
        )
        self.assertEqual(saved_state["episode_sequence"], 1)
        self.assertEqual(saved_state["lifecycle"]["rollback_count"], 1)
        self.assertEqual(saved_state["exception_router"]["usage_clock"], 2)
        self.assertEqual(
            saved_state["exception_router"]["records"][adapter_id]["usage_count"],
            2,
        )
        self.assertTrue(
            bitwise_state_equal(saved_state["rng"], RNGSnapshot.capture())
        )

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.0,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed.state_machine.rollback_count, 1)
        self.assertEqual(resumed.exception_router.get(adapter_id).usage_count, 2)
        self.assertTrue(
            bitwise_state_equal(saved_state["rng"], RNGSnapshot.capture())
        )

        resumed_trainer.begin_fd_psc_episode(
            resumed.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        real_save = resumed.checkpoints.save_committed

        def fail_snapshot_after_prepared(state, **kwargs):
            with mock.patch.object(
                resumed.checkpoints,
                "_load_version",
                side_effect=CheckpointValidationError(
                    "secondary snapshot validation failure"
                ),
            ):
                return real_save(state, **kwargs)

        try:
            try:
                raise RuntimeError("primary planner failure")
            except RuntimeError:
                with mock.patch.object(
                    resumed.checkpoints,
                    "save_committed",
                    side_effect=fail_snapshot_after_prepared,
                ):
                    resumed.abort_episode("planner_exception_test")
                raise
        except RuntimeError as primary:
            self.assertEqual(str(primary), "primary planner failure")
            # Python 3.11+ also carries the secondary failure as an exception
            # note. The supported Python 3.9 runtime relies on the structured
            # diagnostics asserted below.
            if callable(getattr(primary, "add_note", None)):
                self.assertTrue(
                    any(
                        "episode snapshot failed" in note
                        for note in getattr(primary, "__notes__", ())
                    )
                )
        else:  # pragma: no cover - the primary error must always survive
            self.fail("abort snapshot failure masked the primary planner error")
        diagnostic_codes = {event.code for event in resumed.diagnostics.events()}
        self.assertIn("episode_snapshot_failed", diagnostic_codes)
        self.assertIn("abort_episode_snapshot_failed", diagnostic_codes)
        self.assertEqual(resumed.state_machine.state, FDPSCState.IDLE)
        self.assertEqual(
            resumed.checkpoints.read_journal(
                "snapshot-episode-00000002"
            )["status"],
            "aborted",
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "unresolved episode durability journals",
        ):
            self._trainer(
                learning_rate=0.0,
                config_overrides=resumed_algorithm,
            )
        tombstone_path = (
            resumed.checkpoints.state_directory
            / "journal-snapshot-episode-00000002.json"
        )
        prepared_tombstone = json.loads(
            tombstone_path.read_text(encoding="utf-8")
        )
        prepared_tombstone["status"] = "prepared"
        tombstone_path.write_text(
            json.dumps(prepared_tombstone, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "unresolved episode durability journals",
        ):
            self._trainer(
                learning_rate=0.0,
                config_overrides=resumed_algorithm,
            )
        self.assertTheta0Bitwise(theta0)
        self.assertTheta0Bitwise(resumed_theta0)

    def test_commit_checkpoint_failure_snapshots_consumed_query_and_route_usage(self):
        state_directory = self.root / "failed-commit-state"
        latest_pointer = self.root / "failed-commit-latest.json"
        algorithm = {
            "exception": {
                "enabled": True,
                "maximum_adapters": 2,
                "maximum_rank": 2,
                "local_replay_windows": 2,
            },
            "checkpoint": {
                "enabled": True,
                "state_directory": str(state_directory),
                "latest_pointer_path": str(latest_pointer),
                "save_every_episodes": 1,
                "retention_versions": 4,
            }
        }
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        adapter_id = self._install_zero_exception(system, obs)
        episode_id = system.next_episode_id
        begin = trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        self.assertEqual(begin["route_adapter_id"], adapter_id)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])

        manager = system.checkpoints
        real_save = manager.save_committed
        failed_commit_ids = []

        def fail_first_model_commit(state, **kwargs):
            commit_id = str(kwargs["commit_id"])
            if commit_id.startswith("commit-") and not failed_commit_ids:
                failed_commit_ids.append(commit_id)
                with mock.patch.object(
                    manager,
                    "_load_version",
                    side_effect=CheckpointValidationError(
                        "fault-injected immutable-version validation failure"
                    ),
                ):
                    return real_save(state, **kwargs)
            return real_save(state, **kwargs)

        with mock.patch.object(
            manager,
            "save_committed",
            side_effect=fail_first_model_commit,
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertFalse(report["committed"])
        self.assertEqual(report["fd_psc_outcome"], FDPSCState.REJECT_QUERY.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertIn("sidecar checkpoint commit failed", report["commit_error"])
        self.assertEqual(failed_commit_ids, ["commit-00000001"])
        self.assertEqual(
            manager.read_journal("commit-00000001")["status"],
            "aborted",
        )
        self.assertEqual(
            manager.next_available_id("commit-00000001"),
            "commit-00000001-attempt-00000002",
        )
        saved_state, saved_reference = manager.load_latest()
        self.assertEqual(
            saved_reference.commit_id,
            "snapshot-episode-00000001",
        )
        self.assertEqual(saved_state["commit_sequence"], 0)
        self.assertEqual(saved_state["episode_sequence"], 1)
        self.assertEqual(
            saved_state["lifecycle"]["persistent_commit_count"],
            0,
        )
        self.assertEqual(saved_state["exception_router"]["usage_clock"], 2)
        self.assertEqual(
            saved_state["exception_router"]["records"][adapter_id]["usage_count"],
            2,
        )
        self.assertIn(episode_id, saved_state["external_data"]["query_consumed"])
        self.assertNotIn(episode_id, saved_state["external_data"]["query_issued"])
        self.assertTrue(
            bitwise_state_equal(
                saved_state["commit_gates"],
                system.gates.state_dict(),
            )
        )
        self.assertEqual(
            set(saved_state["commit_gates"]["invocations"]),
            {episode_id},
        )
        self.assertEqual(
            len(saved_state["commit_gates"]["invocations"]),
            1,
        )
        self.assertTrue(
            bitwise_state_equal(saved_state["rng"], RNGSnapshot.capture())
        )
        self.assertTheta0Bitwise(theta0)

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed._commit_sequence, 0)
        self.assertEqual(resumed.state_machine.persistent_commit_count, 0)
        self.assertEqual(
            resumed.external.commit_query_access_count(episode_id),
            1,
        )
        self.assertEqual(
            resumed.exception_router.get(adapter_id).usage_count,
            2,
        )
        self.assertTrue(
            bitwise_state_equal(
                resumed.gates.state_dict(),
                saved_state["commit_gates"],
            )
        )
        self.assertTrue(
            bitwise_state_equal(saved_state["rng"], RNGSnapshot.capture())
        )
        self.assertTheta0Bitwise(resumed_theta0)

    def test_gate7_precommit_failure_rolls_back_all_persistent_state(self):
        canary_manifest = _write_canary_manifest(self.root)
        requests = []

        def evaluator(request):
            requests.append(request)
            self.assertNotIn("external_data", request.state)
            self.assertNotIn("query_token", request.state)
            return {"success": request.state_label == "before"}

        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides={
                "canary": {
                    "enabled": True,
                    "every_episodes": 1,
                    "rollout_count": 1,
                    "manifest_path": str(canary_manifest),
                    "unavailable_policy": "error",
                },
                "checkpoint": {
                    "enabled": True,
                    "state_directory": str(self.root / "pre-canary-state"),
                    "latest_pointer_path": str(self.root / "pre-canary-latest.json"),
                    "retention_versions": 2,
                },
            },
        )
        system = trainer.fd_psc_system
        persistent_before = {
            key: copy.deepcopy(adapter.adapter_state_dict())
            for key, adapter in system.injection.adapters.items()
        }
        replay_before = copy.deepcopy(system.replay.state_dict())
        router_before = copy.deepcopy(system.exception_router.state_dict())
        subspaces_before = copy.deepcopy(system.subspaces.state_dict())
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        with mock.patch.object(system, "_canary_high_risk", return_value=True):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertFalse(report["committed"])
        self.assertEqual(report["fd_psc_outcome"], FDPSCState.REJECT_QUERY.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertIn("Gate-7 pre_commit canary failed", report["commit_error"])
        self.assertEqual([item.state_label for item in requests], ["before", "candidate"])
        self.assertEqual(system._commit_sequence, 0)
        self.assertTrue(bitwise_state_equal(system.replay.state_dict(), replay_before))
        self.assertTrue(bitwise_state_equal(system.exception_router.state_dict(), router_before))
        self.assertTrue(bitwise_state_equal(system.subspaces.state_dict(), subspaces_before))
        for key, adapter in system.injection.adapters.items():
            current = adapter.adapter_state_dict()
            expected = persistent_before[key]
            self.assertTrue(torch.equal(current["slow_B"], expected["slow_B"]))
            self.assertTrue(torch.equal(current["slow_A"], expected["slow_A"]))
        self.assertTheta0Bitwise(theta0)

    def test_gate7_periodic_postcommit_failure_rolls_back_candidate_updates(self):
        canary_manifest = _write_canary_manifest(self.root)
        requests = []

        def evaluator(request):
            requests.append(request)
            post_candidate = (
                request.state_label == "candidate"
                and request.state["commit_sequence"] == 1
            )
            return {"success": not post_candidate}

        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides={
                "canary": {
                    "enabled": True,
                    "every_episodes": 1,
                    "rollout_count": 1,
                    "manifest_path": str(canary_manifest),
                    "unavailable_policy": "error",
                },
                "checkpoint": {
                    "enabled": True,
                    "state_directory": str(self.root / "post-canary-state"),
                    "latest_pointer_path": str(self.root / "post-canary-latest.json"),
                    "retention_versions": 2,
                },
            },
        )
        system = trainer.fd_psc_system
        replay_before = copy.deepcopy(system.replay.state_dict())
        subspaces_before = copy.deepcopy(system.subspaces.state_dict())
        slow_before = {
            key: (
                adapter.get_slow_factors().B.detach().clone(),
                adapter.get_slow_factors().A.detach().clone(),
            )
            for key, adapter in system.injection.adapters.items()
        }
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertFalse(report["committed"])
        self.assertIn("Gate-7 post_commit canary failed", report["commit_error"])
        self.assertEqual(
            [item.state_label for item in requests],
            ["before", "candidate", "before", "candidate"],
        )
        self.assertEqual(requests[0].state["commit_sequence"], 0)
        self.assertEqual(requests[1].state["commit_sequence"], 0)
        self.assertEqual(requests[2].state["commit_sequence"], 0)
        self.assertEqual(requests[3].state["commit_sequence"], 1)
        # Periodic rollback publishes rollback-00000001 and retains 1 as the
        # commit/journal high-water mark even though memory returned to zero.
        self.assertEqual(system._commit_sequence, 1)
        self.assertEqual(system._successful_slow_commit_count, 0)
        self.assertEqual(len(system.replay), 0)
        self.assertTrue(bitwise_state_equal(system.replay.state_dict(), replay_before))
        self.assertTrue(bitwise_state_equal(system.subspaces.state_dict(), subspaces_before))
        for key, adapter in system.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.get_slow_factors().B, slow_before[key][0]))
            self.assertTrue(torch.equal(adapter.get_slow_factors().A, slow_before[key][1]))
        self.assertTheta0Bitwise(theta0)

    def test_gate7_periodic_k3_failure_restores_known_good_period_and_journals(self):
        canary_manifest = _write_canary_manifest(self.root)
        requests = []

        def evaluator(request):
            requests.append(request)
            regressed_period = (
                request.state_label == "candidate"
                and request.state["commit_sequence"] == 3
            )
            return {"success": not regressed_period}

        state_directory = self.root / "canary-period-state"
        latest_pointer = self.root / "canary-period-latest.json"
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides={
                "canary": {
                    "enabled": True,
                    "every_episodes": 3,
                    "rollout_count": 1,
                    "manifest_path": str(canary_manifest),
                    "unavailable_policy": "error",
                },
                "checkpoint": {
                    "enabled": True,
                    "state_directory": str(state_directory),
                    "latest_pointer_path": str(latest_pointer),
                    "save_every_episodes": 1,
                    "retention_versions": 5,
                },
            },
        )
        system = trainer.fd_psc_system
        replay_good = copy.deepcopy(system.replay.state_dict())
        router_good = copy.deepcopy(system.exception_router.state_dict())
        subspaces_good = copy.deepcopy(system.subspaces.state_dict())
        slow_good = {
            key: (
                adapter.get_slow_factors().B.detach().clone(),
                adapter.get_slow_factors().A.detach().clone(),
            )
            for key, adapter in system.injection.adapters.items()
        }

        reports = []
        for index in range(3):
            obs, actions = _support_segment()
            trainer.begin_fd_psc_episode(
                system.next_episode_id,
                "ctx",
                initial_obs=obs,
                metadata={"trajectory_id": f"period-trajectory-{index}"},
            )
            system.register_support_segment(obs, actions, iteration=0)
            trainer.finetune([obs], [actions])
            reports.append(trainer.end_fd_psc_episode([obs], [actions]))

        self.assertTrue(reports[0]["committed"])
        self.assertTrue(reports[1]["committed"])
        self.assertFalse(reports[2]["committed"])
        self.assertEqual(reports[0]["commit_id"], "commit-00000001")
        self.assertEqual(reports[1]["commit_id"], "commit-00000002")
        self.assertEqual(reports[2]["rollback_commit_id"], "rollback-00000003")
        self.assertEqual(
            reports[2]["reverted_commit_ids"],
            ["commit-00000001", "commit-00000002", "commit-00000003"],
        )
        self.assertEqual(system._commit_sequence, 3)
        self.assertEqual(system._episode_sequence, 3)
        self.assertEqual(system.state_machine.persistent_commit_count, 0)
        self.assertEqual(system._successful_slow_commit_count, 0)
        self.assertEqual(system.state_machine.rollback_count, 1)
        self.assertEqual(system._canary_known_good["commit_sequence"], 0)
        self.assertEqual(system._canary_pending_commit_ids, [])
        self.assertEqual(len(system.gates.state_dict()["invocations"]), 3)
        self.assertTrue(bitwise_state_equal(system.replay.state_dict(), replay_good))
        self.assertTrue(bitwise_state_equal(system.exception_router.state_dict(), router_good))
        self.assertTrue(bitwise_state_equal(system.subspaces.state_dict(), subspaces_good))
        for key, adapter in system.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.get_slow_factors().B, slow_good[key][0]))
            self.assertTrue(torch.equal(adapter.get_slow_factors().A, slow_good[key][1]))

        # High-risk pre-commit checks may add earlier pairs; the scheduled
        # period comparison itself must be known-good(0) versus cumulative(3).
        self.assertEqual(requests[-2].state_label, "before")
        self.assertEqual(requests[-2].state["commit_sequence"], 0)
        self.assertEqual(requests[-1].state_label, "candidate")
        self.assertEqual(requests[-1].state["commit_sequence"], 3)

        saved_state, saved_reference = system.checkpoints.load_latest()
        self.assertEqual(saved_reference.commit_id, "rollback-00000003")
        self.assertEqual(saved_reference.commit_sequence, 3)
        self.assertEqual(saved_state["commit_sequence"], 3)
        self.assertEqual(
            saved_state["canary_period"]["known_good"]["commit_sequence"], 0
        )
        self.assertEqual(saved_state["canary_period"]["pending_commit_ids"], [])
        self.assertEqual(saved_state["lifecycle"]["persistent_commit_count"], 0)
        self.assertEqual(
            saved_state["lifecycle"]["successful_slow_commit_count"], 0
        )
        self.assertEqual(saved_state["lifecycle"]["rollback_count"], 1)
        for commit_id in (
            "commit-00000001",
            "commit-00000002",
            "commit-00000003",
        ):
            journal = system.checkpoints.read_journal(commit_id)
            self.assertEqual(journal["status"], "rolled_back")
            self.assertEqual(journal["rollback_id"], "rollback-00000003")
            version_file = journal.get("version_file")
            if version_file is not None:
                with self.assertRaises(CheckpointValidationError):
                    system.checkpoints.load_version(state_directory / version_file)
        self.assertTheta0Bitwise(theta0)

    def test_gate7_known_good_period_survives_resume_before_k3_failure(self):
        canary_manifest = _write_canary_manifest(self.root)

        def evaluator(request):
            return {
                "success": not (
                    request.state_label == "candidate"
                    and request.state["commit_sequence"] == 3
                )
            }

        state_directory = self.root / "canary-resume-state"
        latest_pointer = self.root / "canary-resume-latest.json"
        common = {
            "canary": {
                "enabled": True,
                "every_episodes": 3,
                "rollout_count": 1,
                "manifest_path": str(canary_manifest),
                "unavailable_policy": "error",
            },
            "checkpoint": {
                "enabled": True,
                "state_directory": str(state_directory),
                "latest_pointer_path": str(latest_pointer),
                "retention_versions": 5,
            },
        }
        _, trainer, _ = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides=common,
        )
        system = trainer.fd_psc_system
        slow_good = {
            key: (
                adapter.get_slow_factors().B.detach().clone(),
                adapter.get_slow_factors().A.detach().clone(),
            )
            for key, adapter in system.injection.adapters.items()
        }
        replay_good = copy.deepcopy(system.replay.state_dict())
        for index in range(2):
            obs, actions = _support_segment()
            trainer.begin_fd_psc_episode(
                system.next_episode_id,
                "ctx",
                initial_obs=obs,
                metadata={"trajectory_id": f"resume-period-{index}"},
            )
            system.register_support_segment(obs, actions, iteration=0)
            trainer.finetune([obs], [actions])
            self.assertTrue(trainer.end_fd_psc_episode([obs], [actions])["committed"])
        self.assertEqual(system._canary_known_good["commit_sequence"], 0)
        self.assertEqual(system._successful_slow_commit_count, 2)
        self.assertEqual(
            system._canary_pending_commit_ids,
            ["commit-00000001", "commit-00000002"],
        )

        resumed_overrides = copy.deepcopy(common)
        resumed_overrides["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides=resumed_overrides,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._commit_sequence, 2)
        self.assertEqual(resumed._episode_sequence, 2)
        self.assertEqual(resumed._canary_known_good["commit_sequence"], 0)
        self.assertEqual(resumed._successful_slow_commit_count, 2)
        self.assertEqual(
            resumed._canary_pending_commit_ids,
            ["commit-00000001", "commit-00000002"],
        )

        obs, actions = _support_segment()
        resumed_trainer.begin_fd_psc_episode(
            resumed.next_episode_id,
            "ctx",
            initial_obs=obs,
            metadata={"trajectory_id": "resume-period-2"},
        )
        resumed.register_support_segment(obs, actions, iteration=0)
        resumed_trainer.finetune([obs], [actions])
        report = resumed_trainer.end_fd_psc_episode([obs], [actions])
        self.assertFalse(report["committed"])
        self.assertEqual(report["rollback_commit_id"], "rollback-00000003")
        self.assertEqual(resumed._commit_sequence, 3)
        self.assertEqual(resumed._episode_sequence, 3)
        self.assertEqual(resumed._successful_slow_commit_count, 0)
        self.assertEqual(resumed._canary_pending_commit_ids, [])
        self.assertTrue(bitwise_state_equal(resumed.replay.state_dict(), replay_good))
        for key, adapter in resumed.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.get_slow_factors().B, slow_good[key][0]))
            self.assertTrue(torch.equal(adapter.get_slow_factors().A, slow_good[key][1]))

        rollback_state, rollback_reference = resumed.checkpoints.load_latest()
        self.assertEqual(rollback_reference.commit_id, "rollback-00000003")
        self.assertEqual(
            rollback_state["metrics"]["sequence"],
            resumed.metrics.state_dict()["sequence"],
        )
        self.assertEqual(
            rollback_state["commit_gates"], resumed.gates.state_dict()
        )
        self.assertEqual(
            rollback_state["external_data"]["query_consumed"],
            resumed.external.state_dict()["query_consumed"],
        )

        # A second process resumes the rollback checkpoint itself, proving the
        # high-water ID and known-good metadata are not merely live attributes.
        _, verified_trainer, verified_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            canary_evaluator=evaluator,
            config_overrides=resumed_overrides,
        )
        verified = verified_trainer.fd_psc_system
        self.assertEqual(verified._commit_sequence, 3)
        self.assertEqual(verified._episode_sequence, 3)
        self.assertEqual(verified._successful_slow_commit_count, 0)
        self.assertEqual(verified._canary_known_good["commit_sequence"], 0)
        self.assertEqual(verified._canary_pending_commit_ids, [])
        self.assertEqual(len(verified.gates.state_dict()["invocations"]), 3)
        self.assertTrue(bitwise_state_equal(verified.replay.state_dict(), replay_good))
        for key, adapter in verified.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.get_slow_factors().B, slow_good[key][0]))
            self.assertTrue(torch.equal(adapter.get_slow_factors().A, slow_good[key][1]))
        self.assertTheta0Bitwise(resumed_theta0)
        self.assertTheta0Bitwise(verified_theta0)

    def test_accumulate_baseline_retains_one_pilot_across_episodes_without_sleep(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides={"run_mode": "accumulate"},
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()

        first_episode = system.next_episode_id
        trainer.begin_fd_psc_episode(first_episode, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        retained = {
            key: (adapter.pilot_B.detach().clone(), adapter.pilot_A.detach().clone())
            for key, adapter in system.injection.adapters.items()
        }
        first_report = trainer.end_fd_psc_episode([obs], [actions])
        system.reset_episode()
        self.assertEqual(first_report["fd_psc_outcome"], "NO_SLEEP")
        self.assertEqual(first_report["commit_query_access_count"], 0)
        self.assertEqual(system.external.commit_query_access_count(first_episode), 0)

        second_episode = system.next_episode_id
        trainer.begin_fd_psc_episode(second_episode, "ctx", initial_obs=obs)
        for key, adapter in system.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.pilot_B, retained[key][0]))
            self.assertTrue(torch.equal(adapter.pilot_A, retained[key][1]))
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        second_report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertEqual(second_report["fd_psc_outcome"], "NO_SLEEP")
        self.assertEqual(second_report["commit_query_access_count"], 0)
        self.assertEqual(system.external.commit_query_access_count(second_episode), 0)
        self.assertEqual(system._commit_sequence, 0)
        self.assertEqual(system.state_machine.persistent_commit_count, 0)
        self.assertEqual(len(system.replay), 0)
        self.assertFalse(system.subspaces.state_dict())
        for adapter in system.injection.adapters.values():
            self.assertEqual(adapter.get_slow_factors().rank, 0)
            self.assertGreater(float(torch.linalg.vector_norm(adapter.pilot_B)), 0.0)
        for episode_id in (first_episode, second_episode):
            transitions = [
                item.new_state
                for item in system.state_machine.transition_log
                if item.episode_id == episode_id
            ]
            self.assertNotIn(FDPSCState.SLEEP_CALIBRATION, transitions)
            self.assertNotIn(FDPSCState.COMMIT_SLOW, transitions)
            self.assertNotIn(FDPSCState.FINAL_GATE, transitions)
        self.assertTheta0Bitwise(theta0)

    def test_accumulate_sidecar_restores_the_persistent_adapter(self):
        state_directory = self.root / "accumulate-state"
        latest_pointer = self.root / "accumulate-latest.json"
        checkpoint = {
            "enabled": True,
            "state_directory": str(state_directory),
            "latest_pointer_path": str(latest_pointer),
            "save_every_episodes": 1,
            "retention_versions": 3,
        }
        algorithm = {"run_mode": "accumulate", "checkpoint": checkpoint}
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        trainer.end_fd_psc_episode([obs], [actions])

        retained = system._adapter_participant.state_dict()
        self.assertTrue(
            any(
                float(torch.linalg.vector_norm(adapter.pilot_B)) > 0.0
                for adapter in system.injection.adapters.values()
            )
        )
        saved_state, saved_reference = system.checkpoints.load_latest()
        self.assertEqual(
            saved_reference.commit_id,
            "snapshot-episode-00000001",
        )
        self.assertTrue(
            bitwise_state_equal(
                saved_state["accumulate_adapter_state"],
                retained,
            )
        )
        missing_persistent_adapter = copy.deepcopy(saved_state)
        missing_persistent_adapter.pop("accumulate_adapter_state")
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "missing the persistent adapter state",
        ):
            system._validate_checkpoint_state(missing_persistent_adapter)

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed._commit_sequence, 0)
        self.assertTrue(
            bitwise_state_equal(
                resumed._adapter_participant.state_dict(),
                retained,
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for adapter in resumed.injection.adapters.values()
                for parameter in adapter.trainable_episode_parameters()
            )
        )

        resumed_trainer.begin_fd_psc_episode(
            resumed.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        self.assertTrue(
            bitwise_state_equal(
                resumed._adapter_participant.state_dict(),
                retained,
            )
        )
        resumed.abort_episode("accumulate-resume-test-cleanup")
        self.assertTheta0Bitwise(theta0)
        self.assertTheta0Bitwise(resumed_theta0)

    def test_plain_svd_baseline_merges_directly_without_query_or_gates(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides={"run_mode": "plain_svd"},
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])

        with mock.patch.object(
            system.external,
            "issue_commit_query_token",
            side_effect=AssertionError("plain_svd touched commit-query"),
        ), mock.patch.object(
            system.gates,
            "evaluate_once",
            side_effect=AssertionError("plain_svd invoked commit gates"),
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertTrue(report["committed"])
        self.assertEqual(report["fd_psc_outcome"], FDPSCState.COMMIT_SLOW.value)
        self.assertEqual(report["commit_query_access_count"], 0)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(system.external.commit_query_access_count(episode_id), 0)
        self.assertEqual(system._commit_sequence, 1)
        self.assertEqual(system.state_machine.persistent_commit_count, 1)
        self.assertEqual(system.state_machine.final_gate_count, 0)
        self.assertEqual(system.state_machine.final_proposal_count, 1)
        self.assertEqual(len(system.replay), 0)
        self.assertFalse(system.subspaces.state_dict())
        self.assertTrue(
            any(adapter.get_slow_factors().rank > 0 for adapter in system.injection.adapters.values())
        )
        transitions = [
            item.new_state
            for item in system.state_machine.transition_log
            if item.episode_id == episode_id
        ]
        self.assertEqual(transitions.count(FDPSCState.SLEEP_CALIBRATION), 1)
        self.assertEqual(transitions.count(FDPSCState.COMMIT_SLOW), 1)
        self.assertNotIn(FDPSCState.FINAL_GATE, transitions)
        self.assertTheta0Bitwise(theta0)

    def test_plain_svd_commit_is_durable_before_terminal_cleanup(self):
        state_directory = self.root / "plain-svd-state"
        latest_pointer = self.root / "plain-svd-latest.json"
        checkpoint = {
            "enabled": True,
            "state_directory": str(state_directory),
            "latest_pointer_path": str(latest_pointer),
            "save_every_episodes": 1,
            "retention_versions": 3,
        }
        algorithm = {"run_mode": "plain_svd", "checkpoint": checkpoint}
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])
        report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertTrue(report["committed"])
        self.assertEqual(report["commit_id"], "commit-00000001")
        saved_state, saved_reference = system.checkpoints.load_latest()
        self.assertEqual(saved_reference.commit_id, "commit-00000001")
        self.assertEqual(saved_state["commit_sequence"], 1)
        self.assertEqual(
            saved_state["lifecycle"]["persistent_commit_count"],
            1,
        )
        self.assertEqual(
            saved_state["metrics"]["sequence"],
            system.metrics.state_dict()["sequence"],
        )
        slow_after = {
            key: (
                adapter.get_slow_factors().B.detach().clone(),
                adapter.get_slow_factors().A.detach().clone(),
            )
            for key, adapter in system.injection.adapters.items()
        }
        self.assertEqual(len(tuple(state_directory.glob("state-*.pt"))), 1)

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed._episode_sequence, 1)
        self.assertEqual(resumed._commit_sequence, 1)
        self.assertEqual(resumed.state_machine.persistent_commit_count, 1)
        for key, adapter in resumed.injection.adapters.items():
            self.assertTrue(torch.equal(adapter.get_slow_factors().B, slow_after[key][0]))
            self.assertTrue(torch.equal(adapter.get_slow_factors().A, slow_after[key][1]))
        self.assertTheta0Bitwise(theta0)
        self.assertTheta0Bitwise(resumed_theta0)

    def test_plain_svd_checkpoint_failure_rolls_back_live_commit_state(self):
        state_directory = self.root / "plain-svd-failure-state"
        latest_pointer = self.root / "plain-svd-failure-latest.json"
        algorithm = {
            "run_mode": "plain_svd",
            "checkpoint": {
                "enabled": True,
                "state_directory": str(state_directory),
                "latest_pointer_path": str(latest_pointer),
                "save_every_episodes": 1,
                "retention_versions": 3,
            },
        }
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        system.register_support_segment(obs, actions, iteration=0)
        trainer.finetune([obs], [actions])

        adapter_before = system._adapter_participant.state_dict()
        counters_before = {
            "episode_sequence": system._episode_sequence,
            "commit_sequence": system._commit_sequence,
            "successful_slow_commit_count": system._successful_slow_commit_count,
        }
        machine_before = system.state_machine.state_dict()
        rng_before = RNGSnapshot.capture()
        with mock.patch.object(
            system.checkpoints,
            "save_committed",
            side_effect=OSError("fault-injected plain-SVD checkpoint failure"),
        ):
            with self.assertRaisesRegex(OSError, "plain-SVD checkpoint failure"):
                trainer.end_fd_psc_episode([obs], [actions])

        self.assertTrue(
            bitwise_state_equal(
                system._adapter_participant.state_dict(),
                adapter_before,
            )
        )
        self.assertEqual(
            {
                "episode_sequence": system._episode_sequence,
                "commit_sequence": system._commit_sequence,
                "successful_slow_commit_count": system._successful_slow_commit_count,
            },
            counters_before,
        )
        self.assertTrue(
            bitwise_state_equal(system.state_machine.state_dict(), machine_before)
        )
        self.assertTrue(bitwise_state_equal(RNGSnapshot.capture(), rng_before))
        self.assertFalse(latest_pointer.exists())

        system.abort_episode("plain-svd-checkpoint-failure-test-cleanup")
        self.assertEqual(system.state_machine.state, FDPSCState.IDLE)
        self.assertTheta0Bitwise(theta0)

    def test_global_repair_success_short_circuits_new_exception_path(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides={
                "repair": {
                    "enabled": True,
                    "maximum_steps": 2,
                    "candidate_steps": [1, 2],
                    "windows_per_batch": 1,
                    "proximal_enabled": False,
                    "pcgrad_enabled": False,
                },
                "exception": {
                    "enabled": True,
                    "maximum_adapters": 2,
                    "maximum_rank": 2,
                    "local_replay_windows": 2,
                },
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        calibration = system.external.calibration("ctx")
        loss_before = trainer.evaluate_external_records(calibration)
        trainer.finetune([obs], [actions])
        loss_fast = trainer.evaluate_external_records(calibration)
        self.assertGreater(loss_before - loss_fast, system.config.gates.absolute_numerical_tolerance)

        events = []
        real_make_candidates = system._make_candidates

        def observe_make(proposal_type, *args, **kwargs):
            events.append(f"make:{proposal_type.value}")
            return real_make_candidates(proposal_type, *args, **kwargs)

        def reject_quick(_trainer, candidates, **kwargs):
            proposal_type = candidates[0].proposal_type
            events.append(f"screen:{proposal_type.value}:reject")
            for candidate in candidates:
                candidate.calibration_loss = float(kwargs["calibration_before"])
                candidate.calibration_gain = 0.0
                candidate.screening_reason = "fault_injected_quick_rejection"
            return None, [candidate.summary() for candidate in candidates]

        def successful_repair(_trainer, seed_candidate, **kwargs):
            events.append(f"repair:{seed_candidate.proposal_type.value}:success")
            system.state_machine.enter_repair("fault_injected_global_repair")
            repaired = copy.deepcopy(seed_candidate)
            repaired.calibration_loss = float(kwargs["calibration_fast"])
            repaired.calibration_gain = float(
                kwargs["calibration_before"] - kwargs["calibration_fast"]
            )
            repaired.repair_step = 1
            repaired.screening_reason = "fault_injected_repair_success"
            return repaired

        with mock.patch.object(
            system, "_make_candidates", side_effect=observe_make
        ), mock.patch.object(
            system, "_screen_candidates", side_effect=reject_quick
        ), mock.patch.object(
            system, "_repair_candidate", side_effect=successful_repair
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertTrue(report["committed"])
        self.assertEqual(report["proposal_type"], ProposalType.GLOBAL_SLOW.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertEqual(
            events,
            [
                "make:global_slow",
                "screen:global_slow:reject",
                "repair:global_slow:success",
            ],
        )
        self.assertFalse(any("new_exception" in event for event in events))
        self.assertEqual(len(system.exception_router), 0)
        self.assertTheta0Bitwise(theta0)

    def test_failed_global_repair_then_valid_fast_gain_generates_exception(self):
        state_directory = self.root / "exception-sidecar"
        latest_pointer = self.root / "exception-latest.json"
        algorithm = {
            "repair": {
                "enabled": True,
                "maximum_steps": 2,
                "candidate_steps": [1, 2],
                "windows_per_batch": 1,
                "proximal_enabled": False,
                "pcgrad_enabled": False,
            },
            "exception": {
                "enabled": True,
                "maximum_adapters": 2,
                "minimum_calibration_fast_gain": 1.0e-4,
                "maximum_rank": 2,
                "local_replay_windows": 2,
            },
            "checkpoint": {
                "enabled": True,
                "state_directory": str(state_directory),
                "latest_pointer_path": str(latest_pointer),
                "save_every_episodes": 1,
                "retention_versions": 3,
            },
        }
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides=algorithm,
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        calibration = system.external.calibration("ctx")
        loss_before = trainer.evaluate_external_records(calibration)
        trainer.finetune([obs], [actions])
        loss_fast = trainer.evaluate_external_records(calibration)
        fast_gain = loss_before - loss_fast
        self.assertGreater(fast_gain, system.config.exception.minimum_calibration_fast_gain)

        events = []
        real_make_candidates = system._make_candidates

        def observe_make(proposal_type, *args, **kwargs):
            events.append(f"make:{proposal_type.value}")
            candidates = real_make_candidates(proposal_type, *args, **kwargs)
            if proposal_type == ProposalType.NEW_EXCEPTION:
                # Keep this test's sparse-exception restore coverage explicit.
                # Small-but-nonzero spectra are no longer (incorrectly)
                # canonicalized to rank zero by an absolute energy cutoff.
                zero_layer = sorted(candidates[0].factors_by_layer)[0]
                for candidate in candidates:
                    factors = candidate.factors_by_layer[zero_layer]
                    candidate.factors_by_layer[zero_layer] = LowRankFactors.zeros(
                        factors.out_features,
                        factors.in_features,
                        device=factors.b.device,
                        dtype=factors.b.dtype,
                    )
                    candidate.selected_rank_by_layer[zero_layer] = 0
                    candidate.functional_error_by_layer[zero_layer] = 0.0
            return candidates

        def screen_by_path(_trainer, candidates, **kwargs):
            proposal_type = candidates[0].proposal_type
            if proposal_type == ProposalType.GLOBAL_SLOW:
                events.append("screen:global_slow:reject")
                for candidate in candidates:
                    candidate.calibration_loss = float(kwargs["calibration_before"])
                    candidate.calibration_gain = 0.0
                    candidate.screening_reason = "fault_injected_quick_rejection"
                return None, [candidate.summary() for candidate in candidates]
            self.assertEqual(proposal_type, ProposalType.NEW_EXCEPTION)
            events.append("screen:new_exception:select")
            selected = candidates[0]
            selected.calibration_loss = float(kwargs["calibration_fast"])
            selected.calibration_gain = float(
                kwargs["calibration_before"] - kwargs["calibration_fast"]
            )
            selected.screening_reason = "fault_injected_exception_selection"
            return selected, [selected.summary()]

        def failed_repair(_trainer, seed_candidate, **_kwargs):
            events.append(f"repair:{seed_candidate.proposal_type.value}:failed")
            system.state_machine.enter_repair("fault_injected_global_repair_failure")
            return None

        with mock.patch.object(
            system, "_make_candidates", side_effect=observe_make
        ), mock.patch.object(
            system, "_screen_candidates", side_effect=screen_by_path
        ), mock.patch.object(
            system, "_repair_candidate", side_effect=failed_repair
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertIn("committed", report, msg=(report, events, fast_gain))
        self.assertTrue(report["committed"])
        self.assertEqual(report["proposal_type"], ProposalType.NEW_EXCEPTION.value)
        self.assertEqual(report["commit_query_access_count"], 1)
        self.assertEqual(
            events,
            [
                "make:global_slow",
                "screen:global_slow:reject",
                "repair:global_slow:failed",
                "make:new_exception",
                "screen:new_exception:select",
            ],
        )
        self.assertEqual(len(system.exception_router), 1)
        self.assertEqual(len(system.replay), 0)
        self.assertTheta0Bitwise(theta0)

        adapter_id = system.exception_router.adapter_ids()[0]
        stored = system.exception_router.get(adapter_id).adapter_state["layers"]
        zero_layers = {
            logical_id
            for logical_id, factors in stored.items()
            if int(factors["B"].shape[1]) == 0
        }
        self.assertTrue(zero_layers)
        self.assertTrue(latest_pointer.is_file())

        resumed_algorithm = copy.deepcopy(algorithm)
        resumed_algorithm["checkpoint"]["resume_path"] = str(latest_pointer)
        _, resumed_trainer, resumed_theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides=resumed_algorithm,
        )
        resumed = resumed_trainer.fd_psc_system
        self.assertEqual(resumed.exception_router.adapter_ids(), (adapter_id,))
        begin = resumed_trainer.begin_fd_psc_episode(
            resumed.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        self.assertEqual(begin["route_adapter_id"], adapter_id)
        for logical_id, adapter in resumed.injection.adapters.items():
            if logical_id in zero_layers:
                self.assertEqual(adapter.get_exception_factors().rank, 0)
                self.assertIsNone(adapter.active_exception_id)
            else:
                self.assertGreater(adapter.get_exception_factors().rank, 0)
                self.assertEqual(adapter.active_exception_id, adapter_id)
        resumed.finish_episode_without_sleep("sparse_exception_restore_verified")
        self.assertTheta0Bitwise(resumed_theta0)

    def test_repair_trajectory_uses_support_and_replay_and_screens_only_clones(self):
        engine = RepairEngine(
            maximum_steps=3,
            candidate_steps=[1, 2, 3],
            windows_per_batch=2,
            current_weight=1.0,
            replay_weight=1.0,
            proximal_enabled=False,
            proximal_weight=0.0,
            pcgrad_enabled=True,
            seed=41,
            sampling="balanced_uniform",
        )
        current = [
            SimpleNamespace(window_id="support-0"),
            SimpleNamespace(window_id="support-1"),
        ]
        replay = [_replay_window(index, context) for context, index in (("a", 0), ("b", 0))]
        trajectory = []
        screening = []

        def train_step(live, batch, step, pcgrad):
            trajectory.append(
                {
                    "object_id": id(live),
                    "value_before": live["value"],
                    "step": step,
                    "current": tuple(item.window_id for item in batch.current),
                    "replay": tuple(item.window_id for item in batch.replay),
                    "weights": dict(batch.normalized_weights),
                    "pcgrad": pcgrad,
                }
            )
            live["value"] += 1
            live["optimizer_moments"].append(step)
            return {"trajectory_value": live["value"]}

        def screen_candidate(candidate, step):
            screening.append((step, candidate["value"], id(candidate)))
            # Deliberately destructive recompression/screening mutation. It
            # must remain confined to this checkpoint clone.
            candidate["value"] += 1000
            candidate["optimizer_moments"].append(f"screen-{step}")
            return ScreeningResult(step == 3, {"screen_step": float(step)}, "injected")

        result = engine.run(
            {"value": 0, "optimizer_moments": []},
            current_windows=current,
            replay_windows=replay,
            train_step=train_step,
            screen_candidate=screen_candidate,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.selected_step, 3)
        self.assertEqual([item["value_before"] for item in trajectory], [0, 1, 2])
        self.assertEqual(len({item["object_id"] for item in trajectory}), 1)
        self.assertEqual([item[:2] for item in screening], [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(len({item[2] for item in screening}), 3)
        for item in trajectory:
            self.assertEqual(len(item["current"]), 2)
            self.assertEqual(len(item["replay"]), 2)
            self.assertEqual(item["weights"], {"current": 0.5, "replay": 0.5})
            self.assertTrue(item["pcgrad"])
        self.assertEqual(
            [checkpoint.candidate_state["value"] for checkpoint in result.checkpoints],
            [1001, 1002, 1003],
        )
        self.assertEqual(
            [len(checkpoint.step_metrics) for checkpoint in result.checkpoints],
            [1, 2, 3],
        )
        self.assertEqual(result.final_state["value"], 1003)
        self.assertEqual(result.final_state["optimizer_moments"][:3], [1, 2, 3])

    def test_failed_repair_does_not_generate_exception_below_fast_gain_threshold(self):
        _, trainer, theta0 = self._trainer(
            learning_rate=0.25,
            steps=3,
            config_overrides={
                "repair": {
                    "enabled": True,
                    "maximum_steps": 1,
                    "candidate_steps": [1],
                    "windows_per_batch": 1,
                    "proximal_enabled": False,
                    "pcgrad_enabled": False,
                },
                "exception": {
                    "enabled": True,
                    "maximum_adapters": 2,
                    "minimum_calibration_fast_gain": 1.0,
                    "maximum_rank": 2,
                    "local_replay_windows": 2,
                },
            },
        )
        system = trainer.fd_psc_system
        obs, actions = _support_segment()
        episode_id = system.next_episode_id
        trainer.begin_fd_psc_episode(episode_id, "ctx", initial_obs=obs)
        system.register_support_segment(obs, actions, iteration=0)
        calibration = system.external.calibration("ctx")
        loss_before = trainer.evaluate_external_records(calibration)
        trainer.finetune([obs], [actions])
        loss_fast = trainer.evaluate_external_records(calibration)
        fast_gain = loss_before - loss_fast
        self.assertGreater(fast_gain, system.config.gates.absolute_numerical_tolerance)
        self.assertLess(fast_gain, system.config.exception.minimum_calibration_fast_gain)

        events = []
        real_make_candidates = system._make_candidates

        def observe_make(proposal_type, *args, **kwargs):
            events.append(f"make:{proposal_type.value}")
            return real_make_candidates(proposal_type, *args, **kwargs)

        def reject_global(_trainer, candidates, **kwargs):
            self.assertEqual(candidates[0].proposal_type, ProposalType.GLOBAL_SLOW)
            events.append("screen:global_slow:reject")
            for candidate in candidates:
                candidate.calibration_loss = float(kwargs["calibration_before"])
                candidate.calibration_gain = 0.0
                candidate.screening_reason = "fault_injected_quick_rejection"
            return None, [candidate.summary() for candidate in candidates]

        def failed_repair(_trainer, seed_candidate, **_kwargs):
            events.append(f"repair:{seed_candidate.proposal_type.value}:failed")
            system.state_machine.enter_repair("fault_injected_global_repair_failure")
            return None

        with mock.patch.object(
            system, "_make_candidates", side_effect=observe_make
        ), mock.patch.object(
            system, "_screen_candidates", side_effect=reject_global
        ), mock.patch.object(
            system, "_repair_candidate", side_effect=failed_repair
        ):
            report = trainer.end_fd_psc_episode([obs], [actions])

        self.assertEqual(report["fd_psc_outcome"], FDPSCState.REJECT_NO_PROPOSAL.value)
        self.assertEqual(report["commit_query_access_count"], 0)
        self.assertEqual(
            events,
            [
                "make:global_slow",
                "screen:global_slow:reject",
                "repair:global_slow:failed",
            ],
        )
        self.assertEqual(len(system.exception_router), 0)
        self.assertEqual(system.external.commit_query_access_count(episode_id), 0)
        self.assertTheta0Bitwise(theta0)

    def test_balanced_uniform_repair_is_reproducible_context_balanced_and_reports_duplicates(self):
        replay = (
            [_replay_window(index, "a") for index in range(3)]
            + [_replay_window(0, "b")]
            + [_replay_window(index, "c") for index in range(2)]
        )

        def run_once(seed):
            engine = RepairEngine(
                maximum_steps=2,
                candidate_steps=[2],
                windows_per_batch=6,
                current_weight=1.0,
                replay_weight=1.0,
                proximal_enabled=False,
                proximal_weight=0.0,
                pcgrad_enabled=False,
                seed=seed,
                sampling="balanced_uniform",
            )
            batches = []

            def train_step(_state, batch, _step, _pcgrad):
                batches.append(
                    (
                        tuple(item.window_id for item in batch.replay),
                        tuple(item.context_identifier for item in batch.replay),
                        batch.duplicate_rate,
                    )
                )
                return {}

            result = engine.run(
                {"unused": 0},
                current_windows=[SimpleNamespace(window_id="support")],
                replay_windows=replay,
                train_step=train_step,
                screen_candidate=lambda _candidate, _step: ScreeningResult(False),
            )
            return engine, batches, result

        first_engine, first_batches, first_result = run_once(73)
        _, second_batches, second_result = run_once(73)
        self.assertEqual(first_batches, second_batches)
        self.assertTrue(bitwise_state_equal(first_result, second_result))
        for identifiers, contexts, duplicate_rate in first_batches:
            counts = {context: contexts.count(context) for context in set(contexts)}
            self.assertEqual(counts, {"a": 2, "b": 2, "c": 2})
            expected_rate = 1.0 - len(set(identifiers)) / len(identifiers)
            self.assertAlmostEqual(duplicate_rate, expected_rate)
            self.assertAlmostEqual(duplicate_rate, 1.0 / 6.0)
        checkpoint_metrics = first_result.checkpoints[0].step_metrics
        self.assertEqual(len(checkpoint_metrics), 2)
        self.assertTrue(
            all(
                abs(item["grasp_duplicate_rate"] - 1.0 / 6.0) < 1.0e-12
                for item in checkpoint_metrics
            )
        )

        restored = RepairEngine(
            maximum_steps=2,
            candidate_steps=[2],
            windows_per_batch=6,
            current_weight=1.0,
            replay_weight=1.0,
            proximal_enabled=False,
            proximal_weight=0.0,
            pcgrad_enabled=False,
            seed=73,
            sampling="balanced_uniform",
        )
        restored.load_state_dict(first_engine.state_dict())
        original_next = first_engine._sample_balanced_uniform(replay, step_index=2)
        restored_next = restored._sample_balanced_uniform(replay, step_index=2)
        self.assertEqual(
            tuple(item.window_id for item in original_next.windows),
            tuple(item.window_id for item in restored_next.windows),
        )
        self.assertEqual(original_next.duplicate_rate, restored_next.duplicate_rate)


if __name__ == "__main__":
    unittest.main()
