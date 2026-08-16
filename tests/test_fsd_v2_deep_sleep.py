from __future__ import annotations

import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fd_psc.config import DeepSleepConfig, minimal_fsd_v2_config
from fd_psc.low_rank_merge import as_factors
from fd_psc.v2.deep_sleep import (
    DeepSleepController,
    FunctionalComparison,
    ResidualTarget,
    partition_residual_targets,
)
from tests.test_fsd_v2_no_external_data import (
    _support_segment,
    _theta0_snapshot,
    _trainer,
)


def _deep_config(
    *,
    checkpoint: bool = False,
    resume_path: str | None = None,
    maximum_steps: int = 20,
    functional_error_threshold: float = 10.0,
):
    return minimal_fsd_v2_config(
        seed=41,
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
            "budget_fraction_initial": 1.0,
            "budget_fraction_maximum": 1.0,
            "geometry_windows": 4,
            "geometry_maximum_rank": 2,
            "geometry_energy_threshold": 1.0,
        },
        deep_sleep={
            "enabled": True,
            "trigger_relative_rank_error": 0.0,
            "trigger_consecutive_commits": 1,
            "minimum_replay_windows": 1,
            "strategy": "residual_distill",
            "maximum_steps": maximum_steps,
            "learning_rate": 0.05,
            "output_residual_weight": 1.0,
            "hidden_residual_weight": 0.1,
            "current_task_weight": 0.1,
            "residual_rank": 1,
            "batch_size": 2,
            "hidden_layer_maximum": 4,
            "validation_fraction": 0.5,
            "minimum_validation_windows": 1,
            "functional_error_threshold": functional_error_threshold,
            "functional_error_absolute_tolerance": 0.0,
            "epsilon": 1.0e-8,
            "core_storage": {"mode": "dense_delta"},
        },
        checkpoint={
            "enabled": checkpoint,
            "state_directory": "fsd_v2_state",
            "latest_pointer_path": "fsd_v2_state_latest.json",
            "resume_path": resume_path,
            "save_every_episodes": 1,
            "retention_versions": 4,
        },
    )


def _force_overflow(system):
    original = system._compress

    def wrapped(accepted):
        compressed, diagnostics = original(accepted)
        forced = {
            key: dataclasses.replace(
                value,
                relative_discarded_frobenius=max(
                    value.relative_discarded_frobenius, 0.5
                ),
            )
            for key, value in diagnostics.items()
        }
        return compressed, forced

    return mock.patch.object(system, "_compress", side_effect=wrapped)


def _run_episode(trainer, episode: int, scale: float, *, force_overflow=False):
    obs, actions = _support_segment(scale)
    trainer.begin_fd_psc_episode(
        f"episode-{episode}",
        f"context-{episode}",
        metadata={"sample_idx": episode},
    )
    losses = trainer.finetune([obs], [actions])
    if not losses:
        raise AssertionError("wake update did not run")
    if force_overflow:
        with _force_overflow(trainer.fd_psc_system):
            return trainer.end_fd_psc_episode([obs], [actions])
    return trainer.end_fd_psc_episode([obs], [actions])


class DeepSleepControllerTests(unittest.TestCase):
    def test_trigger_requires_consecutive_overflow_and_checkpoint_restores_rng(self):
        config = DeepSleepConfig(
            trigger_relative_rank_error=0.1,
            trigger_consecutive_commits=3,
            minimum_replay_windows=2,
            minimum_validation_windows=2,
        )
        controller = DeepSleepController(config, seed=73)
        self.assertFalse(controller.should_trigger(0.2, available_windows=2))
        self.assertFalse(controller.should_trigger(0.2, available_windows=2))
        checkpoint = copy.deepcopy(controller.state_dict())
        expected_batch = controller.sample_indices(8, 3)

        restored = DeepSleepController(config, seed=73)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.sample_indices(8, 3), expected_batch)
        self.assertTrue(restored.should_trigger(0.2, available_windows=2))
        restored.mark_success()
        self.assertEqual(restored.consecutive_overflows, 0)

        self.assertFalse(controller.should_trigger(0.0, available_windows=2))
        self.assertEqual(controller.consecutive_overflows, 0)

    def test_held_out_split_is_deterministic_checkpointed_and_disjoint(self):
        config = DeepSleepConfig(
            validation_fraction=0.25,
            minimum_validation_windows=2,
        )
        controller = DeepSleepController(config, seed=79)
        checkpoint = copy.deepcopy(controller.state_dict())
        first = controller.split_fit_validation_indices(10)

        restored = DeepSleepController(config, seed=79)
        restored.load_state_dict(checkpoint)
        second = restored.split_fit_validation_indices(10)
        self.assertEqual(first, second)
        self.assertEqual(len(first.validation_indices), 3)
        self.assertFalse(set(first.fit_indices) & set(first.validation_indices))
        self.assertEqual(
            set(first.fit_indices) | set(first.validation_indices),
            set(range(10)),
        )

    def test_validation_targets_never_enter_optimizer_partition(self):
        def target(source, identifier):
            zero = torch.zeros(1)
            return ResidualTarget(
                source=source,
                payload={"window_id": identifier},
                core_output=zero,
                teacher_output_residual=zero,
                core_hidden={},
                teacher_hidden_residual={},
            )

        targets = tuple(target("raw_replay", f"history-{index}") for index in range(4)) + (
            target("current_support", "current-0"),
        )
        controller = DeepSleepController(
            DeepSleepConfig(
                validation_fraction=0.5,
                minimum_validation_windows=2,
            ),
            seed=83,
        )
        split = controller.split_fit_validation_indices(4)
        fit, validation = partition_residual_targets(
            targets,
            historical_count=4,
            split=split,
        )
        fit_ids = {item.payload["window_id"] for item in fit}
        validation_ids = {item.payload["window_id"] for item in validation}
        self.assertFalse(fit_ids & validation_ids)
        self.assertIn("current-0", fit_ids)
        self.assertTrue(all(item.source == "raw_replay" for item in validation))


class DeepSleepIntegrationTests(unittest.TestCase):
    def assert_theta0_unchanged(self, snapshot) -> None:
        for tensor, expected, parameter in snapshot:
            self.assertTrue(torch.equal(tensor.detach(), expected))
            if parameter:
                self.assertFalse(tensor.requires_grad)

    def test_real_deep_sleep_freezes_teacher_uses_internal_data_and_reclaims_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, trainer, theta0 = _trainer(
                root, _deep_config(checkpoint=True), seed=59
            )
            first = _run_episode(trainer, 0, 1.0)
            self.assertFalse(first["deep_sleep_triggered"])

            with mock.patch(
                "fd_psc.external_data.ExternalDataRegistry",
                side_effect=AssertionError("external data accessed by Deep Sleep"),
            ):
                second = _run_episode(
                    trainer, 1, 1.7, force_overflow=True
                )

            self.assertTrue(second["deep_sleep_triggered"])
            self.assertEqual(second["deep_sleep_status"], "committed")
            self.assertTrue(second["deep_sleep_teacher_frozen"])
            self.assertFalse(second["deep_sleep_rolled_back"])
            self.assertEqual(
                second["deep_sleep_source_counts"],
                {
                    "fit_current_support": 1,
                    "fit_raw_replay": 0,
                    "validation_raw_replay": 1,
                },
            )
            self.assertEqual(second["deep_sleep_fit_count"], 1)
            self.assertEqual(second["deep_sleep_validation_count"], 1)
            self.assertGreater(second["deep_sleep_output_residual_loss"], 0.0)
            self.assertGreaterEqual(second["deep_sleep_hidden_residual_loss"], 0.0)
            self.assertGreaterEqual(second["deep_sleep_current_jepa_loss"], 0.0)
            self.assertLessEqual(
                second["deep_sleep_final_functional_error"],
                trainer.fd_psc_system.config.deep_sleep.functional_error_threshold,
            )
            self.assertGreater(second["deep_sleep_core_write_frobenius"], 0.0)
            self.assertGreater(second["deep_sleep_rank_reclaimed"], 0)
            self.assertLessEqual(second["deep_sleep_residual_rank"], 1)
            for adapter in trainer.fd_psc_system.injection.adapters.values():
                self.assertLessEqual(as_factors(adapter.get_slow_factors()).rank, 1)
            self.assert_theta0_unchanged(theta0)

            before_resume_core = trainer.fd_psc_system._core_snapshot()
            before_resume_slow = trainer.fd_psc_system._slow_snapshot()
            before_resume_controller = copy.deepcopy(
                trainer.fd_psc_system.deep_sleep_controller.state_dict()
            )
            _model, resumed, resumed_theta0 = _trainer(
                root,
                _deep_config(
                    checkpoint=True,
                    resume_path="fsd_v2_state_latest.json",
                ),
                seed=59,
            )
            resumed_system = resumed.fd_psc_system
            for key, value in before_resume_core.items():
                self.assertTrue(
                    torch.equal(value, resumed_system._core_snapshot()[key])
                )
                before = before_resume_slow[key]
                after = resumed_system._slow_snapshot()[key]
                self.assertTrue(torch.equal(before.b, after.b))
                self.assertTrue(torch.equal(before.a, after.a))
            resumed_controller = resumed_system.deep_sleep_controller.state_dict()
            for name in (
                "consecutive_overflows",
                "observation_count",
                "trigger_count",
                "success_count",
                "last_relative_rank_error",
                "last_available_windows",
            ):
                self.assertEqual(
                    before_resume_controller[name], resumed_controller[name]
                )
            self.assertTrue(
                torch.equal(
                    before_resume_controller["generator_state"],
                    resumed_controller["generator_state"],
                )
            )
            self.assert_theta0_unchanged(resumed_theta0)
            self.assertIs(model.predictor, trainer.wm.predictor)

    def test_functional_rejection_rolls_back_core_controller_and_sampler_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _model, trainer, theta0 = _trainer(
                root,
                _deep_config(
                    checkpoint=False,
                    maximum_steps=1,
                    functional_error_threshold=0.0,
                ),
                seed=67,
            )
            _run_episode(trainer, 0, 1.0)
            system = trainer.fd_psc_system
            core_before = system._core_snapshot()
            controller_before = copy.deepcopy(
                system.deep_sleep_controller.state_dict()
            )
            budget_before = copy.deepcopy(system.budget_controller.state_dict())
            rejected = FunctionalComparison(
                relative_error=1.0,
                absolute_error=1.0,
                reference_norm=1.0,
                finite=True,
            )
            with mock.patch(
                "fd_psc.v2.system.compare_functional_outputs",
                return_value=rejected,
            ):
                second = _run_episode(
                    trainer, 1, 2.0, force_overflow=True
                )

            self.assertTrue(second["deep_sleep_triggered"])
            self.assertTrue(second["deep_sleep_rolled_back"])
            self.assertEqual(
                second["deep_sleep_status"], "functional_error_threshold"
            )
            self.assertEqual(second["deep_sleep_rank_reclaimed"], 0)
            self.assertEqual(second["deep_sleep_core_write_frobenius"], 0.0)
            for key, value in core_before.items():
                self.assertTrue(torch.equal(value, system._core_snapshot()[key]))
            controller_after = system.deep_sleep_controller.state_dict()
            for name in controller_before:
                if torch.is_tensor(controller_before[name]):
                    self.assertTrue(
                        torch.equal(controller_before[name], controller_after[name])
                    )
                else:
                    self.assertEqual(
                        controller_before[name], controller_after[name]
                    )
            budget_after = system.budget_controller.state_dict()
            self.assertEqual(
                budget_after["update_count"],
                budget_before["update_count"] + 1,
            )
            # The episode still commits the pre-existing normal compressed
            # fallback, exactly as the guide specifies for residual-refit
            # rejection.
            self.assertEqual(second["status"], "committed")
            self.assertEqual(system._commit_sequence, 2)
            self.assert_theta0_unchanged(theta0)


if __name__ == "__main__":
    unittest.main()
