from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fd_psc.state_machine import ProposalType
from fd_psc.trainer import _Candidate
from planning.adajepa import AdaJEPATrainer
from tests.test_fd_psc_integration import (
    _ToyWorldModel,
    _config,
    _support_segment,
    _write_fixed_manifest,
)


class PlasticityProbeFairnessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.split_paths = _write_fixed_manifest(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _trainer(self, *, steps=2, always_on_sdc=False):
        overrides = {
            "gates": {
                "allow_unsafe_ablation": True,
                "plasticity_enabled": True,
            },
            "external_eval_data": {
                "plasticity_support_path": str(
                    self.split_paths["plasticity_support"]
                ),
                "plasticity_query_path": str(
                    self.split_paths["plasticity_query"]
                ),
            },
        }
        if always_on_sdc:
            overrides.update(
                {
                    "gradient_geometry": {
                        "enabled": True,
                        "current_batches": 1,
                        "history_batches": 1,
                        "anchor_batches": 1,
                        "windows_per_batch": 1,
                    },
                    "sdc": {
                        "enabled": True,
                        "event_triggered": False,
                    },
                    "anchor_data": {
                        "windows": 1,
                        "manifest_path": str(self.manifest),
                        "data_path": str(self.split_paths["anchor"]),
                    },
                }
            )
        trainer = AdaJEPATrainer(
            wm=_ToyWorldModel(),
            lr=0.2,
            steps=steps,
            optimizer_name="sgd",
            finetune_encoder=True,
            last_layer_only=False,
            encoder_lr=0.07,
            encoder_last_layer_only=False,
            fd_psc=_config(self.manifest, self.split_paths, **overrides),
            runtime_output_dir=str(self.root),
        )
        obs, _ = _support_segment()
        trainer.begin_fd_psc_episode(
            trainer.fd_psc_system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        return trainer

    @staticmethod
    def _identity_candidate(system) -> _Candidate:
        factors = system._proposal_base_factors(ProposalType.GLOBAL_SLOW)
        cloned = {
            key: type(value)(
                value.B.detach().clone(),
                value.A.detach().clone(),
            )
            for key, value in factors.items()
        }
        return _Candidate(
            proposal_type=ProposalType.GLOBAL_SLOW,
            factors_by_layer=cloned,
            task_factors_by_layer=copy.deepcopy(cloned),
            functional_error_by_layer={key: 0.0 for key in cloned},
            selected_rank_by_layer={key: value.rank for key, value in cloned.items()},
            alpha_shared=0.731,
            alpha_safe=0.913,
            spectral_variant="identity-test",
        )

    def test_before_and_candidate_use_identical_pilot_rng_and_optimizer(self):
        trainer = self._trainer(steps=3)
        system = trainer.fd_psc_system
        phases = []
        original_generator = system._episode_generator

        def record_generator(logical_id, phase="pilot"):
            generator = original_generator(logical_id, phase)
            phases.append((logical_id, phase, generator.get_state().clone()))
            return generator

        optimizer_configs = []
        real_sgd = torch.optim.SGD

        def record_optimizer(groups, *args, **kwargs):
            optimizer = real_sgd(groups, *args, **kwargs)
            optimizer_configs.append(
                {
                    "defaults": copy.deepcopy(optimizer.defaults),
                    "lrs": tuple(group["lr"] for group in optimizer.param_groups),
                    "sizes": tuple(
                        sum(parameter.numel() for parameter in group["params"])
                        for group in optimizer.param_groups
                    ),
                }
            )
            return optimizer

        before_outer_rng = torch.random.get_rng_state().clone()
        with mock.patch.object(
            system,
            "_episode_generator",
            side_effect=record_generator,
        ), mock.patch.object(torch.optim, "SGD", side_effect=record_optimizer):
            gain_before = system._plasticity_gain(trainer, state="before")
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_outer_rng))

            # Candidate screening may be separated by arbitrary stochastic
            # work. The paired probe must still rewind to the same RNG.
            torch.rand(19)
            candidate_outer_rng = torch.random.get_rng_state().clone()
            gain_candidate = system._plasticity_gain(
                trainer,
                state="candidate",
                candidate=self._identity_candidate(system),
            )
            self.assertTrue(torch.equal(torch.random.get_rng_state(), candidate_outer_rng))

        self.assertEqual(gain_before, gain_candidate)
        self.assertEqual(len(optimizer_configs), 2)
        self.assertEqual(optimizer_configs[0], optimizer_configs[1])
        self.assertEqual(optimizer_configs[0]["lrs"], (trainer.lr, trainer.encoder_lr))
        by_layer = {}
        for logical_id, phase, generator_state in phases:
            self.assertEqual(phase, "plasticity-probe-paired")
            by_layer.setdefault(logical_id, []).append(generator_state)
        self.assertTrue(by_layer)
        for states in by_layer.values():
            self.assertEqual(len(states), 2)
            self.assertTrue(torch.equal(states[0], states[1]))

        system.abort_episode("plasticity_fairness_test_complete")

    def test_screening_matches_gate4_negative_and_near_zero_fallback(self):
        trainer = self._trainer(steps=1)
        system = trainer.fd_psc_system
        self.assertFalse(system._plasticity_screening_feasible(-1.0, -0.1))
        self.assertFalse(system._plasticity_screening_feasible(0.0, -1.0e-3))
        self.assertTrue(system._plasticity_screening_feasible(-1.0, 0.0))
        self.assertTrue(system._plasticity_screening_feasible(0.0, 0.0))
        system.abort_episode("plasticity_screening_test_complete")

    def test_always_on_sdc_reuses_two_pass_helper_from_first_probe_step(self):
        trainer = self._trainer(steps=2, always_on_sdc=True)
        system = trainer.fd_psc_system
        with mock.patch.object(
            system,
            "backward_with_sdc",
            wraps=system.backward_with_sdc,
        ) as backward:
            gain = system._plasticity_gain(trainer, state="before")

        self.assertTrue(torch.isfinite(torch.tensor(gain)))
        self.assertEqual(backward.call_count, trainer.steps)
        first = backward.call_args_list[0]
        self.assertIsNotNone(first.kwargs["loss_closure"])
        self.assertIsNotNone(first.kwargs["forward_rng"])
        self.assertIsNone(system._online_hooks)
        self.assertTrue(all(system._sdc_active.values()))
        self.assertGreaterEqual(
            system.metrics.counter("sdc_two_pass_updates"),
            1,
        )

        system.abort_episode("plasticity_sdc_test_complete")


if __name__ == "__main__":
    unittest.main()
