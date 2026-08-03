from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from fd_psc.low_rank_merge import LowRankFactors
from fd_psc.state_machine import ProposalType
from fd_psc.trainer import _Candidate
from planning.adajepa import AdaJEPATrainer
from tests.test_fd_psc_integration import (
    _ToyWorldModel,
    _config,
    _support_segment,
    _write_fixed_manifest,
)


class CandidateStateSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest, split_paths = _write_fixed_manifest(self.root)
        model = _ToyWorldModel()
        self.trainer = AdaJEPATrainer(
            wm=model,
            lr=0.1,
            steps=1,
            optimizer_name="sgd",
            finetune_encoder=True,
            last_layer_only=False,
            encoder_lr=0.1,
            encoder_last_layer_only=False,
            fd_psc=_config(
                manifest,
                split_paths,
                slow_lora={
                    "initial_rank": 1,
                    "allowed_ranks": [1, 2],
                    "maximum_rank": 2,
                    "spectral_energy_threshold": 0.9,
                    "functional_error_threshold": 0.1,
                },
            ),
            runtime_output_dir=str(self.root),
        )
        self.system = self.trainer.fd_psc_system

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _rank_candidate(self, alpha: float) -> _Candidate:
        factors = {}
        references = {}
        tasks = {}
        ranks = {}
        errors = {}
        for logical_id, adapter in sorted(self.system.injection.adapters.items()):
            self.assertEqual((adapter.out_features, adapter.in_features), (2, 2))
            reference = LowRankFactors(
                torch.diag(torch.tensor([1.0, 0.2])),
                torch.eye(2),
            )
            factors[logical_id] = LowRankFactors(
                reference.B.clone(), reference.A.clone()
            )
            references[logical_id] = LowRankFactors(
                reference.B.clone(), reference.A.clone()
            )
            tasks[logical_id] = LowRankFactors(
                reference.B.clone(), reference.A.clone()
            )
            ranks[logical_id] = 2
            errors[logical_id] = 0.0
        return _Candidate(
            proposal_type=ProposalType.GLOBAL_SLOW,
            factors_by_layer=factors,
            task_factors_by_layer=tasks,
            functional_error_by_layer=errors,
            selected_rank_by_layer=ranks,
            alpha_shared=alpha,
            alpha_safe=1.0,
            spectral_variant="none",
            rank_reference_by_layer=references,
        )

    def test_each_candidate_uses_its_own_activation_rank_fixed_point(self) -> None:
        first = self._rank_candidate(0.25)
        second = self._rank_candidate(0.75)
        calls = []

        def candidate_activations(trainer, records, *, state, candidate):
            self.assertEqual(state, "candidate")
            calls.append(candidate)
            row = (
                torch.tensor([[1.0, 0.0]])
                if candidate is first
                else torch.tensor([[0.0, 1.0]])
            )
            return {
                key: row.clone() for key in self.system.injection.adapters
            }

        with mock.patch.object(
            self.system,
            "_collect_activations",
            side_effect=candidate_activations,
        ):
            self.assertTrue(
                self.system._stabilize_candidate_rank(
                    self.trainer, first, calibration=(object(),)
                )
            )
            self.assertTrue(
                self.system._stabilize_candidate_rank(
                    self.trainer, second, calibration=(object(),)
                )
            )

        self.assertTrue(all(rank == 1 for rank in first.selected_rank_by_layer.values()))
        self.assertTrue(all(rank == 2 for rank in second.selected_rank_by_layer.values()))
        call_ids = [id(value) for value in calls]
        self.assertIn(id(first), call_ids)
        self.assertIn(id(second), call_ids)
        self.assertGreaterEqual(call_ids.count(id(first)), 2)

    def test_rank_cap_failure_is_retained_as_bounded_repair_seed(self) -> None:
        self.system.config.merge.soft_ness_enabled = False
        self.system.config.slow_lora.allowed_ranks = [1]
        self.system.config.slow_lora.maximum_rank = 1
        self.system.config.slow_lora.spectral_energy_threshold = 0.99
        task = {
            logical_id: LowRankFactors(torch.eye(2), torch.eye(2))
            for logical_id in self.system.injection.adapters
        }
        activations = {
            logical_id: torch.eye(2)
            for logical_id in self.system.injection.adapters
        }

        candidates = self.system._make_candidates(
            ProposalType.GLOBAL_SLOW,
            activations,
            (("none", task),),
        )

        self.assertEqual(len(candidates), 1)
        seed = candidates[0]
        self.assertTrue(all(value.rank == 1 for value in seed.factors_by_layer.values()))
        self.assertTrue(
            all(value.rank == 2 for value in seed.rank_reference_by_layer.values())
        )
        # A rank-one approximation captures only half the equal spectrum, so
        # Path A will reject it while Path B still receives a complete seed.
        self.assertTrue(
            all(value > 0.0 for value in seed.functional_error_by_layer.values())
        )

    def test_gate2_cold_start_evidence_survives_empty_replay(self) -> None:
        self.system.config.gates.history_enabled = True
        self.system._successful_slow_commit_count = 1
        candidate = self._rank_candidate(1.0)

        selected, reports = self.system._screen_candidates(
            self.trainer,
            (candidate,),
            calibration=(),
            calibration_before=1.0,
            calibration_fast=0.0,
            history_windows=(),
            history_before=None,
            anchor=(),
            anchor_before=None,
            plasticity_before=None,
        )

        self.assertIsNone(selected)
        self.assertEqual(
            reports[0]["screening_reason"],
            "historical_replay_required_after_slow_commit",
        )
        with self.system._persistent_transaction():
            self.system._successful_slow_commit_count = 7
        self.assertEqual(self.system._successful_slow_commit_count, 1)

    def test_lpr_uses_true_candidate_key_output_with_upstream_gradient(self) -> None:
        obs, _ = _support_segment()
        self.trainer.begin_fd_psc_episode(
            self.system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        records = self.system.external.calibration("ctx")
        factors = {}
        tasks = {}
        ranks = {}
        errors = {}
        for logical_id, adapter in sorted(self.system.injection.adapters.items()):
            b = nn.Parameter(torch.eye(adapter.out_features, 2))
            a = nn.Parameter(torch.eye(2, adapter.in_features))
            factors[logical_id] = LowRankFactors(b, a)
            tasks[logical_id] = LowRankFactors(b.detach().clone(), a.detach().clone())
            ranks[logical_id] = 2
            errors[logical_id] = 0.0
        candidate = _Candidate(
            proposal_type=ProposalType.GLOBAL_SLOW,
            factors_by_layer=factors,
            task_factors_by_layer=tasks,
            functional_error_by_layer=errors,
            selected_rank_by_layer=ranks,
            alpha_shared=1.0,
            alpha_safe=1.0,
            spectral_variant="none",
        )
        predictor_id = next(
            entry.logical_layer_id
            for entry in self.system.target_manifest.entries
            if entry.injected and entry.module_path.endswith("predictor.proj")
        )
        before = self.system._collect_key_layer_outputs(
            self.trainer,
            records,
            logical_ids=(predictor_id,),
            state="before",
        )[predictor_id]
        current = self.system._collect_key_layer_outputs(
            self.trainer,
            records,
            logical_ids=(predictor_id,),
            state="candidate",
            candidate=candidate,
            differentiable_candidate=True,
        )[predictor_id]
        loss = torch.mean((current - before.to(current)) ** 2)
        loss.backward()

        encoder_id = next(
            entry.logical_layer_id
            for entry in self.system.target_manifest.entries
            if entry.injected and entry.module_path.endswith("encoder.projector")
        )
        self.assertIsNotNone(candidate.factors_by_layer[encoder_id].B.grad)
        self.assertGreater(
            float(torch.linalg.vector_norm(candidate.factors_by_layer[encoder_id].B.grad)),
            0.0,
        )
        self.trainer.abort_fd_psc_episode("test_cleanup")

    def test_global_gate6_before_uses_slow_only_not_routed_exception(self) -> None:
        obs, _ = _support_segment()
        self.trainer.begin_fd_psc_episode(
            self.system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        for adapter in self.system.injection.adapters.values():
            adapter.set_active_exception(
                torch.ones(adapter.out_features, 1),
                torch.ones(1, adapter.in_features),
                adapter_id="test-exception",
            )
        self.system._episode_start_adapter_states = (
            self.system._adapter_participant.state_dict()
        )

        global_before = self.system._factor_maps_for_state(
            proposal_type=ProposalType.GLOBAL_SLOW
        )
        exception_before = self.system._factor_maps_for_state(
            proposal_type=ProposalType.REPLACE_EXCEPTION
        )
        self.assertTrue(all(value.rank == 0 for value in global_before.values()))
        self.assertTrue(all(value.rank > 0 for value in exception_before.values()))
        self.trainer.abort_fd_psc_episode("test_cleanup")


if __name__ == "__main__":
    unittest.main()
