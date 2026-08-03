from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from fd_psc.diagnostics import bitwise_state_equal
from fd_psc.transaction import RNGSnapshot
from fd_psc.trainer import FDPSCIntegrationError
from planning.adajepa import AdaJEPATrainer
from tests.test_fd_psc_integration import (
    _ToyWorldModel,
    _config,
    _support_segment,
    _write_fixed_manifest,
)


class SDCAnchorRegressionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.split_paths = _write_fixed_manifest(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _trainer(self) -> AdaJEPATrainer:
        return AdaJEPATrainer(
            wm=_ToyWorldModel(),
            lr=0.1,
            steps=1,
            optimizer_name="sgd",
            finetune_encoder=True,
            last_layer_only=False,
            encoder_lr=0.1,
            encoder_last_layer_only=False,
            fd_psc=_config(
                self.manifest,
                self.split_paths,
                gradient_geometry={
                    "enabled": True,
                    "current_batches": 1,
                    "history_batches": 1,
                    "anchor_batches": 1,
                    "windows_per_batch": 1,
                },
                sdc={
                    "enabled": True,
                    "event_triggered": True,
                    "check_every_replans": 1,
                    "drift_threshold": 0.25,
                    "anchor_regression_trigger": 0.0,
                },
                anchor_data={
                    "windows": 1,
                    "manifest_path": str(self.manifest),
                    "data_path": str(self.split_paths["anchor"]),
                },
            ),
            runtime_output_dir=str(self.root),
        )

    def test_positive_cosine_anchor_loss_regression_triggers_without_state_leak(self):
        trainer = self._trainer()
        system = trainer.fd_psc_system
        obs, _ = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )

        entries = system.injection.manifest.by_logical_id()
        predictor_id = next(
            logical_id
            for logical_id, entry in sorted(entries.items())
            if entry.module_group == "predictor"
        )
        predictor = system.injection.adapters[predictor_id]
        with torch.no_grad():
            # P_before predicts zero in this fixture.  Make P_fast map the
            # anchor source [1, 0] to [10, 0], increasing the unchanged JEPA
            # anchor loss while rho_anchor remains explicitly positive.
            predictor.pilot_A.copy_(
                torch.tensor(
                    [[1.0, 0.0]],
                    device=predictor.pilot_A.device,
                    dtype=predictor.pilot_A.dtype,
                )
            )
            predictor.pilot_B.copy_(
                torch.tensor(
                    [[10.0], [0.0]],
                    device=predictor.pilot_B.device,
                    dtype=predictor.pilot_B.dtype,
                )
            )
        system._latest_gradient_cosines = {
            logical_id: {"history": None, "anchor": 0.75}
            for logical_id in system.injection.adapters
        }

        adapter_before = system._adapter_participant.state_dict()
        adapter_parameter_slots = {
            (logical_id, name): getattr(adapter, name)
            for logical_id, adapter in sorted(system.injection.adapters.items())
            for name in ("pilot_A", "pilot_B", "center_A", "center_B")
            if isinstance(getattr(adapter, name), nn.Parameter)
        }
        requires_grad_before = {
            id(parameter): bool(parameter.requires_grad)
            for parameter in trainer.wm.parameters()
        }
        buffers_before = {
            name: value.detach().clone()
            for name, value in trainer.wm.named_buffers()
        }
        modes_before = {
            name: module.training
            for name, module in trainer.wm.named_modules()
        }
        router_before = system.exception_router.state_dict()
        rng_before = RNGSnapshot.capture()
        original_evaluate = trainer.evaluate_external_records

        def stochastic_anchor_evaluate(records):
            # Make RNG restoration observable rather than relying on the toy
            # model's otherwise deterministic frozen-eval forward.
            random.random()
            np.random.rand()
            torch.rand(3)
            return original_evaluate(records)

        with mock.patch(
            "fd_psc.trainer.spectral_drift",
            return_value=SimpleNamespace(value=1.0, available=True),
        ), mock.patch.object(
            trainer,
            "evaluate_external_records",
            side_effect=stochastic_anchor_evaluate,
        ), mock.patch.object(
            system,
            "_evaluate_state",
            wraps=system._evaluate_state,
        ) as evaluate:
            system.after_finetune_event(
                trainer,
                (),
                (),
                conflict_evaluated_per_step=True,
            )
            system.after_finetune_event(
                trainer,
                (),
                (),
                conflict_evaluated_per_step=True,
            )

        self.assertGreater(system._sdc_anchor_regression_value, 0.0)
        self.assertTrue(all(system._sdc_active.values()))
        self.assertTrue(
            all(
                values["anchor"] > 0.0
                for values in system._latest_gradient_cosines.values()
            )
        )
        # P_before is cached once; each scheduled check evaluates only P_fast.
        self.assertEqual(
            [call.kwargs["state"] for call in evaluate.call_args_list],
            ["before", "fast", "fast"],
        )
        self.assertEqual(system._replan_index, 2)

        self.assertTrue(
            bitwise_state_equal(adapter_before, system._adapter_participant.state_dict())
        )
        for (logical_id, name), parameter in adapter_parameter_slots.items():
            self.assertIs(getattr(system.injection.adapters[logical_id], name), parameter)
        self.assertEqual(
            requires_grad_before,
            {
                id(parameter): bool(parameter.requires_grad)
                for parameter in trainer.wm.parameters()
            },
        )
        buffers_after = dict(trainer.wm.named_buffers())
        self.assertEqual(set(buffers_before), set(buffers_after))
        for name, expected in buffers_before.items():
            self.assertTrue(torch.equal(buffers_after[name], expected), msg=name)
        self.assertEqual(
            modes_before,
            {
                name: module.training
                for name, module in trainer.wm.named_modules()
            },
        )
        self.assertTrue(
            bitwise_state_equal(router_before, system.exception_router.state_dict())
        )
        self.assertTrue(bitwise_state_equal(rng_before, RNGSnapshot.capture()))
        system.assert_base_frozen()
        system.abort_episode("sdc-anchor-regression-test-complete")
        self.assertEqual(system._sdc_anchor_records, ())
        self.assertIsNone(system._sdc_anchor_before_loss)
        self.assertIsNone(system._sdc_anchor_current_loss)
        self.assertIsNone(system._sdc_anchor_regression_value)

    def test_missing_runtime_anchor_fails_closed_before_tracker_activation(self):
        trainer = self._trainer()
        system = trainer.fd_psc_system
        obs, _ = _support_segment()
        trainer.begin_fd_psc_episode(
            system.next_episode_id,
            "ctx",
            initial_obs=obs,
        )
        external = system.external
        system.external = None
        try:
            with self.assertRaisesRegex(
                FDPSCIntegrationError,
                "fixed anchor registry is unavailable",
            ):
                system.after_finetune_event(
                    trainer,
                    (),
                    (),
                    conflict_evaluated_per_step=True,
                )
            self.assertTrue(all(not active for active in system._sdc_active.values()))
            self.assertIsNone(system._sdc_anchor_before_loss)
            self.assertEqual(system._replan_index, 0)
        finally:
            system.external = external
            system.abort_episode("sdc-missing-anchor-test-complete")


if __name__ == "__main__":
    unittest.main()
