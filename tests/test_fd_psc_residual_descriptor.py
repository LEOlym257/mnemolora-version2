from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fd_psc.diagnostics import bitwise_state_equal
from fd_psc.exception_router import ExceptionRouter
from planning.adajepa import AdaJEPATrainer
from tests.test_fd_psc_integration import (
    _ToyWorldModel,
    _config,
    _support_segment,
    _theta0_snapshot,
    _write_fixed_manifest,
)


class ResidualDescriptorTests(unittest.TestCase):
    def test_equal_mse_patterns_produce_distinct_exception_residual_prototypes(self):
        model = _ToyWorldModel()
        trainer = AdaJEPATrainer(
            wm=model,
            lr=0.1,
            steps=1,
            optimizer_name="sgd",
            finetune_encoder=False,
            last_layer_only=False,
            fd_psc={"enabled": False},
        )
        model.eval()

        # Both targets contain one unit residual and therefore have identical
        # scalar MSE.  The unit error occupies a different feature channel,
        # which the residual-pattern descriptor must preserve.
        channel_zero = torch.zeros(1, 2, 3, 2)
        channel_one = torch.zeros_like(channel_zero)
        channel_zero[:, 1, 0, 0] = 1.0
        channel_one[:, 1, 0, 1] = 1.0

        loss_zero = trainer._prediction_loss(channel_zero)
        loss_one = trainer._prediction_loss(channel_one)
        descriptor_zero = trainer._prediction_residual_descriptor(channel_zero)
        descriptor_one = trainer._prediction_residual_descriptor(channel_one)

        self.assertEqual(float(loss_zero), float(loss_one))
        self.assertGreater(descriptor_zero.numel(), 1)
        self.assertEqual(descriptor_zero.shape, descriptor_one.shape)
        self.assertFalse(torch.equal(descriptor_zero, descriptor_one))

        router = ExceptionRouter(
            maximum_adapters=2,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=11,
        )
        first, _ = router.commit_new(
            adapter_state={"name": "channel-zero"},
            context_descriptors=[torch.tensor([1.0, 0.0])],
            residual_descriptors=[descriptor_zero],
        )
        second, _ = router.commit_new(
            adapter_state={"name": "channel-one"},
            context_descriptors=[torch.tensor([1.0, 0.0])],
            residual_descriptors=[descriptor_one],
        )
        first_record = router.get(first)
        second_record = router.get(second)
        self.assertTrue(first_record.residual_available)
        self.assertTrue(second_record.residual_available)
        self.assertFalse(
            torch.equal(first_record.residual_prototype, second_record.residual_prototype)
        )

    def test_replay_descriptor_is_theta0_eval_only_and_runtime_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, split_paths = _write_fixed_manifest(root)
            model = _ToyWorldModel()
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
                fd_psc=_config(manifest, split_paths),
                runtime_output_dir=str(root),
            )
            system = trainer.fd_psc_system
            obs, actions = _support_segment()
            trainer.begin_fd_psc_episode(system.next_episode_id, "ctx", initial_obs=obs)
            system.register_support_segment(obs, actions, iteration=0)

            def set_adapter_value(value: float) -> None:
                with torch.no_grad():
                    for adapter in system.injection.adapters.values():
                        if adapter.pilot_A is not None:
                            adapter.pilot_A.fill_(0.5)
                        if adapter.pilot_B is not None:
                            adapter.pilot_B.fill_(value)

            def build(value: float):
                set_adapter_value(value)
                model.train()
                model.encoder.train()
                model.predictor.train()
                adapter_before = system._adapter_participant.state_dict()
                calls = []
                real_predict = model.predict

                def observed_predict(source):
                    calls.append(
                        (
                            model.training,
                            model.predictor.training,
                            tuple(
                                adapter.adapters_enabled
                                for _, adapter in sorted(system.injection.adapters.items())
                            ),
                        )
                    )
                    return real_predict(source)

                with mock.patch.object(model, "predict", side_effect=observed_predict):
                    windows = system._build_replay_windows(trainer)
                self.assertTrue(calls)
                for world_training, predictor_training, enabled in calls:
                    self.assertFalse(world_training)
                    self.assertFalse(predictor_training)
                    self.assertFalse(any(enabled))
                self.assertTrue(model.training)
                self.assertTrue(model.encoder.training)
                self.assertTrue(model.predictor.training)
                self.assertTrue(
                    bitwise_state_equal(
                        system._adapter_participant.state_dict(),
                        adapter_before,
                    )
                )
                self.assertEqual(len(windows), 1)
                return windows[0]

            first = build(1.0)
            second = build(-3.0)
            self.assertTrue(torch.equal(first.residual, second.residual))
            self.assertEqual(first.difficulty_score, second.difficulty_score)
            self.assertGreater(torch.as_tensor(first.residual).numel(), 1)
            self.assertEqual(
                first.metadata["residual_descriptor_schema"],
                "theta0_jepa_pattern_v1",
            )

            # The test is sensitive to adapter leakage: evaluating the same
            # descriptor with the nonzero episodic branches enabled differs
            # from the stored theta_0 descriptor.
            model.eval()
            prepared_obs, prepared_actions = trainer._prepare_segment(obs, actions)
            with torch.no_grad():
                adapted = trainer._prediction_residual_descriptor(
                    model.encode(prepared_obs, prepared_actions)
                )
            self.assertFalse(torch.equal(first.residual, adapted))

            for tensor, expected in theta0:
                self.assertTrue(torch.equal(tensor.detach(), expected))
            system.finish_episode_without_sleep("residual_descriptor_test_complete")


if __name__ == "__main__":
    unittest.main()
