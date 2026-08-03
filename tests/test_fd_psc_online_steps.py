from __future__ import annotations

import unittest
from typing import Optional

import torch
from torch import nn

from planning.adajepa import AdaJEPATrainer


class _StepPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pilot = nn.Parameter(torch.tensor(0.0))
        self.centered: Optional[nn.Parameter] = None

    def activate_centered(self) -> nn.Parameter:
        if self.centered is not None:
            raise AssertionError("Centered branch activated twice")
        self.centered = nn.Parameter(torch.tensor(0.0))
        return self.centered

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.pilot
        if self.centered is not None:
            scale = scale + self.centered
        return value * scale


class _StepWorldModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = _StepPredictor()
        self.encoder = nn.Identity()
        self.num_hist = 1
        self.concat_dim = 0
        self.action_dim = 1
        self.stop_grad = True

    def encode(self, obs, actions):
        del actions
        return obs["z"]

    def predict(self, value):
        return self.predictor(value)


class _StepLifecycleSystem:
    def __init__(self, model: _StepWorldModel) -> None:
        self.model = model
        self.optimizer_step_calls = 0
        self.conflict_evaluation_calls = 0
        self.after_event_calls = 0
        self.after_event_was_per_step = False
        self.finished = False
        self.centered_parameter: Optional[nn.Parameter] = None

    def require_active_episode(self) -> None:
        return None

    def prepare_online_mode(self, **kwargs) -> None:
        del kwargs

    def online_parameter_groups(self, *, include_encoder: bool):
        self.assert_no_encoder(include_encoder)
        parameters = [self.model.predictor.pilot]
        if self.model.predictor.centered is not None:
            parameters.append(self.model.predictor.centered)
        return parameters, []

    @staticmethod
    def assert_no_encoder(include_encoder: bool) -> None:
        if include_encoder:
            raise AssertionError("fixture does not adapt the encoder")

    @staticmethod
    def capture_update_rng():
        return None

    @staticmethod
    def backward_with_sdc(loss, optimizer, **kwargs) -> None:
        del optimizer, kwargs
        loss.backward()

    def note_optimizer_step(self, step: int, loss: float) -> None:
        del step, loss
        self.optimizer_step_calls += 1

    def after_optimizer_step(self, trainer, segments, step_losses) -> bool:
        del trainer, segments, step_losses
        self.conflict_evaluation_calls += 1
        if self.conflict_evaluation_calls == 1:
            self.centered_parameter = self.model.predictor.activate_centered()
            return True
        return False

    def after_finetune_event(
        self,
        trainer,
        segments,
        step_losses,
        *,
        conflict_evaluated_per_step: bool = False,
    ) -> None:
        del trainer, segments, step_losses
        self.after_event_calls += 1
        self.after_event_was_per_step = bool(conflict_evaluated_per_step)

    def finish_online_mode(self) -> None:
        self.finished = True


def _trainer(*, steps: int):
    model = _StepWorldModel()
    trainer = AdaJEPATrainer(
        wm=model,
        lr=0.1,
        steps=steps,
        optimizer_name="sgd",
        finetune_encoder=False,
        last_layer_only=False,
        fd_psc={"enabled": False},
    )
    lifecycle = _StepLifecycleSystem(model)
    trainer.fd_psc_system = lifecycle
    constructed = []
    original_make_optimizer = trainer._make_optimizer

    def tracked_make_optimizer():
        optimizer = original_make_optimizer()
        constructed.append(optimizer)
        return optimizer

    trainer._make_optimizer = tracked_make_optimizer
    return model, trainer, lifecycle, constructed


def _segment():
    # Two patches are required because AdaJEPA excludes the final action token.
    z = torch.tensor([[[[1.0], [0.0]], [[2.0], [0.0]]]])
    obs = {"z": z}
    actions = torch.zeros(1, 1, 1)
    return obs, actions


class OnlineStepLifecycleTests(unittest.TestCase):
    def test_first_step_slice_rebuilds_optimizer_and_updates_centered_on_second(self):
        model, trainer, lifecycle, optimizers = _trainer(steps=2)
        obs, actions = _segment()
        losses = trainer._finetune_fd_psc([obs], [actions])

        self.assertEqual(len(losses), 2)
        self.assertEqual(lifecycle.optimizer_step_calls, 2)
        self.assertEqual(lifecycle.conflict_evaluation_calls, 2)
        self.assertEqual(lifecycle.after_event_calls, 1)
        self.assertTrue(lifecycle.after_event_was_per_step)
        self.assertTrue(lifecycle.finished)
        self.assertEqual(len(optimizers), 2)
        centered = lifecycle.centered_parameter
        self.assertIsNotNone(centered)
        self.assertIn(
            id(centered),
            {
                id(parameter)
                for group in optimizers[1].param_groups
                for parameter in group["params"]
            },
        )
        self.assertNotEqual(float(centered.detach()), 0.0)
        self.assertIs(model.predictor.centered, centered)

    def test_single_step_evaluates_conflict_once_without_pointless_rebuild(self):
        _, trainer, lifecycle, optimizers = _trainer(steps=1)
        obs, actions = _segment()
        losses = trainer._finetune_fd_psc([obs], [actions])

        self.assertEqual(len(losses), 1)
        self.assertEqual(lifecycle.conflict_evaluation_calls, 1)
        self.assertEqual(lifecycle.after_event_calls, 1)
        self.assertTrue(lifecycle.after_event_was_per_step)
        self.assertEqual(len(optimizers), 1)


if __name__ == "__main__":
    unittest.main()
