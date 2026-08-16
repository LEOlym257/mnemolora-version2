import copy
import math
import unittest

from fd_psc.config import AdaptiveBudgetConfig, RTRCConfig
from fd_psc.v2.budget_controller import AdaptiveBudgetController


def _controller():
    return AdaptiveBudgetController(
        AdaptiveBudgetConfig(
            controller_learning_rate=0.5,
            plasticity_loss_target=0.1,
            history_regression_target=0.01,
            history_weight=2.0,
            error_clip=1.0,
            minimum_wake_gain=1.0e-8,
        ),
        RTRCConfig(
            budget_fraction_initial=0.2,
            budget_fraction_minimum=0.02,
            budget_fraction_maximum=1.0,
        ),
    )


class AdaptiveBudgetControllerTests(unittest.TestCase):
    def test_high_plasticity_cost_and_low_history_regression_increases_beta(self):
        controller = _controller()
        before = controller.beta
        update = controller.update(
            current_before=1.0,
            current_fast=0.0,
            current_rtrc=0.8,
            history_before=1.0,
            history_rtrc=1.0,
            epsilon=1.0e-8,
        )
        self.assertGreater(update.beta_after, before)
        self.assertGreater(update.operational_plasticity_loss_ratio, 0.1)

    def test_high_history_regression_and_low_plasticity_cost_decreases_beta(self):
        controller = _controller()
        before = controller.beta
        update = controller.update(
            current_before=1.0,
            current_fast=0.0,
            current_rtrc=0.0,
            history_before=1.0,
            history_rtrc=2.0,
            epsilon=1.0e-8,
        )
        self.assertLess(update.beta_after, before)
        self.assertGreater(update.historical_replay_regression, 0.01)

    def test_near_zero_wake_gain_ignores_plasticity_without_nan(self):
        controller = _controller()
        update = controller.update(
            current_before=1.0,
            current_fast=1.0,
            current_rtrc=100.0,
            history_before=None,
            history_rtrc=None,
            epsilon=1.0e-8,
        )
        self.assertFalse(update.plasticity_signal_available)
        self.assertFalse(update.update_applied)
        self.assertEqual(update.operational_plasticity_loss_ratio, 0.0)
        self.assertTrue(math.isfinite(update.beta_after))

    def test_empty_replay_marks_history_unavailable_but_controller_is_valid(self):
        controller = _controller()
        u_before = controller.u
        update = controller.update(
            current_before=1.0,
            current_fast=0.5,
            current_rtrc=0.75,
            history_before=None,
            history_rtrc=None,
            epsilon=1.0e-8,
        )
        self.assertFalse(update.history_signal_available)
        self.assertTrue(update.update_applied)
        expected_plasticity_error = 0.5 - 0.1
        self.assertAlmostEqual(
            update.controller_error,
            expected_plasticity_error,
            places=7,
        )
        self.assertAlmostEqual(
            controller.u - u_before,
            0.5 * expected_plasticity_error,
            places=7,
        )
        self.assertTrue(math.isfinite(controller.beta))

    def test_checkpoint_roundtrip_restores_u_beta_and_update_count(self):
        controller = _controller()
        controller.update(
            current_before=1.0,
            current_fast=0.5,
            current_rtrc=0.9,
            history_before=1.0,
            history_rtrc=1.1,
            epsilon=1.0e-8,
        )
        state = copy.deepcopy(controller.state_dict())
        restored = _controller()
        restored.load_state_dict(state)
        self.assertEqual(restored.u, controller.u)
        self.assertEqual(restored.beta, controller.beta)
        self.assertEqual(restored.update_count, controller.update_count)


if __name__ == "__main__":
    unittest.main()
