import unittest

import torch

from fd_psc.low_rank_merge import LowRankFactors
from fd_psc.v2.deep_sleep import (
    ModelTrace,
    ResidualTarget,
    _residual_losses,
    compare_functional_outputs,
    refit_parameter_residual,
)


class ResidualDistillationMathTests(unittest.TestCase):
    def test_output_and_relative_hidden_residual_losses_match_definition(self):
        target = ResidualTarget(
            source="raw_replay",
            payload=None,
            core_output=torch.tensor([1.0, -1.0]),
            teacher_output_residual=torch.tensor([2.0, 4.0]),
            core_hidden={"critical": torch.tensor([10.0, 20.0])},
            teacher_hidden_residual={"critical": torch.tensor([2.0, 4.0])},
        )
        exact = ModelTrace(
            output=torch.tensor([3.0, 3.0]),
            hidden={"critical": torch.tensor([12.0, 24.0])},
        )
        output, hidden = _residual_losses((target,), (exact,), epsilon=1.0e-8)
        self.assertEqual(float(output), 0.0)
        self.assertEqual(float(hidden), 0.0)

        shifted = ModelTrace(
            output=torch.tensor([4.0, 2.0]),
            hidden={"critical": torch.tensor([13.0, 23.0])},
        )
        output, hidden = _residual_losses((target,), (shifted,), epsilon=1.0e-8)
        expected_output = torch.tensor([1.0, -1.0]).square().mean()
        hidden_difference = torch.tensor([1.0, -1.0])
        hidden_target = torch.tensor([2.0, 4.0])
        expected_hidden = hidden_difference.square().mean() / (
            hidden_target.square().mean() + 1.0e-8
        )
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(hidden, expected_hidden)

    def test_parameter_residual_refit_uses_core_old_plus_uncompressed_slow(self):
        core_old = {"layer": torch.tensor([[0.2, 0.0], [0.0, -0.1]])}
        slow_dense = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        slow = {
            "layer": LowRankFactors(
                torch.eye(2),
                slow_dense,
            )
        }
        core_new = {"layer": torch.tensor([[1.2, 0.0], [0.0, -0.1]])}
        result = refit_parameter_residual(
            core_old=core_old,
            slow_uncompressed=slow,
            core_new=core_new,
            residual_rank=1,
            epsilon=1.0e-8,
        )
        fitted = result.factors["layer"]
        expected_residual = core_old["layer"] + slow_dense - core_new["layer"]
        self.assertEqual(fitted.rank, 1)
        self.assertEqual(result.numerical_rank_before["layer"], 2)
        # The exact rank-1 tail error is the omitted second singular value.
        error = torch.linalg.vector_norm(expected_residual - fitted.b @ fitted.a)
        self.assertAlmostEqual(float(error), 1.0, places=6)

    def test_functional_error_is_relative_squared_output_error(self):
        comparison = compare_functional_outputs(
            (torch.tensor([3.0, 4.0]),),
            (torch.tensor([0.0, 4.0]),),
            epsilon=1.0e-12,
        )
        self.assertAlmostEqual(comparison.absolute_error, 3.0)
        self.assertAlmostEqual(comparison.reference_norm, 5.0)
        self.assertAlmostEqual(comparison.relative_error, 9.0 / 25.0)
        self.assertTrue(comparison.passes(0.4, 0.0))
        self.assertFalse(comparison.passes(0.3, 0.0))


if __name__ == "__main__":
    unittest.main()
