import math
import unittest

import torch

from fd_psc.activation_subspace import (
    ActivationSubspace,
    conv2d_group_activation_matrices,
)
from fd_psc.gradient_geometry import (
    ConflictEMA,
    c_pcgrad,
    dual_constraint_projection,
    gradient_cosine,
)
from fd_psc.gradient_hooks import EffectiveWeightGradientCollector
from fd_psc.low_rank_merge import (
    LowRankFactors,
    clipped_rank_candidates,
    concatenate_factors,
    factor_svd,
    factors_from_svd,
    functional_error,
    select_rank,
)
from fd_psc.slice_initializer import (
    initialize_slice,
    match_first_step_magnitude,
    simulate_factor_first_step,
)
from fd_psc.spectral_control import (
    SDCEventTracker,
    compute_base_spectrum,
    effective_gradient_proxy,
    project_box_weighted_l2,
    sdc_correct_gradient,
    spectral_drift,
    spectral_surgery,
)


class DeterministicTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(7)


class EffectiveWeightHookTests(DeterministicTestCase):
    def test_linear_hook_matches_weight_grad_with_reentry(self):
        module = torch.nn.Linear(4, 3, bias=True)
        x1 = torch.randn(2, 5, 4, requires_grad=True)
        x2 = torch.randn(2, 5, 4, requires_grad=True)
        coefficient1 = torch.randn(2, 5, 3)
        coefficient2 = torch.randn(2, 5, 3)
        with EffectiveWeightGradientCollector(module, "predictor.linear") as collector:
            loss = (module(x1) * coefficient1).sum() + (module(x2) * coefficient2).sum()
            loss.backward()
            captured = collector.matrix_gradients["predictor.linear"]
            self.assertTrue(torch.allclose(captured, module.weight.grad, atol=1e-6, rtol=1e-6))
            self.assertEqual(collector.statistics.forward_invocations, 2)
            self.assertEqual(collector.statistics.backward_invocations, 2)
            self.assertEqual(collector.statistics.pending_invocations, 0)

    def test_grouped_conv_hook_matches_weight_grad_with_padding_mode(self):
        module = torch.nn.Conv2d(
            4,
            6,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=2,
            padding_mode="reflect",
            bias=True,
        )
        inputs = torch.randn(2, 4, 9, 9, requires_grad=True)
        with EffectiveWeightGradientCollector(module, "encoder.projector.conv") as collector:
            output = module(inputs)
            (output.square().mean() + output.abs().mean()).backward()
            self.assertTrue(
                torch.allclose(collector.weight_gradient, module.weight.grad, atol=2e-6, rtol=2e-6)
            )
            matrices = collector.matrix_gradients
            self.assertEqual(
                set(matrices),
                {
                    "encoder.projector.conv::group=0",
                    "encoder.projector.conv::group=1",
                },
            )
            expected = module.weight.grad.reshape(2, 3, -1)
            self.assertTrue(torch.allclose(matrices["encoder.projector.conv::group=0"], expected[0]))
            self.assertTrue(torch.allclose(matrices["encoder.projector.conv::group=1"], expected[1]))

    def test_same_padding_conv_hook_matches_weight_grad(self):
        module = torch.nn.Conv2d(2, 3, kernel_size=4, padding="same", dilation=1, bias=False)
        inputs = torch.randn(1, 2, 8, 7, requires_grad=True)
        with EffectiveWeightGradientCollector(module, "same") as collector:
            module(inputs).sum().backward()
            self.assertTrue(
                torch.allclose(collector.weight_gradient, module.weight.grad, atol=2e-6, rtol=2e-6)
            )


class GradientGeometryTests(DeterministicTestCase):
    def test_zero_reference_cosine_is_unavailable(self):
        result = gradient_cosine(torch.ones(2, 2), torch.zeros(2, 2))
        self.assertFalse(result.available)
        self.assertIsNone(result.value)

    def test_c_pcgrad_changes_only_negative_conflict(self):
        current = torch.tensor([1.0, -1.0])
        aligned = torch.tensor([2.0, -2.0])
        self.assertIs(c_pcgrad(current, aligned), current)
        conflicting = torch.tensor([-1.0, 0.0])
        projected = c_pcgrad(current, conflicting)
        self.assertGreaterEqual(float(torch.dot(projected, conflicting)), -1e-6)
        self.assertEqual(float(projected[1]), -1.0)

    def test_conflict_ema_first_value_and_consecutive_trigger(self):
        tracker = ConflictEMA(beta=0.5, threshold=-0.1, consecutive_required=2)
        first = tracker.update(-0.2)
        self.assertAlmostEqual(first.ema, -0.2)
        self.assertFalse(first.triggered)
        second = tracker.update(-0.4)
        self.assertTrue(second.triggered)
        unavailable = tracker.update(None)
        self.assertFalse(unavailable.available)
        self.assertEqual(unavailable.consecutive_conflicts, 0)

    def test_dual_constraint_active_set_satisfies_both_constraints(self):
        current = torch.tensor([[-1.0, -2.0]])
        history = torch.tensor([[1.0, 0.0]])
        anchor = torch.tensor([[0.0, 1.0]])
        result = dual_constraint_projection(current, history, anchor)
        self.assertTrue(result.feasible)
        self.assertEqual(set(result.active_constraints), {"history", "anchor"})
        self.assertGreaterEqual(float((result.gradient * history).sum()), -1e-7)
        self.assertGreaterEqual(float((result.gradient * anchor).sum()), -1e-7)

    def test_dual_constraint_no_conflict_and_collinear_reference(self):
        current = torch.tensor([1.0, 2.0])
        history = torch.tensor([1.0, 0.0])
        anchor = 2.0 * history
        unchanged = dual_constraint_projection(current, history, anchor)
        self.assertTrue(torch.equal(unchanged.gradient, current))
        conflict = dual_constraint_projection(-current, history, anchor)
        self.assertTrue(conflict.feasible)
        self.assertGreaterEqual(float(torch.dot(conflict.gradient, history)), -1e-7)


class SliceTests(DeterministicTestCase):
    def test_slice_exact_and_dimension_fallback(self):
        gradient = torch.randn(8, 8)
        generator = torch.Generator().manual_seed(11)
        exact = initialize_slice(
            gradient,
            requested_rank=2,
            mode="slice_exact",
            generator=generator,
            magnitude_mode="none",
        )
        self.assertTrue(exact.success)
        self.assertEqual(exact.mode, "slice_exact")
        self.assertEqual(exact.b0.shape, (8, 2))
        self.assertEqual(exact.a0.shape, (2, 8))

        fallback = initialize_slice(
            torch.randn(3, 4),
            requested_rank=2,
            mode="slice_exact",
            generator=torch.Generator().manual_seed(3),
            magnitude_mode="none",
        )
        self.assertTrue(fallback.success)
        self.assertEqual(fallback.mode, "slice_symmetric")
        self.assertIn("dimension_fallback", fallback.reason)

    def test_slice_symmetric_reconstructs_returned_singular_values(self):
        result = initialize_slice(
            torch.randn(7, 5),
            requested_rank=3,
            mode="slice_symmetric",
            generator=torch.Generator().manual_seed(5),
            magnitude_mode="none",
        )
        reconstructed_singular = torch.linalg.svdvals(result.b0 @ result.a0)
        self.assertTrue(
            torch.allclose(
                reconstructed_singular[: result.actual_rank],
                result.singular_values[: result.actual_rank],
                atol=2e-4,
                rtol=2e-4,
            )
        )

    def test_first_step_matching_uses_real_optimizer_and_descent_direction(self):
        gradient = torch.randn(5, 4)
        seed = initialize_slice(
            gradient,
            requested_rank=2,
            mode="slice_symmetric",
            generator=torch.Generator().manual_seed(9),
            magnitude_mode="none",
            alpha=2.0,
        )
        known_beta = 0.7
        known = simulate_factor_first_step(
            gradient,
            math.sqrt(known_beta) * seed.b0,
            math.sqrt(known_beta) * seed.a0,
            scaling=1.0,
            optimizer_name="sgd",
            learning_rate=1e-2,
            weight_decay=0.0,
        )
        match = match_first_step_magnitude(
            gradient,
            seed.b0,
            seed.a0,
            known.norm,
            scaling=1.0,
            maximum_beta=2.0,
            optimizer_name="sgd",
            learning_rate=1e-2,
            weight_decay=0.0,
        )
        self.assertTrue(match.matched, match)
        self.assertLessEqual(match.relative_error, 0.05)
        self.assertGreater(match.descent_cosine, 0.0)

    def test_zero_gradient_keeps_pilot(self):
        result = initialize_slice(torch.zeros(4, 4), requested_rank=2, magnitude_mode="none")
        self.assertFalse(result.success)
        self.assertTrue(result.fallback_to_pilot)


class SpectralControlTests(DeterministicTestCase):
    def test_drift_and_sdc_gamma(self):
        base = torch.diag(torch.tensor([5.0, 1.0]))
        spectrum = compute_base_spectrum(base, energy_threshold=0.9)
        principal = LowRankFactors(torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]]))
        orthogonal = LowRankFactors(torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0, 1.0]]))
        self.assertGreater(spectral_drift(principal, spectrum).value, 0.99)
        self.assertLess(spectral_drift(orthogonal, spectrum).value, 1e-6)
        zero_drift = spectral_drift(LowRankFactors.zeros(2, 2), spectrum)
        self.assertTrue(zero_drift.available)
        self.assertEqual(zero_drift.value, 0.0)
        mixed = LowRankFactors(
            torch.eye(2),
            torch.diag(torch.tensor([1.0, math.sqrt(3.0)])),
        )
        # V2 §10.2 is a ratio of squared Frobenius energies: 1/4.
        self.assertAlmostEqual(spectral_drift(mixed, spectrum).value, 0.25, places=5)
        correction = sdc_correct_gradient(
            torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
            spectrum,
            minimum_gamma=0.1,
            active=True,
        )
        self.assertTrue(correction.applied)
        self.assertAlmostEqual(correction.gamma, 0.1, places=6)
        self.assertAlmostEqual(float(correction.gradient[0, 0]), 0.2, places=6)

    def test_sdc_proxy_produces_target_factor_gradient(self):
        gradient_difference = torch.randn(4, 3)
        b = torch.randn(4, 2, requires_grad=True)
        a = torch.randn(2, 3, requires_grad=True)
        loss = effective_gradient_proxy(gradient_difference, b, a, scaling=0.5)
        loss.backward()
        self.assertTrue(
            torch.allclose(b.grad, 0.5 * gradient_difference @ a.detach().T, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(a.grad, 0.5 * b.detach().T @ gradient_difference, atol=1e-6)
        )

    def test_sdc_event_requires_drift_and_anchor_signal(self):
        tracker = SDCEventTracker(
            check_every_replans=1,
            drift_threshold=0.25,
            drift_consecutive_checks=2,
        )
        no_anchor = tracker.update(0, 0.4, anchor_regression=0.0, anchor_cosine=0.2)
        self.assertFalse(no_anchor.active)
        triggered = tracker.update(1, 0.5, anchor_regression=0.1)
        self.assertTrue(triggered.active)
        disabled = tracker.update(2, 0.1, anchor_regression=0.1)
        self.assertFalse(disabled.active)

    def test_joint_box_l2_projection_and_spectral_surgery(self):
        singular = torch.tensor([3.0, 2.0, 1.0])
        projected = project_box_weighted_l2(
            torch.tensor([0.1, 2.0, 1.0]), singular, 0.75, 1.25
        )
        self.assertTrue(projected.feasible, projected.reason)
        self.assertTrue(torch.all(projected.scales >= 0.75))
        self.assertTrue(torch.all(projected.scales <= 1.25))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(projected.scales * singular)),
            float(torch.linalg.vector_norm(singular)),
            places=5,
        )

        factors = LowRankFactors(torch.randn(5, 3), torch.randn(3, 4))
        result = spectral_surgery(
            factors,
            torch.randn(5, 4),
            steps=2,
            learning_rate=0.01,
            preserve_spectral_l2_norm=True,
        )
        self.assertTrue(torch.all(result.scales >= 0.75))
        self.assertTrue(torch.all(result.scales <= 1.25))
        before = factor_svd(factors).singular_values
        after = factor_svd(result.factors).singular_values
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(before)),
            float(torch.linalg.vector_norm(after)),
            places=4,
        )


class ActivationSubspaceTests(DeterministicTestCase):
    def test_grouped_conv_unfold_keeps_groups_separate_and_matches_conv_geometry(self):
        conv = torch.nn.Conv2d(
            4,
            6,
            kernel_size=2,
            stride=2,
            padding=1,
            dilation=2,
            groups=2,
            padding_mode="reflect",
            bias=False,
        )
        inputs = torch.arange(2 * 4 * 7 * 8, dtype=torch.float32).reshape(2, 4, 7, 8)
        groups = conv2d_group_activation_matrices(conv, inputs)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].shape[1], 2 * 2 * 2)
        padded = torch.nn.functional.pad(
            inputs, conv._reversed_padding_repeated_twice, mode="reflect"
        )
        explicit = torch.nn.functional.unfold(
            padded,
            kernel_size=conv.kernel_size,
            dilation=conv.dilation,
            padding=0,
            stride=conv.stride,
        )
        locations = explicit.shape[-1]
        expected = explicit.reshape(2, 2, 8, locations)
        for group_index in range(2):
            expected_group = (
                expected[:, group_index]
                .permute(0, 2, 1)
                .reshape(2 * locations, 8)
            )
            self.assertTrue(torch.equal(groups[group_index], expected_group))

    def test_q_uses_input_right_singular_vectors_and_soft_weights(self):
        activations = torch.diag(torch.tensor([5.0, 2.0, 0.2])).repeat(3, 1)
        state = ActivationSubspace.from_activations(
            activations,
            maximum_rank=3,
            spectral_energy_threshold=0.99,
        )
        self.assertEqual(state.q.shape[0], activations.shape[1])
        self.assertEqual(state.q.shape[1], state.energies.shape[0])
        first_projector = state.q[:, :1] @ state.q[:, :1].T
        self.assertTrue(torch.allclose(first_projector, torch.diag(torch.tensor([1.0, 0.0, 0.0])), atol=1e-5))
        weights = state.soft_ness_weights()
        self.assertTrue(torch.all(weights.weights >= 0.0))
        self.assertTrue(torch.all(weights.weights <= 1.0))
        self.assertGreaterEqual(float(weights.weights[0]), float(weights.weights[-1]))

    def test_minimum_energy_uses_covariance_units_independent_of_sample_count(self):
        minimum_energy = 1.0e-4
        covariance_energies = torch.tensor([2.0e-4, 5.0e-5])
        base = torch.diag(torch.sqrt(2.0 * covariance_energies))
        repeated = base.repeat(100, 1)

        small = ActivationSubspace.from_activations(
            base,
            maximum_rank=2,
            spectral_energy_threshold=1.0,
            minimum_energy=minimum_energy,
        )
        large = ActivationSubspace.from_activations(
            repeated,
            maximum_rank=2,
            spectral_energy_threshold=1.0,
            minimum_energy=minimum_energy,
        )

        self.assertEqual(small.rank, 1)
        self.assertEqual(large.rank, 1)
        torch.testing.assert_close(small.energies, covariance_energies[:1])
        torch.testing.assert_close(large.energies, covariance_energies[:1])

    def test_factor_only_soft_ness_matches_explicit_small_projector(self):
        h = torch.randn(30, 4)
        state = ActivationSubspace.from_activations(h, maximum_rank=3)
        factors = LowRankFactors(torch.randn(5, 2), torch.randn(2, 4))
        weights = state.soft_ness_weights().weights
        transformed = state.transform_right_factors(factors, 0.25, 0.75, weights)
        projector = state.q @ torch.diag(weights) @ state.q.T
        explicit_a = factors.a @ (0.75 * torch.eye(4) + (0.25 - 0.75) * projector)
        self.assertTrue(torch.allclose(transformed.a, explicit_a, atol=1e-6, rtol=1e-6))

        empty = ActivationSubspace.empty(4)
        safe = empty.transform_right_factors(factors, 0.0, 0.75)
        self.assertTrue(torch.allclose(safe.a, 0.75 * factors.a))

    def test_incremental_update_tracks_new_input_direction(self):
        state = ActivationSubspace.empty(3)
        new_h = torch.tensor([[0.0, 4.0, 0.0], [0.0, 3.0, 0.0]])
        updated = state.update(
            new_h,
            forgetting_factor=0.5,
            maximum_rank=2,
            spectral_energy_threshold=0.99,
        )
        projector = updated.q[:, :1] @ updated.q[:, :1].T
        self.assertTrue(torch.allclose(projector, torch.diag(torch.tensor([0.0, 1.0, 0.0])), atol=1e-6))


class LowRankMergeTests(DeterministicTestCase):
    def test_qr_small_svd_and_factor_concatenation_match_dense(self):
        first = LowRankFactors(torch.randn(7, 3), torch.randn(3, 6))
        second = LowRankFactors(torch.randn(7, 2), torch.randn(2, 6))
        merged = concatenate_factors((first, second))
        decomposition = factor_svd(merged)
        reconstructed = factors_from_svd(decomposition)
        dense = first.b @ first.a + second.b @ second.a
        self.assertTrue(
            torch.allclose(reconstructed.b @ reconstructed.a, dense, atol=2e-5, rtol=2e-5)
        )

    def test_functional_error_and_minimum_feasible_rank(self):
        singular = torch.tensor([3.0, 1.0, 0.1])
        roots = torch.sqrt(singular)
        candidate = LowRankFactors(torch.diag(roots), torch.diag(roots))
        activations = torch.eye(3)
        strict_selection = select_rank(
            candidate,
            activations,
            allowed_ranks=[1, 2, 3],
            maximum_rank=3,
            spectral_energy_threshold=0.99,
            functional_error_threshold=0.0005,
        )
        self.assertTrue(strict_selection.feasible, strict_selection.reason)
        self.assertEqual(strict_selection.rank, 3)
        selection = select_rank(
            candidate,
            activations,
            allowed_ranks=[1, 2, 3],
            maximum_rank=3,
            spectral_energy_threshold=0.99,
            functional_error_threshold=0.002,
        )
        self.assertTrue(selection.feasible, selection.reason)
        self.assertEqual(selection.rank, 2)
        error = functional_error(candidate, selection.factors, activations)
        self.assertAlmostEqual(error.relative_error, 0.01 / 10.01, places=6)
        self.assertLessEqual(error.relative_error, 0.002)

    def test_rank_cap_failure_zero_rank_and_small_layer_clipping(self):
        self.assertEqual(clipped_rank_candidates([8, 16, 32], 32, 3, 5), [3])
        self.assertEqual(clipped_rank_candidates([32], 16, 64, 64), [16])
        zero = LowRankFactors.zeros(3, 5)
        zero_selection = select_rank(zero, torch.randn(4, 5), [8, 16], 16)
        self.assertTrue(zero_selection.feasible)
        self.assertEqual(zero_selection.rank, 0)

        candidate = LowRankFactors(torch.eye(3), torch.eye(3))
        failed = select_rank(
            candidate,
            torch.eye(3),
            allowed_ranks=[1],
            maximum_rank=1,
            spectral_energy_threshold=0.99,
            functional_error_threshold=0.0,
        )
        self.assertFalse(failed.feasible)
        self.assertIsNone(failed.factors)

    def test_tiny_nonzero_spectrum_is_not_canonicalized_to_rank_zero(self):
        singular_value = 5.0e-5
        root = singular_value ** 0.5
        candidate = LowRankFactors(
            torch.tensor([[root]], dtype=torch.float32),
            torch.tensor([[root]], dtype=torch.float32),
        )
        selection = select_rank(
            candidate,
            torch.ones(1, 1),
            allowed_ranks=[1],
            maximum_rank=1,
            spectral_energy_threshold=1.0,
            functional_error_threshold=0.0,
            epsilon=1.0e-8,
            absolute_tolerance=1.0e-4,
        )

        self.assertTrue(selection.feasible, selection.reason)
        self.assertEqual(selection.rank, 1)
        self.assertEqual(len(selection.diagnostics), 1)
        self.assertLess(
            selection.diagnostics[0].functional_error.reference_energy,
            1.0e-8,
        )
        self.assertGreater(float(selection.factors.b @ selection.factors.a), 0.0)

    def test_factor_rank_with_zero_numerical_spectrum_is_canonical_zero(self):
        candidate = LowRankFactors(
            torch.tensor([[1.0], [0.0]]),
            torch.zeros(1, 2),
        )
        selection = select_rank(
            candidate,
            torch.eye(2),
            allowed_ranks=[1, 2],
            maximum_rank=2,
        )

        self.assertTrue(selection.feasible, selection.reason)
        self.assertEqual(selection.rank, 0)
        self.assertEqual(selection.reason, "canonical_zero_adapter")


if __name__ == "__main__":
    unittest.main()
