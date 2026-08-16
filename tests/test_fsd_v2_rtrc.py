import math
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from fd_psc.activation_subspace import (
    ActivationSubspace,
    conv2d_group_activation_matrices,
)
from fd_psc.low_rank_merge import LowRankFactors
from fd_psc.v2.geometry import ReplayGeometry
from fd_psc.v2.rtrc import RTRCLayerInput, project_full_depth


@dataclass
class _RTRCTestConfig:
    use_shared_dual: bool = True
    tail_mode: str = "conservative_isotropic"
    bisection_iterations: int = 120
    bisection_relative_tolerance: float = 1.0e-11
    epsilon: float = 1.0e-12


def _orthonormal(input_dim: int, rank: int, seed: int) -> torch.Tensor:
    if rank == 0:
        return torch.empty(input_dim, 0, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(input_dim, rank, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    return q


def _factors(out_features: int, rank: int, in_features: int, seed: int, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return LowRankFactors(
        torch.randn(out_features, rank, generator=generator, dtype=dtype),
        torch.randn(rank, in_features, generator=generator, dtype=dtype),
    )


def _geometry(
    input_dim: int,
    eigenvalues,
    *,
    tail: float = 0.0,
    seed: int = 1,
    q: torch.Tensor = None,
    sample_count: int = 64,
) -> ReplayGeometry:
    values = torch.as_tensor(eigenvalues, dtype=torch.float64)
    basis = _orthonormal(input_dim, int(values.numel()), seed) if q is None else q
    return ReplayGeometry(
        q=basis,
        eigenvalues=values.to(device=basis.device),
        tail_upper_bound=float(tail),
        sample_count=int(sample_count),
        input_dim=int(input_dim),
        output_energy=3.0,
    )


def _empty_geometry(input_dim: int) -> ReplayGeometry:
    return ReplayGeometry(
        q=torch.empty(input_dim, 0, dtype=torch.float32),
        eigenvalues=torch.empty(0, dtype=torch.float64),
        tail_upper_bound=0.0,
        sample_count=0,
        input_dim=input_dim,
        output_energy=0.0,
    )


def _sigma_bar(geometry: ReplayGeometry) -> torch.Tensor:
    q = geometry.q.to(dtype=torch.float64)
    eigenvalues = geometry.eigenvalues.to(dtype=torch.float64)
    tail = float(geometry.tail_upper_bound)
    identity = torch.eye(geometry.input_dim, dtype=torch.float64, device=q.device)
    if q.shape[1] == 0:
        return tail * identity
    return (
        tail * identity
        + q @ torch.diag(eigenvalues - tail) @ q.transpose(0, 1)
    )


def _dense(factors: LowRankFactors) -> torch.Tensor:
    # Dense task materialization is deliberately confined to this small-matrix
    # reference test module; the production RTRC path remains factor-only.
    return factors.B @ factors.A


def _weighted_dense_drift(
    factors: LowRankFactors,
    geometry: ReplayGeometry,
    omega: float,
) -> float:
    value = _dense(factors).to(dtype=torch.float64)
    sigma = _sigma_bar(geometry).to(device=value.device)
    return float((float(omega) * torch.sum((value @ sigma) * value)).item())


class RTRCMathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(100)
        self.config = _RTRCTestConfig()

    def test_raw_update_feasible_returns_eta_zero_and_exact_task(self):
        task = _factors(5, 2, 4, seed=2)
        geometry = _geometry(4, [4.0, 1.5, 0.5], seed=3)
        result = project_full_depth(
            [RTRCLayerInput("linear", task, geometry, omega=0.75)],
            beta=1.0,
            config=self.config,
        )

        accepted = result.layers["linear"].accepted
        self.assertEqual(result.eta, 0.0)
        torch.testing.assert_close(accepted.B, task.B, rtol=0.0, atol=0.0)
        torch.testing.assert_close(accepted.A, task.A, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(result.raw_drift, result.accepted_drift, places=12)
        self.assertAlmostEqual(result.delta, result.raw_drift, places=12)

    def test_active_budget_saturates_shared_constraint(self):
        task = _factors(6, 3, 5, seed=4)
        geometry = _geometry(5, [6.0, 2.0, 0.8], tail=0.25, seed=5)
        result = project_full_depth(
            [RTRCLayerInput("active", task, geometry, omega=0.4)],
            beta=0.23,
            config=self.config,
        )

        self.assertGreater(result.eta, 0.0)
        self.assertTrue(math.isfinite(result.eta))
        self.assertLessEqual(result.accepted_drift, result.delta + 1.0e-10)
        relative = abs(result.accepted_drift - result.delta) / max(
            result.delta, self.config.epsilon
        )
        self.assertLess(relative, 5.0e-10)

    def test_positive_budget_preserves_realized_matrix_rank(self):
        task = _factors(7, 3, 6, seed=6)
        geometry = _geometry(
            6,
            [8.0, 5.0, 3.0, 1.5, 0.7, 0.2],
            seed=7,
        )
        result = project_full_depth(
            [RTRCLayerInput("ranked", task, geometry, omega=1.3)],
            beta=0.11,
            config=self.config,
        )
        accepted = result.layers["ranked"].accepted

        self.assertEqual(
            int(torch.linalg.matrix_rank(_dense(task)).item()),
            int(torch.linalg.matrix_rank(_dense(accepted)).item()),
        )
        self.assertEqual(result.layers["ranked"].rank_before, task.rank)
        self.assertEqual(result.layers["ranked"].rank_after, task.rank)

    def test_eta_monotonicity_of_drift_and_distortion(self):
        task = _factors(5, 3, 5, seed=8)
        geometry = _geometry(5, [9.0, 4.0, 1.0], tail=0.4, seed=9)
        layer = RTRCLayerInput("monotone", task, geometry, omega=0.65)
        results = [
            project_full_depth([layer], beta=beta, config=self.config)
            for beta in (0.8, 0.4, 0.1)
        ]

        self.assertLess(results[0].eta, results[1].eta)
        self.assertLess(results[1].eta, results[2].eta)
        self.assertGreater(results[0].accepted_drift, results[1].accepted_drift)
        self.assertGreater(results[1].accepted_drift, results[2].accepted_drift)
        distortions = [value.layers["monotone"].distortion_frobenius for value in results]
        self.assertLess(distortions[0], distortions[1])
        self.assertLess(distortions[1], distortions[2])

    def test_factor_projection_equals_dense_conservative_reference(self):
        task = _factors(6, 3, 5, seed=10)
        geometry = _geometry(5, [7.0, 2.5, 0.9], tail=0.2, seed=11)
        omega = 0.7
        result = project_full_depth(
            [RTRCLayerInput("reference", task, geometry, omega)],
            beta=0.37,
            config=self.config,
        )
        accepted = result.layers["reference"].accepted
        sigma = _sigma_bar(geometry)
        inverse = torch.linalg.inv(
            torch.eye(geometry.input_dim, dtype=torch.float64)
            + result.eta * omega * sigma
        )
        expected = _dense(task) @ inverse

        torch.testing.assert_close(_dense(accepted), expected, rtol=2.0e-11, atol=2.0e-11)
        self.assertAlmostEqual(
            result.layers["reference"].distortion_frobenius,
            float(torch.linalg.vector_norm(expected - _dense(task))),
            places=10,
        )

    def test_three_layers_use_one_shared_eta_and_one_global_budget(self):
        definitions = (
            ("a", _factors(4, 2, 3, 12), _geometry(3, [5.0, 1.0], tail=0.2, seed=13), 0.4),
            ("b", _factors(6, 3, 5, 14), _geometry(5, [8.0, 3.0, 0.7], tail=0.1, seed=15), 1.1),
            ("c", _factors(3, 2, 4, 16), _geometry(4, [2.0, 0.8], tail=0.3, seed=17), 2.0),
        )
        layers = [
            RTRCLayerInput(logical_id, task, geometry, omega)
            for logical_id, task, geometry, omega in definitions
        ]
        result = project_full_depth(layers, beta=0.29, config=self.config)

        self.assertGreater(result.eta, 0.0)
        self.assertAlmostEqual(
            math.fsum(value.accepted_drift for value in result.layers.values()),
            result.accepted_drift,
            places=12,
        )
        self.assertLessEqual(result.accepted_drift, result.delta + 1.0e-10)
        self.assertLess(
            abs(result.accepted_drift - result.delta) / result.delta,
            5.0e-10,
        )
        for logical_id, task, geometry, omega in definitions:
            sigma = _sigma_bar(geometry)
            expected = _dense(task) @ torch.linalg.inv(
                torch.eye(task.in_features, dtype=torch.float64)
                + result.eta * omega * sigma
            )
            torch.testing.assert_close(
                _dense(result.layers[logical_id].accepted),
                expected,
                rtol=3.0e-11,
                atol=3.0e-11,
            )

    def test_cold_start_accepts_all_tasks_without_geometry(self):
        tasks = {
            "first": _factors(4, 2, 3, 18),
            "second": _factors(3, 1, 5, 19),
        }
        result = project_full_depth(
            [
                RTRCLayerInput(key, task, _empty_geometry(task.in_features), 1.0)
                for key, task in tasks.items()
            ],
            beta=0.2,
            config=self.config,
        )

        self.assertEqual(result.eta, 0.0)
        self.assertEqual(result.raw_drift, 0.0)
        self.assertEqual(result.accepted_drift, 0.0)
        self.assertEqual(result.delta, 0.0)
        for key, task in tasks.items():
            torch.testing.assert_close(result.layers[key].accepted.B, task.B)
            torch.testing.assert_close(result.layers[key].accepted.A, task.A)

    def test_exact_zero_drift_skips_root_finding_without_nan(self):
        task = LowRankFactors(
            torch.zeros(4, 2, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64),
        )
        geometry = _geometry(3, [4.0, 1.0], tail=0.2, seed=22)
        result = project_full_depth(
            [RTRCLayerInput("zero-drift", task, geometry, 1.0)],
            beta=0.2,
            config=self.config,
        )
        self.assertEqual(result.raw_drift, 0.0)
        self.assertEqual(result.delta, 0.0)
        self.assertEqual(result.eta, 0.0)
        self.assertEqual(result.accepted_drift, 0.0)
        self.assertTrue(torch.isfinite(result.layers["zero-drift"].accepted.A).all())

    def test_exact_thin_svd_isotropic_completion_dominates_empirical_covariance(self):
        generator = torch.Generator().manual_seed(24)
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=str(dtype)):
                activations = torch.randn(
                    41,
                    9,
                    generator=generator,
                    dtype=torch.float32,
                ).to(dtype=dtype)
                subspace, tail = ActivationSubspace.from_activations_with_tail(
                    activations,
                    maximum_rank=3,
                    spectral_energy_threshold=0.7,
                )
                geometry = ReplayGeometry(
                    q=subspace.q,
                    eigenvalues=subspace.energies.to(torch.float64),
                    tail_upper_bound=tail,
                    sample_count=activations.shape[0],
                    input_dim=activations.shape[1],
                    output_energy=1.0,
                )
                sigma_hat = _sigma_bar(geometry)
                h64 = activations.to(torch.float32).to(torch.float64)
                empirical = h64.transpose(0, 1) @ h64 / float(h64.shape[0])
                minimum = float(
                    torch.linalg.eigvalsh(sigma_hat - empirical).min().item()
                )
                tolerance = 2.0e-5 * max(
                    1.0,
                    float(torch.linalg.eigvalsh(empirical).max().item()),
                )
                self.assertGreaterEqual(minimum, -tolerance)

    def test_conservative_tail_bounds_true_full_spectrum_drift(self):
        input_dim = 6
        full_basis = _orthonormal(input_dim, input_dim, seed=20)
        full_eigenvalues = torch.tensor(
            [10.0, 5.0, 2.0, 0.8, 0.3, 0.05],
            dtype=torch.float64,
        )
        retained = 2
        geometry = _geometry(
            input_dim,
            full_eigenvalues[:retained],
            tail=float(full_eigenvalues[retained]),
            q=full_basis[:, :retained],
        )
        task = _factors(5, 3, input_dim, seed=21)
        omega = 0.9
        result = project_full_depth(
            [RTRCLayerInput("tail", task, geometry, omega)],
            beta=0.18,
            config=self.config,
        )
        accepted = result.layers["tail"].accepted
        full_sigma = full_basis @ torch.diag(full_eigenvalues) @ full_basis.transpose(0, 1)
        dense_accepted = _dense(accepted)
        actual_full_drift = float(
            (omega * torch.sum((dense_accepted @ full_sigma) * dense_accepted)).item()
        )

        self.assertLessEqual(
            actual_full_drift,
            result.layers["tail"].accepted_drift + 1.0e-10,
        )
        self.assertLessEqual(actual_full_drift, result.delta + 1.0e-10)

    def test_grouped_conv_geometry_has_no_cross_group_mixing_and_projects_each_group(self):
        conv = torch.nn.Conv2d(
            4,
            6,
            kernel_size=2,
            stride=1,
            padding=1,
            groups=2,
            bias=False,
        )
        inputs = torch.randn(2, 4, 5, 6)
        matrices = conv2d_group_activation_matrices(conv, inputs)
        perturbed = inputs.clone()
        perturbed[:, 2:] += 100.0
        changed = conv2d_group_activation_matrices(conv, perturbed)

        self.assertEqual(len(matrices), 2)
        self.assertEqual(matrices[0].shape[1], 2 * 2 * 2)
        self.assertTrue(torch.equal(matrices[0], changed[0]))
        self.assertFalse(torch.equal(matrices[1], changed[1]))

        layers = []
        references = {}
        for group_index, activations in enumerate(matrices):
            _, singular_values, vh = torch.linalg.svd(
                activations.to(dtype=torch.float32),
                full_matrices=False,
            )
            spectrum = singular_values.to(dtype=torch.float64).square() / activations.shape[0]
            retained = 3
            geometry = ReplayGeometry(
                q=vh[:retained].transpose(0, 1).contiguous(),
                eigenvalues=spectrum[:retained],
                tail_upper_bound=float(spectrum[retained]),
                sample_count=activations.shape[0],
                input_dim=activations.shape[1],
                output_energy=2.0 + group_index,
            )
            logical_id = f"conv::group={group_index}"
            task = _factors(3, 2, activations.shape[1], seed=30 + group_index)
            omega = 0.5 + group_index
            layers.append(RTRCLayerInput(logical_id, task, geometry, omega))
            references[logical_id] = (task, geometry, omega)

        result = project_full_depth(layers, beta=0.31, config=self.config)
        for logical_id, (task, geometry, omega) in references.items():
            accepted = result.layers[logical_id].accepted
            self.assertEqual(accepted.in_features, 8)
            self.assertEqual(accepted.out_features, 3)
            sigma = _sigma_bar(geometry)
            expected = _dense(task) @ torch.linalg.inv(
                torch.eye(8, dtype=torch.float64) + result.eta * omega * sigma
            )
            torch.testing.assert_close(_dense(accepted), expected, rtol=2.0e-6, atol=2.0e-6)


class RTRCValidationAndPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = _RTRCTestConfig()
        self.task = _factors(4, 2, 3, seed=40)
        self.geometry = _geometry(3, [3.0, 1.0], tail=0.2, seed=41)

    def test_rejects_duplicate_ids_bad_dimensions_nonfinite_factors_and_omega(self):
        layer = RTRCLayerInput("same", self.task, self.geometry, 1.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            project_full_depth([layer, layer], 0.2, self.config)

        wrong_geometry = _geometry(4, [2.0], seed=42)
        with self.assertRaisesRegex(ValueError, "input dimensions"):
            project_full_depth(
                [RTRCLayerInput("wrong", self.task, wrong_geometry, 1.0)],
                0.2,
                self.config,
            )

        nonfinite = LowRankFactors(
            self.task.B.clone(),
            self.task.A.clone(),
        )
        nonfinite.A[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            project_full_depth(
                [RTRCLayerInput("nan", nonfinite, self.geometry, 1.0)],
                0.2,
                self.config,
            )

        for omega in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                project_full_depth(
                    [RTRCLayerInput("omega", self.task, self.geometry, omega)],
                    0.2,
                    self.config,
                )

    def test_rejects_invalid_geometry_shared_dual_tail_mode_and_beta(self):
        with self.assertRaises(ValueError):
            invalid_geometry = ReplayGeometry(
                q=torch.ones(3, 1, dtype=torch.float64) * 2.0,
                eigenvalues=torch.tensor([1.0], dtype=torch.float64),
                tail_upper_bound=0.0,
                sample_count=4,
                input_dim=3,
                output_energy=1.0,
            )
            project_full_depth(
                [RTRCLayerInput("bad-q", self.task, invalid_geometry, 1.0)],
                0.2,
                self.config,
            )

        with self.assertRaisesRegex(ValueError, "shared dual"):
            project_full_depth(
                [RTRCLayerInput("dual", self.task, self.geometry, 1.0)],
                0.2,
                SimpleNamespace(**{**self.config.__dict__, "use_shared_dual": False}),
            )
        with self.assertRaisesRegex(ValueError, "tail_mode"):
            project_full_depth(
                [RTRCLayerInput("tail", self.task, self.geometry, 1.0)],
                0.2,
                SimpleNamespace(**{**self.config.__dict__, "tail_mode": "discarded_zero"}),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            project_full_depth(
                [RTRCLayerInput("beta", self.task, self.geometry, 1.0)],
                -0.1,
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            project_full_depth(
                [RTRCLayerInput("zero-beta", self.task, self.geometry, 1.0)],
                0.0,
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            project_full_depth(
                [RTRCLayerInput("large-beta", self.task, self.geometry, 1.0)],
                1.01,
                self.config,
            )

    def test_float64_statistics_match_ill_conditioned_dense_reference(self):
        b = torch.tensor(
            [[1.0e8, 1.0], [1.0e8, -1.0], [1.0, 2.0]],
            dtype=torch.float64,
        )
        a = torch.tensor(
            [[1.0e-8, -2.0e-8, 3.0e-8], [1.0, -2.0, 0.5]],
            dtype=torch.float64,
        )
        task = LowRankFactors(b, a)
        geometry = _geometry(
            3,
            [4.0, 1.5],
            tail=0.25,
            q=torch.eye(3, dtype=torch.float64)[:, :2],
        )
        omega = 0.37
        result = project_full_depth(
            [RTRCLayerInput("precision", task, geometry, omega)],
            beta=1.0,
            config=self.config,
        )
        expected = _weighted_dense_drift(task, geometry, omega)
        self.assertAlmostEqual(result.raw_drift, expected, places=11)

    def test_returned_factors_preserve_adapter_dtype(self):
        task = _factors(4, 2, 3, seed=44, dtype=torch.float32)
        result = project_full_depth(
            [RTRCLayerInput("dtype", task, self.geometry, 1.0)],
            beta=0.4,
            config=self.config,
        )
        accepted = result.layers["dtype"].accepted
        self.assertEqual(accepted.B.dtype, torch.float32)
        self.assertEqual(accepted.A.dtype, torch.float32)
        self.assertTrue(torch.isfinite(accepted.B).all())
        self.assertTrue(torch.isfinite(accepted.A).all())
        tolerance = max(1.0e-7, 5.0e-6 * result.delta)
        self.assertLessEqual(result.accepted_drift, result.delta + tolerance)

    def test_float16_cast_is_rebudgeted_on_the_feasible_side(self):
        torch.manual_seed(49)
        input_dim = 8
        q, _ = torch.linalg.qr(torch.randn(input_dim, 4), mode="reduced")
        geometry = ReplayGeometry(
            q=q,
            eigenvalues=torch.tensor([10.0, 3.0, 0.8, 0.2], dtype=torch.float64),
            tail_upper_bound=0.05,
            sample_count=50,
            input_dim=input_dim,
            output_energy=2.0,
        )
        task = LowRankFactors(
            torch.randn(7, 3, dtype=torch.float16),
            torch.randn(3, input_dim, dtype=torch.float16),
        )
        result = project_full_depth(
            [RTRCLayerInput("half", task, geometry, 0.7)],
            beta=0.083,
            config=self.config,
        )

        self.assertTrue(math.isfinite(result.eta))
        self.assertEqual(result.layers["half"].accepted.A.dtype, torch.float16)
        self.assertLessEqual(result.accepted_drift, result.delta)


if __name__ == "__main__":
    unittest.main()
