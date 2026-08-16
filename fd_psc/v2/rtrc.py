"""Factor-space Representation Trust-Region Consolidation (RTRC).

The production implementation in this module never materializes an effective
``out_features x in_features`` task matrix, an input-space covariance matrix,
or an input-space inverse.  A task is represented canonically as ``B @ A``;
all trust-region operations update only its right factor ``A``.

Per-layer drift values reported here are *weighted contributions* to the
full-depth budget.  Consequently, summing ``raw_drift`` or
``accepted_drift`` over all layer results exactly recovers the corresponding
global result (up to floating-point summation error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from ..low_rank_merge import LowRankFactors, as_factors
from .geometry import ReplayGeometry


class RTRCNumericalError(RuntimeError):
    """Raised when a finite, budget-feasible shared dual cannot be found."""


@dataclass(frozen=True)
class RTRCLayerInput:
    logical_layer_id: str
    task: LowRankFactors
    geometry: ReplayGeometry
    omega: float


@dataclass(frozen=True)
class RTRCLayerResult:
    logical_layer_id: str
    accepted: LowRankFactors
    raw_drift: float
    accepted_drift: float
    distortion_frobenius: float
    rank_before: int
    rank_after: int


@dataclass(frozen=True)
class RTRCResult:
    eta: float
    delta: float
    beta: float
    raw_drift: float
    accepted_drift: float
    layers: Mapping[str, RTRCLayerResult]

    @property
    def distortion_frobenius(self) -> float:
        """Full-depth unweighted task distortion Frobenius norm."""

        return math.sqrt(
            math.fsum(
                value.distortion_frobenius * value.distortion_frobenius
                for value in self.layers.values()
            )
        )


@dataclass(frozen=True)
class _LayerStatistics:
    logical_layer_id: str
    task: LowRankFactors
    q: torch.Tensor
    eigenvalues: torch.Tensor
    gram_b: torch.Tensor
    a64: torch.Tensor
    directional_energy: torch.Tensor
    perpendicular_energy: float
    tail_upper_bound: float
    omega: float
    eta_terms: Tuple[Tuple[float, float], ...]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _validate_config(config: Any) -> Tuple[int, float, float]:
    if not bool(config.use_shared_dual):
        raise ValueError("RTRC requires one shared dual across all logical layers")
    if str(config.tail_mode) != "conservative_isotropic":
        raise ValueError(
            "production RTRC requires tail_mode='conservative_isotropic'"
        )
    iterations = int(config.bisection_iterations)
    if iterations <= 0:
        raise ValueError("bisection_iterations must be positive")
    relative_tolerance = _finite_float(
        config.bisection_relative_tolerance,
        "bisection_relative_tolerance",
    )
    if relative_tolerance <= 0.0:
        raise ValueError("bisection_relative_tolerance must be positive")
    epsilon = _finite_float(config.epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return iterations, relative_tolerance, epsilon


def _validate_geometry_and_task(
    layer: RTRCLayerInput,
) -> Tuple[LowRankFactors, torch.Tensor, torch.Tensor, float, float]:
    logical_id = str(layer.logical_layer_id)
    if not logical_id:
        raise ValueError("logical_layer_id must be non-empty")
    value = as_factors(layer.task)
    if value.out_features <= 0 or value.in_features <= 0:
        raise ValueError(f"{logical_id}: task matrix dimensions must be positive")
    if not value.b.is_floating_point() or not value.a.is_floating_point():
        raise TypeError(f"{logical_id}: RTRC task factors must be floating-point")
    if not torch.isfinite(value.b).all() or not torch.isfinite(value.a).all():
        raise ValueError(f"{logical_id}: RTRC task factors must be finite")

    geometry = layer.geometry
    q = geometry.q
    eigenvalues = geometry.eigenvalues
    if q.ndim != 2 or eigenvalues.ndim != 1:
        raise ValueError(f"{logical_id}: geometry Q/eigenvalue ranks are invalid")
    if int(geometry.input_dim) != value.in_features or q.shape[0] != value.in_features:
        raise ValueError(f"{logical_id}: task and geometry input dimensions differ")
    if q.shape[1] != eigenvalues.shape[0]:
        raise ValueError(f"{logical_id}: geometry Q/eigenvalue counts differ")
    if not q.is_floating_point() or not eigenvalues.is_floating_point():
        raise TypeError(f"{logical_id}: geometry must be floating-point")
    if not torch.isfinite(q).all() or not torch.isfinite(eigenvalues).all():
        raise ValueError(f"{logical_id}: geometry must be finite")
    if torch.any(eigenvalues < 0):
        raise ValueError(f"{logical_id}: geometry eigenvalues must be non-negative")

    omega = _finite_float(layer.omega, f"{logical_id}.omega")
    if omega <= 0.0:
        raise ValueError(f"{logical_id}: omega must be positive")
    tail = _finite_float(
        geometry.tail_upper_bound,
        f"{logical_id}.tail_upper_bound",
    )
    if tail < 0.0:
        raise ValueError(f"{logical_id}: tail_upper_bound must be non-negative")

    q64 = q.detach().to(device=value.a.device, dtype=torch.float64)
    eigenvalues64 = eigenvalues.detach().to(
        device=value.a.device,
        dtype=torch.float64,
    )
    if q64.shape[1]:
        identity = torch.eye(q64.shape[1], device=q64.device, dtype=torch.float64)
        if not torch.allclose(
            q64.transpose(0, 1) @ q64,
            identity,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(f"{logical_id}: geometry Q must have orthonormal columns")
        eigenvalue_list = [float(item) for item in eigenvalues64.detach().cpu()]
        ordering_tolerance = 1.0e-12 * max(1.0, abs(eigenvalue_list[0]))
        if any(
            right > left + ordering_tolerance
            for left, right in zip(eigenvalue_list, eigenvalue_list[1:])
        ):
            raise ValueError(f"{logical_id}: geometry eigenvalues must be non-increasing")
        if tail > eigenvalue_list[-1] + ordering_tolerance:
            raise ValueError(
                f"{logical_id}: tail_upper_bound exceeds the last retained eigenvalue"
            )

    # Snapshot the live episodic factors and sever their optimizer/autograd
    # graph.  Canonical LoRA scaling has already been absorbed into B.
    task = LowRankFactors(
        value.b.detach().clone(),
        value.a.detach().clone(),
    )
    return task, q64, eigenvalues64, tail, omega


def _layer_statistics(layer: RTRCLayerInput) -> _LayerStatistics:
    task, q, eigenvalues, tail, omega = _validate_geometry_and_task(layer)
    b64 = task.b.to(dtype=torch.float64)
    a64 = task.a.to(dtype=torch.float64)
    gram_b = b64.transpose(0, 1) @ b64
    task_frobenius_sq_t = torch.sum(a64 * (gram_b @ a64))
    task_frobenius_sq = max(float(task_frobenius_sq_t.detach().cpu()), 0.0)
    if q.shape[1]:
        u = a64 @ q
        directional = torch.sum(u * (gram_b @ u), dim=0)
        directional = torch.clamp(directional, min=0.0)
    else:
        directional = torch.empty(0, device=a64.device, dtype=torch.float64)
    directional_sum = float(torch.sum(directional).detach().cpu())
    perpendicular = max(task_frobenius_sq - directional_sum, 0.0)
    if not (
        torch.isfinite(gram_b).all()
        and torch.isfinite(directional).all()
        and math.isfinite(task_frobenius_sq)
        and math.isfinite(perpendicular)
    ):
        raise RTRCNumericalError(
            f"non-finite factor-space statistics for {layer.logical_layer_id}"
        )
    return _LayerStatistics(
        logical_layer_id=str(layer.logical_layer_id),
        task=task,
        q=q,
        eigenvalues=eigenvalues,
        gram_b=gram_b,
        a64=a64,
        directional_energy=directional,
        perpendicular_energy=perpendicular,
        tail_upper_bound=tail,
        omega=omega,
        eta_terms=tuple(
            (float(value), float(energy))
            for value, energy in zip(
                eigenvalues.detach().cpu().tolist(),
                directional.detach().cpu().tolist(),
            )
        ),
    )


def _drift_at_eta(statistics: Sequence[_LayerStatistics], eta: float) -> float:
    contributions = []
    eta_value = float(eta)
    for layer in statistics:
        omega = layer.omega
        for eigenvalue, energy in layer.eta_terms:
            denominator = 1.0 + eta_value * omega * eigenvalue
            shrink = 0.0 if math.isinf(denominator) else 1.0 / denominator
            contributions.append(omega * eigenvalue * energy * shrink * shrink)
        if layer.tail_upper_bound > 0.0 and layer.perpendicular_energy > 0.0:
            denominator = 1.0 + eta_value * omega * layer.tail_upper_bound
            shrink = 0.0 if math.isinf(denominator) else 1.0 / denominator
            contributions.append(
                omega
                * layer.tail_upper_bound
                * layer.perpendicular_energy
                * shrink
                * shrink
            )
    value = math.fsum(contributions)
    if not math.isfinite(value) or value < 0.0:
        raise RTRCNumericalError("shared RTRC drift evaluation became non-finite")
    return value


def _solve_eta(
    statistics: Sequence[_LayerStatistics],
    delta: float,
    *,
    iterations: int,
    relative_tolerance: float,
    epsilon: float,
) -> float:
    lo = 0.0
    hi = 1.0
    hi_value = _drift_at_eta(statistics, hi)
    for _ in range(64):
        if hi_value <= delta:
            break
        hi *= 2.0
        if not math.isfinite(hi):
            raise RTRCNumericalError("failed to bracket a finite shared eta")
        hi_value = _drift_at_eta(statistics, hi)
    else:
        raise RTRCNumericalError("failed to bracket eta")

    scale = max(delta, epsilon)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = _drift_at_eta(statistics, mid)
        if value > delta:
            lo = mid
        else:
            hi = mid
            hi_value = value
        # Stop only on the feasible side of the bracket.  This keeps the
        # accepted update conservative even when the first close midpoint is
        # infinitesimally above the budget.
        if hi_value <= delta and (delta - hi_value) <= relative_tolerance * scale:
            break
    if not math.isfinite(hi) or hi_value > delta:
        raise RTRCNumericalError("bisection did not produce a budget-feasible eta")
    return float(hi)


def _transform_right_factor(layer: _LayerStatistics, eta: float) -> LowRankFactors:
    tail = layer.tail_upper_bound
    s0 = 1.0 / (1.0 + float(eta) * layer.omega * tail)
    if layer.q.shape[1]:
        directional_scales = 1.0 / (
            1.0 + float(eta) * layer.omega * layer.eigenvalues
        )
        transformed_a = (
            s0 * layer.a64
            + (
                (layer.a64 @ layer.q)
                * (directional_scales - s0).unsqueeze(0)
            )
            @ layer.q.transpose(0, 1)
        )
    else:
        transformed_a = s0 * layer.a64
    accepted_a = transformed_a.to(dtype=layer.task.a.dtype)
    if not torch.isfinite(accepted_a).all():
        raise RTRCNumericalError(
            f"accepted factor became non-finite for {layer.logical_layer_id}"
        )
    return LowRankFactors(layer.task.b, accepted_a)


def _weighted_drift_for_a(layer: _LayerStatistics, a: torch.Tensor) -> float:
    a64 = a.to(device=layer.a64.device, dtype=torch.float64)
    if layer.q.shape[1]:
        u = a64 @ layer.q
        directional = torch.clamp(
            torch.sum(u * (layer.gram_b @ u), dim=0),
            min=0.0,
        )
        top = torch.sum(layer.eigenvalues * directional)
        directional_sum = float(torch.sum(directional).detach().cpu())
    else:
        top = torch.zeros((), device=a64.device, dtype=torch.float64)
        directional_sum = 0.0
    total = max(
        float(torch.sum(a64 * (layer.gram_b @ a64)).detach().cpu()),
        0.0,
    )
    perpendicular = max(total - directional_sum, 0.0)
    result = layer.omega * (
        float(top.detach().cpu()) + layer.tail_upper_bound * perpendicular
    )
    if not math.isfinite(result) or result < 0.0:
        raise RTRCNumericalError(
            f"accepted drift became non-finite for {layer.logical_layer_id}"
        )
    return result


def _distortion_frobenius(
    layer: _LayerStatistics,
    accepted: LowRankFactors,
) -> float:
    difference = layer.a64 - accepted.a.to(
        device=layer.a64.device,
        dtype=torch.float64,
    )
    squared = torch.sum(difference * (layer.gram_b @ difference))
    value = math.sqrt(max(float(squared.detach().cpu()), 0.0))
    if not math.isfinite(value):
        raise RTRCNumericalError(
            f"task distortion became non-finite for {layer.logical_layer_id}"
        )
    return value


def _accepted_at_eta(
    statistics: Sequence[_LayerStatistics],
    eta: float,
) -> Tuple[Dict[str, LowRankFactors], float]:
    accepted = {
        layer.logical_layer_id: _transform_right_factor(layer, eta)
        for layer in statistics
    }
    drift = math.fsum(
        _weighted_drift_for_a(layer, accepted[layer.logical_layer_id].a)
        for layer in statistics
    )
    return accepted, drift


def _make_cast_update_feasible(
    statistics: Sequence[_LayerStatistics],
    eta: float,
    delta: float,
    *,
    iterations: int,
    relative_tolerance: float,
) -> Tuple[float, Dict[str, LowRankFactors], float]:
    """Tighten a shared eta when adapter-dtype rounding exceeds the budget.

    The analytic dual is solved in float64, but the committed factor must use
    the adapter dtype.  In particular, float16 rounding can move a root that
    is feasible in float64 to the wrong side of the budget.  This second,
    factor-only bracket retains a single shared eta and evaluates the actual
    returned factors after casting.
    """

    lo = float(eta)
    hi = max(math.nextafter(lo, math.inf), lo * 1.001)
    hi_accepted: Dict[str, LowRankFactors] = {}
    hi_drift = math.inf
    for _ in range(64):
        if not math.isfinite(hi):
            raise RTRCNumericalError(
                "failed to bracket adapter-dtype-feasible shared eta"
            )
        hi_accepted, hi_drift = _accepted_at_eta(statistics, hi)
        if hi_drift <= delta:
            break
        lo = hi
        hi *= 1.5
    else:
        raise RTRCNumericalError(
            "failed to make the cast RTRC update budget-feasible"
        )

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        mid_accepted, mid_drift = _accepted_at_eta(statistics, mid)
        if mid_drift <= delta:
            hi = mid
            hi_accepted = mid_accepted
            hi_drift = mid_drift
        else:
            lo = mid
        if (hi - lo) <= relative_tolerance * max(1.0, hi):
            break
    if hi_drift > delta:
        raise RTRCNumericalError(
            "adapter-dtype correction did not satisfy the RTRC budget"
        )
    return float(hi), hi_accepted, float(hi_drift)


def project_full_depth(
    layers: Sequence[RTRCLayerInput],
    beta: float,
    config: Any,
) -> RTRCResult:
    """Project all logical-layer tasks with one shared RTRC dual ``eta``.

    ``delta`` is derived deterministically as ``beta * raw_drift``.  The
    conservative isotropic tail bound is included both while solving the dual
    and while transforming every right factor.
    """

    iterations, relative_tolerance, epsilon = _validate_config(config)
    beta_value = _finite_float(beta, "beta")
    if beta_value < 0.0:
        raise ValueError("beta must be non-negative")
    if beta_value > 1.0:
        raise ValueError("beta budget fraction must not exceed 1")

    statistics = []
    seen = set()
    for layer in layers:
        logical_id = str(layer.logical_layer_id)
        if logical_id in seen:
            raise ValueError(f"duplicate RTRC logical_layer_id: {logical_id}")
        seen.add(logical_id)
        statistics.append(_layer_statistics(layer))
    # A stable logical-layer order makes the float64/Python-float accumulation
    # deterministic even if a caller supplied a mapping-derived sequence.
    statistics.sort(key=lambda value: value.logical_layer_id)

    raw_drift = _drift_at_eta(statistics, 0.0)
    delta = beta_value * raw_drift
    if not math.isfinite(delta):
        raise RTRCNumericalError("RTRC drift budget became non-finite")

    # The exact zero-drift case is already feasible and must not enter root
    # finding.  A merely small positive drift still receives the requested
    # relative budget; treating it as zero could accept more than delta.
    if raw_drift == 0.0 or raw_drift <= delta:
        eta = 0.0
    else:
        if beta_value == 0.0:
            raise ValueError(
                "beta must be positive when non-zero drift requires a finite shared eta"
            )
        eta = _solve_eta(
            statistics,
            delta,
            iterations=iterations,
            relative_tolerance=relative_tolerance,
            epsilon=epsilon,
        )

    accepted_by_id, accepted_drift = _accepted_at_eta(statistics, eta)
    if eta > 0.0 and accepted_drift > delta:
        eta, accepted_by_id, accepted_drift = _make_cast_update_feasible(
            statistics,
            eta,
            delta,
            iterations=iterations,
            relative_tolerance=relative_tolerance,
        )

    results: Dict[str, RTRCLayerResult] = {}
    for layer in statistics:
        accepted = accepted_by_id[layer.logical_layer_id]
        raw_layer = _weighted_drift_for_a(layer, layer.task.a)
        accepted_layer = _weighted_drift_for_a(layer, accepted.a)
        results[layer.logical_layer_id] = RTRCLayerResult(
            logical_layer_id=layer.logical_layer_id,
            accepted=accepted,
            raw_drift=float(raw_layer),
            accepted_drift=float(accepted_layer),
            distortion_frobenius=float(
                _distortion_frobenius(layer, accepted)
            ),
            rank_before=layer.task.rank,
            rank_after=accepted.rank,
        )

    accepted_drift = math.fsum(value.accepted_drift for value in results.values())
    return RTRCResult(
        eta=float(eta),
        delta=float(delta),
        beta=float(beta_value),
        raw_drift=float(raw_drift),
        accepted_drift=float(accepted_drift),
        layers=results,
    )


__all__ = [
    "RTRCLayerInput",
    "RTRCLayerResult",
    "RTRCNumericalError",
    "RTRCResult",
    "project_full_depth",
]
