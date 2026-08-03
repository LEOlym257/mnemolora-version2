"""SDC-LoRA drift control and factor-space Spectral Surgery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from .low_rank_merge import (
    FactorLike,
    LowRankFactors,
    as_factors,
    factor_frobenius_norm_sq,
    factor_svd,
    factors_from_svd,
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    # Hash storage bytes so the check also works for dtypes NumPy may not
    # understand natively (most notably bfloat16).
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BaseSpectrum:
    u: torch.Tensor
    singular_values: torch.Tensor
    v: torch.Tensor
    energy_rank: int
    weight_hash: str

    @property
    def empty(self) -> bool:
        return self.energy_rank == 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "u": self.u.detach().clone(),
            "singular_values": self.singular_values.detach().clone(),
            "v": self.v.detach().clone(),
            "energy_rank": self.energy_rank,
            "weight_hash": self.weight_hash,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "BaseSpectrum":
        return cls(
            state["u"].detach().clone(),
            state["singular_values"].detach().clone(),
            state["v"].detach().clone(),
            int(state["energy_rank"]),
            str(state["weight_hash"]),
        )


def compute_base_spectrum(weight: torch.Tensor, energy_threshold: float = 0.9) -> BaseSpectrum:
    """Cache only the principal left/right singular subspaces of a base weight."""

    if weight.ndim != 2:
        raise ValueError("base spectrum requires a matrix (flatten Conv groups first)")
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1]")
    matrix = weight.detach().to(dtype=torch.float32)
    if not torch.isfinite(matrix).all():
        raise ValueError("base weight contains non-finite values")
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    energies = singular_values.to(dtype=torch.float64).square()
    total = torch.sum(energies)
    if float(total.detach().cpu()) == 0.0:
        rank = 0
    else:
        cumulative = torch.cumsum(energies, dim=0) / total
        indices = torch.nonzero(cumulative >= float(energy_threshold), as_tuple=False)
        rank = int(indices[0, 0].item()) + 1 if indices.numel() else int(singular_values.numel())
    return BaseSpectrum(
        u[:, :rank].contiguous(),
        singular_values[:rank].contiguous(),
        vh[:rank].transpose(0, 1).contiguous(),
        rank,
        _tensor_sha256(weight),
    )


@dataclass(frozen=True)
class DriftResult:
    value: float
    numerator: float
    denominator: float
    available: bool
    reason: str = ""


def spectral_drift(
    delta: FactorLike,
    base_spectrum: BaseSpectrum,
    epsilon: float = 1.0e-8,
) -> DriftResult:
    """Compute principal-subspace drift directly from canonical factors."""

    factors = as_factors(delta)
    denominator_t = factor_frobenius_norm_sq(factors)
    denominator = float(denominator_t.detach().cpu())
    if not torch.isfinite(denominator_t):
        return DriftResult(float("inf"), float("inf"), denominator, False, "non_finite_drift")
    if base_spectrum.empty:
        # A zero-rank base principal subspace contains no protected direction;
        # the §10.2 numerator is exactly zero, including at cold start.
        return DriftResult(0.0, 0.0, denominator, True, "empty_base_spectrum")
    u = base_spectrum.u.to(device=factors.b.device, dtype=torch.float32)
    v = base_spectrum.v.to(device=factors.a.device, dtype=torch.float32)
    small = (u.transpose(0, 1) @ factors.b.to(dtype=torch.float32)) @ (
        factors.a.to(dtype=torch.float32) @ v
    )
    numerator_t = torch.sum(small * small)
    value_t = numerator_t / (denominator_t + float(epsilon))
    finite = bool(torch.isfinite(value_t) and torch.isfinite(numerator_t))
    return DriftResult(
        float(value_t.detach().cpu()) if finite else float("inf"),
        float(numerator_t.detach().cpu()) if finite else float("inf"),
        denominator,
        finite,
        "" if finite else "non_finite_drift",
    )


@dataclass(frozen=True)
class SDCCorrection:
    gradient: torch.Tensor
    principal_gradient: torch.Tensor
    gamma: float
    principal_energy: float
    residual_energy: float
    applied: bool
    reason: str = ""


def sdc_correct_gradient(
    gradient: torch.Tensor,
    base_spectrum: BaseSpectrum,
    minimum_gamma: float = 0.1,
    epsilon: float = 1.0e-8,
    active: bool = True,
) -> SDCCorrection:
    """Apply the event-gated SDC soft correction to an effective gradient."""

    if gradient.ndim != 2:
        raise ValueError("SDC effective gradient must be a matrix")
    if not 0.0 <= float(minimum_gamma) <= 1.0:
        raise ValueError("minimum_gamma must be in [0, 1]")
    zeros = torch.zeros_like(gradient)
    if not active:
        return SDCCorrection(gradient, zeros, 1.0, 0.0, 0.0, False, "sdc_inactive")
    if base_spectrum.empty:
        return SDCCorrection(gradient, zeros, 1.0, 0.0, 0.0, False, "empty_base_spectrum")
    g = gradient.to(dtype=torch.float32)
    if not torch.isfinite(g).all():
        raise ValueError("non-finite gradient supplied to SDC")
    u = base_spectrum.u.to(device=g.device, dtype=torch.float32)
    v = base_spectrum.v.to(device=g.device, dtype=torch.float32)
    principal = u @ ((u.transpose(0, 1) @ g @ v)) @ v.transpose(0, 1)
    residual = g - principal
    ep_t = torch.sum(principal.square())
    er_t = torch.sum(residual.square())
    ep = float(ep_t.detach().cpu())
    er = float(er_t.detach().cpu())
    if ep <= epsilon:
        return SDCCorrection(gradient, principal.to(dtype=gradient.dtype), 1.0, ep, er, False, "zero_principal_gradient")
    if ep + er <= epsilon:
        return SDCCorrection(gradient, principal.to(dtype=gradient.dtype), 1.0, ep, er, False, "zero_total_gradient")
    gamma_t = torch.sqrt(er_t / (ep_t + er_t + float(epsilon)))
    gamma = min(1.0, max(float(minimum_gamma), float(gamma_t.detach().cpu())))
    corrected = g - (1.0 - gamma) * principal
    return SDCCorrection(
        corrected.to(dtype=gradient.dtype),
        principal.to(dtype=gradient.dtype),
        gamma,
        ep,
        er,
        gamma < 1.0,
    )


@dataclass(frozen=True)
class SDCTriggerStatus:
    checked: bool
    active: bool
    drift_signal: bool
    safety_signal: bool
    consecutive_increases: int
    reason: str


class SDCEventTracker:
    """Episode-local event state for one logical layer."""

    def __init__(
        self,
        check_every_replans: int = 4,
        drift_threshold: float = 0.25,
        drift_consecutive_checks: int = 2,
        drift_increase_tolerance: float = 0.01,
        anchor_regression_trigger: float = 0.0,
    ) -> None:
        if check_every_replans <= 0 or drift_consecutive_checks <= 0:
            raise ValueError("SDC check intervals/counts must be positive")
        self.check_every_replans = int(check_every_replans)
        self.drift_threshold = float(drift_threshold)
        self.drift_consecutive_checks = int(drift_consecutive_checks)
        self.drift_increase_tolerance = float(drift_increase_tolerance)
        self.anchor_regression_trigger = float(anchor_regression_trigger)
        self.reset()

    def reset(self) -> None:
        self.previous_drift: Optional[float] = None
        self.consecutive_increases = 0
        self.active = False

    def update(
        self,
        replan_index: int,
        drift: Optional[float],
        anchor_regression: Optional[float] = None,
        anchor_cosine: Optional[float] = None,
    ) -> SDCTriggerStatus:
        if (int(replan_index) + 1) % self.check_every_replans != 0:
            return SDCTriggerStatus(False, self.active, False, False, self.consecutive_increases, "not_scheduled")
        if drift is None or not torch.isfinite(torch.tensor(float(drift))):
            self.active = False
            self.consecutive_increases = 0
            return SDCTriggerStatus(True, False, False, False, 0, "drift_unavailable")
        drift_value = float(drift)
        if self.previous_drift is not None and drift_value - self.previous_drift > self.drift_increase_tolerance:
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0
        self.previous_drift = drift_value
        drift_signal = (
            drift_value > self.drift_threshold
            or self.consecutive_increases >= self.drift_consecutive_checks
        )
        safety_signal = (
            anchor_regression is not None
            and torch.isfinite(torch.tensor(float(anchor_regression)))
            and float(anchor_regression) > self.anchor_regression_trigger
        ) or (
            anchor_cosine is not None
            and torch.isfinite(torch.tensor(float(anchor_cosine)))
            and float(anchor_cosine) < 0.0
        )
        self.active = bool(drift_signal and safety_signal)
        reason = "triggered" if self.active else "joint_trigger_not_met"
        return SDCTriggerStatus(
            True,
            self.active,
            bool(drift_signal),
            bool(safety_signal),
            self.consecutive_increases,
            reason,
        )


def _factor_inner_with_gradient(
    gradient: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    g = gradient.detach().to(device=b.device, dtype=torch.float32)
    return torch.sum(b.to(dtype=torch.float32) * (g @ a.to(dtype=torch.float32).transpose(0, 1)))


def effective_gradient_proxy(
    gradient_difference: torch.Tensor,
    trainable_b: torch.Tensor,
    trainable_a: torch.Tensor,
    scaling: float = 1.0,
    centered_initial: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Factor-only ``<stopgrad(G'-G), DeltaWtrainable>`` proxy scalar."""

    value = float(scaling) * _factor_inner_with_gradient(
        gradient_difference, trainable_b, trainable_a
    )
    if centered_initial is not None:
        initial_b, initial_a = centered_initial
        constant = float(scaling) * _factor_inner_with_gradient(
            gradient_difference,
            initial_b.detach(),
            initial_a.detach(),
        )
        value = value - constant
    return value


sdc_proxy_loss = effective_gradient_proxy


@dataclass(frozen=True)
class BoxSphereProjection:
    scales: torch.Tensor
    feasible: bool
    iterations: int
    weighted_norm: float
    reason: str = ""


def project_box_weighted_l2(
    proposed_scales: torch.Tensor,
    singular_values: torch.Tensor,
    minimum_scale: float,
    maximum_scale: float,
    target_norm: Optional[float] = None,
    tolerance: float = 1.0e-6,
    maximum_iterations: int = 128,
) -> BoxSphereProjection:
    """Numerically project scales onto box and weighted-L2 constraints jointly.

    A clipped radial path is monotone from the lower to upper box corner, so a
    bisection finds an intersection without performing a post-clip rescale that
    could leave the box.
    """

    if proposed_scales.ndim != 1 or singular_values.ndim != 1 or proposed_scales.shape != singular_values.shape:
        raise ValueError("scale and singular-value vectors must have the same shape")
    if not 0.0 < float(minimum_scale) <= float(maximum_scale):
        raise ValueError("invalid spectral scale box")
    scales = proposed_scales.detach().to(dtype=torch.float64)
    weights = torch.abs(singular_values.detach().to(device=scales.device, dtype=torch.float64))
    if not torch.isfinite(scales).all() or not torch.isfinite(weights).all():
        return BoxSphereProjection(proposed_scales, False, 0, float("inf"), "non_finite_projection_input")
    desired = float(torch.linalg.vector_norm(weights).detach().cpu()) if target_norm is None else float(target_norm)
    minimum_norm = float(torch.linalg.vector_norm(weights * float(minimum_scale)).detach().cpu())
    maximum_norm = float(torch.linalg.vector_norm(weights * float(maximum_scale)).detach().cpu())
    if desired < minimum_norm - tolerance or desired > maximum_norm + tolerance:
        return BoxSphereProjection(proposed_scales, False, 0, float("nan"), "empty_box_sphere_intersection")
    if weights.numel() == 0 or maximum_norm == 0.0:
        clipped = torch.clamp(scales, minimum_scale, maximum_scale).to(dtype=proposed_scales.dtype)
        return BoxSphereProjection(clipped, abs(desired) <= tolerance, 0, 0.0)

    base = torch.clamp(scales, minimum_scale, maximum_scale)

    def candidate(multiplier: float) -> torch.Tensor:
        return torch.clamp(base * multiplier, minimum_scale, maximum_scale)

    def norm_of(value: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(weights * value).detach().cpu())

    low, high = 0.0, 1.0
    while norm_of(candidate(high)) < desired and high < 1.0e12:
        high *= 2.0
    result = candidate(high)
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        middle = (low + high) / 2.0
        value = candidate(middle)
        norm = norm_of(value)
        result = value
        if abs(norm - desired) <= tolerance * max(1.0, desired):
            break
        if norm < desired:
            low = middle
        else:
            high = middle
    final_norm = norm_of(result)
    feasible = (
        torch.all(result >= minimum_scale - tolerance)
        and torch.all(result <= maximum_scale + tolerance)
        and abs(final_norm - desired) <= tolerance * max(1.0, desired) * 4.0
    )
    return BoxSphereProjection(
        result.to(dtype=proposed_scales.dtype),
        bool(feasible),
        iterations,
        final_norm,
        "" if feasible else "box_sphere_projection_did_not_converge",
    )


@dataclass(frozen=True)
class SpectralSurgeryResult:
    factors: LowRankFactors
    scales: torch.Tensor
    applied: bool
    reason: str
    used_float64_fallback: bool = False


def spectral_surgery(
    episodic: FactorLike,
    calibration_gradient: torch.Tensor,
    steps: int = 2,
    learning_rate: float = 0.1,
    minimum_scale: float = 0.75,
    maximum_scale: float = 1.25,
    preserve_spectral_l2_norm: bool = True,
    epsilon: float = 1.0e-8,
) -> SpectralSurgeryResult:
    """Optimize singular scales in one fixed factor-derived U/V basis."""

    original = as_factors(episodic)
    decomposition = factor_svd(original)
    singular = decomposition.singular_values
    if singular.numel() == 0 or float(torch.linalg.vector_norm(singular).detach().cpu()) <= epsilon:
        return SpectralSurgeryResult(
            original,
            torch.ones_like(singular),
            False,
            "zero_episodic_spectrum",
            decomposition.used_float64_fallback,
        )
    if calibration_gradient.shape != (original.out_features, original.in_features):
        raise ValueError("calibration gradient shape does not match episodic factors")
    gradient = calibration_gradient.detach().to(device=singular.device, dtype=decomposition.u.dtype)
    if not torch.isfinite(gradient).all():
        return SpectralSurgeryResult(
            original,
            torch.ones_like(singular),
            False,
            "non_finite_calibration_gradient",
            decomposition.used_float64_fallback,
        )
    if float(torch.linalg.vector_norm(gradient).detach().cpu()) <= epsilon:
        return SpectralSurgeryResult(
            original,
            torch.ones_like(singular),
            False,
            "zero_calibration_gradient",
            decomposition.used_float64_fallback,
        )
    scales = torch.ones_like(singular)
    v = decomposition.vh.transpose(0, 1)
    # diag(U.T G V), held fixed for all scalar steps.
    directional = torch.sum(decomposition.u * (gradient @ v), dim=0)
    scale_gradient = singular * directional
    target_norm = float(torch.linalg.vector_norm(singular).detach().cpu())
    for _ in range(max(0, int(steps))):
        proposed = scales - float(learning_rate) * scale_gradient
        if preserve_spectral_l2_norm:
            projection = project_box_weighted_l2(
                proposed,
                singular,
                minimum_scale,
                maximum_scale,
                target_norm,
            )
            if not projection.feasible:
                return SpectralSurgeryResult(
                    original,
                    torch.ones_like(singular),
                    False,
                    projection.reason,
                    decomposition.used_float64_fallback,
                )
            scales = projection.scales
        else:
            scales = torch.clamp(proposed, minimum_scale, maximum_scale)
    adjusted = singular * scales
    adjusted_decomposition = type(decomposition)(
        decomposition.u,
        adjusted,
        decomposition.vh,
        decomposition.used_float64_fallback,
    )
    candidate = factors_from_svd(adjusted_decomposition, dtype=original.b.dtype)
    return SpectralSurgeryResult(
        candidate,
        scales,
        not torch.allclose(scales, torch.ones_like(scales)),
        "candidate_requires_calibration_screening",
        decomposition.used_float64_fallback,
    )


base_svd = compute_base_spectrum
compute_spectral_drift = spectral_drift


__all__ = [
    "BaseSpectrum",
    "BoxSphereProjection",
    "DriftResult",
    "SDCCorrection",
    "SDCEventTracker",
    "SDCTriggerStatus",
    "SpectralSurgeryResult",
    "base_svd",
    "compute_base_spectrum",
    "compute_spectral_drift",
    "effective_gradient_proxy",
    "project_box_weighted_l2",
    "sdc_correct_gradient",
    "sdc_proxy_loss",
    "spectral_drift",
    "spectral_surgery",
]
