"""Canonical low-rank algebra for FD-PSC.

All factors follow one convention: ``B @ A`` is already the actual effective
weight delta (LoRA scaling has been absorbed exactly once).  Production paths
use thin QR and a small core SVD; no ``dout x din`` candidate is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


@dataclass(frozen=True)
class LowRankFactors:
    b: torch.Tensor
    a: torch.Tensor

    def __post_init__(self) -> None:
        if self.b.ndim != 2 or self.a.ndim != 2:
            raise ValueError("low-rank factors must both be matrices")
        if self.b.shape[1] != self.a.shape[0]:
            raise ValueError(f"factor inner dimensions differ: {self.b.shape[1]} != {self.a.shape[0]}")
        if self.b.device != self.a.device:
            raise ValueError("low-rank factors must share a device")

    @property
    def B(self) -> torch.Tensor:  # compatibility with mathematical notation
        return self.b

    @property
    def A(self) -> torch.Tensor:
        return self.a

    @property
    def rank(self) -> int:
        return int(self.b.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.b.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.a.shape[1])

    def to(self, *args, **kwargs) -> "LowRankFactors":
        return LowRankFactors(self.b.to(*args, **kwargs), self.a.to(*args, **kwargs))

    @classmethod
    def zeros(
        cls,
        out_features: int,
        in_features: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "LowRankFactors":
        return cls(
            torch.empty(out_features, 0, device=device, dtype=dtype),
            torch.empty(0, in_features, device=device, dtype=dtype),
        )

    def apply(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the represented matrix to vectors in the final dimension."""

        return (inputs @ self.a.transpose(0, 1)) @ self.b.transpose(0, 1)


FactorLike = Union[LowRankFactors, Tuple[torch.Tensor, torch.Tensor]]


def as_factors(value: FactorLike) -> LowRankFactors:
    if isinstance(value, LowRankFactors):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return LowRankFactors(value[0], value[1])
    b = getattr(value, "b", getattr(value, "B", None))
    a = getattr(value, "a", getattr(value, "A", None))
    if isinstance(b, torch.Tensor) and isinstance(a, torch.Tensor):
        return LowRankFactors(b, a)
    raise TypeError("expected LowRankFactors, a (B, A) tuple, or an object exposing B/A")


@dataclass(frozen=True)
class FactorSVD:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    used_float64_fallback: bool = False

    @property
    def numerical_rank(self) -> int:
        if self.singular_values.numel() == 0:
            return 0
        tolerance = (
            max(self.u.shape[0], self.vh.shape[1])
            * torch.finfo(self.singular_values.dtype).eps
            * float(self.singular_values[0].detach().cpu())
        )
        return int(torch.count_nonzero(self.singular_values > tolerance).item())


def _factor_svd_impl(factors: LowRankFactors, dtype: torch.dtype) -> FactorSVD:
    b = factors.b.to(dtype=dtype)
    a = factors.a.to(dtype=dtype)
    if factors.rank == 0:
        return FactorSVD(
            torch.empty(factors.out_features, 0, device=b.device, dtype=dtype),
            torch.empty(0, device=b.device, dtype=dtype),
            torch.empty(0, factors.in_features, device=b.device, dtype=dtype),
        )
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
    core = rb @ ra.transpose(0, 1)
    uc, singular_values, vhc = torch.linalg.svd(core, full_matrices=False)
    u = qb @ uc
    vh = vhc @ qa.transpose(0, 1)
    if not (torch.isfinite(u).all() and torch.isfinite(singular_values).all() and torch.isfinite(vh).all()):
        raise RuntimeError("non-finite factor-space SVD")
    return FactorSVD(u, singular_values, vh)


def factor_svd(factors: FactorLike) -> FactorSVD:
    """Compute the exact compact SVD of ``B @ A`` through a small core."""

    canonical = as_factors(factors)
    try:
        return _factor_svd_impl(canonical, torch.float32)
    except (RuntimeError, torch.linalg.LinAlgError):
        result = _factor_svd_impl(canonical, torch.float64)
        return FactorSVD(result.u, result.singular_values, result.vh, True)


def factors_from_svd(svd: FactorSVD, rank: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> LowRankFactors:
    available = int(svd.singular_values.numel())
    chosen = available if rank is None else max(0, min(int(rank), available))
    out_dtype = dtype or svd.u.dtype
    if chosen == 0:
        return LowRankFactors.zeros(
            svd.u.shape[0], svd.vh.shape[1], device=svd.u.device, dtype=out_dtype
        )
    singular = torch.clamp(svd.singular_values[:chosen], min=0.0)
    roots = torch.sqrt(singular)
    b = (svd.u[:, :chosen] * roots.unsqueeze(0)).to(dtype=out_dtype)
    a = (roots.unsqueeze(1) * svd.vh[:chosen]).to(dtype=out_dtype)
    return LowRankFactors(b, a)


def truncate_factors(factors: FactorLike, rank: int, dtype: Optional[torch.dtype] = None) -> LowRankFactors:
    canonical = as_factors(factors)
    return factors_from_svd(factor_svd(canonical), rank, dtype or canonical.b.dtype)


def concatenate_factors(factors: Iterable[FactorLike]) -> LowRankFactors:
    values = [as_factors(value) for value in factors]
    if not values:
        raise ValueError("at least one factor pair is required")
    out_features = values[0].out_features
    in_features = values[0].in_features
    device = values[0].b.device
    dtype = values[0].b.dtype
    for value in values:
        if value.out_features != out_features or value.in_features != in_features:
            raise ValueError("cannot merge factors with different matrix dimensions")
        if value.b.device != device:
            raise ValueError("cannot merge factors on different devices")
    nonempty = [value for value in values if value.rank > 0]
    if not nonempty:
        return LowRankFactors.zeros(out_features, in_features, device=device, dtype=dtype)
    return LowRankFactors(
        torch.cat([value.b.to(dtype=dtype) for value in nonempty], dim=1),
        torch.cat([value.a.to(dtype=dtype) for value in nonempty], dim=0),
    )


def merge_and_compress(
    slow: FactorLike,
    episodic: FactorLike,
    rank: int,
    dtype: Optional[torch.dtype] = None,
) -> LowRankFactors:
    merged = concatenate_factors((slow, episodic))
    return truncate_factors(merged, rank, dtype or as_factors(slow).b.dtype)


def factor_frobenius_norm_sq(factors: FactorLike) -> torch.Tensor:
    value = as_factors(factors)
    if value.rank == 0:
        return torch.zeros((), device=value.b.device, dtype=torch.float32)
    btb = value.b.to(dtype=torch.float32).transpose(0, 1) @ value.b.to(dtype=torch.float32)
    aat = value.a.to(dtype=torch.float32) @ value.a.to(dtype=torch.float32).transpose(0, 1)
    return torch.clamp(torch.sum(btb * aat), min=0.0)


def apply_factors_to_activation_transpose(factors: FactorLike, activations: torch.Tensor) -> torch.Tensor:
    """Return ``(B A) H.T`` for ``H`` shaped ``N x din``."""

    value = as_factors(factors)
    if activations.ndim != 2 or activations.shape[1] != value.in_features:
        raise ValueError("activation matrix must have shape N x din")
    h_t = activations.to(device=value.a.device, dtype=torch.float32).transpose(0, 1)
    return value.b.to(dtype=torch.float32) @ (value.a.to(dtype=torch.float32) @ h_t)


@dataclass(frozen=True)
class FunctionalError:
    relative_error: float
    absolute_error: float
    reference_energy: float
    finite: bool

    def passes(self, relative_threshold: float, absolute_tolerance: float) -> bool:
        if not self.finite:
            return False
        if self.reference_energy <= absolute_tolerance ** 2:
            return self.absolute_error <= absolute_tolerance
        return self.relative_error <= relative_threshold


def functional_error(
    reference: FactorLike,
    approximation: FactorLike,
    activations: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> FunctionalError:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reference_output = apply_factors_to_activation_transpose(reference, activations)
    approximate_output = apply_factors_to_activation_transpose(approximation, activations)
    error_energy = torch.sum((reference_output - approximate_output) ** 2)
    reference_energy = torch.sum(reference_output ** 2)
    error_norm = torch.sqrt(torch.clamp(error_energy, min=0.0))
    # V2 §16.2 defines epsilon_functional as the ratio of squared Frobenius
    # norms.  ``absolute_error`` remains the ordinary Frobenius norm so the
    # near-zero-reference guard can compare it with the configured absolute
    # numerical tolerance in matching units.
    relative = error_energy / (reference_energy + float(epsilon))
    finite = bool(
        torch.isfinite(error_norm)
        and torch.isfinite(reference_energy)
        and torch.isfinite(relative)
    )
    return FunctionalError(
        float(relative.detach().cpu()) if finite else float("inf"),
        float(error_norm.detach().cpu()) if finite else float("inf"),
        float(reference_energy.detach().cpu()) if finite else float("inf"),
        finite,
    )


def _pad_factor_rank(factors: LowRankFactors, rank: int) -> LowRankFactors:
    """Represent the same matrix at an explicitly selected parameter rank."""

    target = int(rank)
    if target < factors.rank:
        raise ValueError("cannot pad factors to a smaller rank")
    if target == factors.rank:
        return factors
    extra = target - factors.rank
    return LowRankFactors(
        torch.cat(
            [
                factors.b,
                torch.zeros(
                    factors.out_features,
                    extra,
                    device=factors.b.device,
                    dtype=factors.b.dtype,
                ),
            ],
            dim=1,
        ),
        torch.cat(
            [
                factors.a,
                torch.zeros(
                    extra,
                    factors.in_features,
                    device=factors.a.device,
                    dtype=factors.a.dtype,
                ),
            ],
            dim=0,
        ),
    )


def clipped_rank_candidates(
    allowed_ranks: Sequence[int],
    maximum_rank: int,
    out_features: int,
    in_features: int,
) -> List[int]:
    dimension_cap = min(int(maximum_rank), int(out_features), int(in_features))
    if dimension_cap <= 0:
        return []
    return sorted({min(int(rank), dimension_cap) for rank in allowed_ranks if int(rank) > 0})


@dataclass(frozen=True)
class RankDiagnostic:
    rank: int
    spectral_energy: float
    functional_error: FunctionalError
    passed: bool


@dataclass(frozen=True)
class RankSelection:
    feasible: bool
    rank: Optional[int]
    factors: Optional[LowRankFactors]
    diagnostics: Tuple[RankDiagnostic, ...] = field(default_factory=tuple)
    reason: str = ""


def select_rank(
    candidate: FactorLike,
    activations: torch.Tensor,
    allowed_ranks: Sequence[int],
    maximum_rank: int,
    spectral_energy_threshold: float = 0.99,
    functional_error_threshold: float = 0.02,
    epsilon: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> RankSelection:
    """Choose the smallest allowed rank satisfying spectral and functional tests."""

    value = as_factors(candidate)
    if not 0.0 < spectral_energy_threshold <= 1.0:
        raise ValueError("spectral_energy_threshold must be in (0, 1]")
    ranks = clipped_rank_candidates(
        allowed_ranks, maximum_rank, value.out_features, value.in_features
    )
    decomposition = factor_svd(value)
    singular_energy = decomposition.singular_values.to(dtype=torch.float64) ** 2
    total_energy_t = torch.sum(singular_energy)
    total_energy = float(total_energy_t.detach().cpu())
    if not torch.isfinite(total_energy_t):
        return RankSelection(False, None, None, reason="non_finite_spectrum")
    # Scale is not rank: an arbitrarily small but non-zero adapter still has
    # to pass the configured spectral and functional/absolute-output tests.
    # Only an actually zero numerical spectrum is the canonical rank-0 case.
    if decomposition.numerical_rank == 0:
        zero = LowRankFactors.zeros(
            value.out_features,
            value.in_features,
            device=value.b.device,
            dtype=output_dtype or value.b.dtype,
        )
        return RankSelection(True, 0, zero, reason="canonical_zero_adapter")
    if not ranks:
        return RankSelection(False, None, None, reason="no_legal_rank_candidates")

    diagnostics: List[RankDiagnostic] = []
    for rank in ranks:
        actual = min(rank, int(decomposition.singular_values.numel()))
        approximation = factors_from_svd(
            decomposition, actual, output_dtype or value.b.dtype
        )
        energy = float((torch.sum(singular_energy[:actual]) / total_energy_t).detach().cpu())
        error = functional_error(value, approximation, activations, epsilon)
        passed = (
            energy + epsilon >= spectral_energy_threshold
            and error.passes(functional_error_threshold, absolute_tolerance)
        )
        diagnostics.append(RankDiagnostic(rank, energy, error, passed))
        if passed:
            # ``rank`` is a configured/clipped slow parameter rank.  A
            # rank-deficient candidate may have fewer numerical singular
            # directions, but that must not silently create an unconfigured
            # dynamic rank; pad with exact zeros and report both facts via the
            # selected rank and factor-SVD numerical-rank diagnostics.
            approximation = _pad_factor_rank(approximation, rank)
            return RankSelection(True, rank, approximation, tuple(diagnostics))
    return RankSelection(
        False,
        None,
        None,
        tuple(diagnostics),
        "rank_cap_failed_spectral_or_functional_threshold",
    )


rank_select = select_rank
qr_small_svd = factor_svd


__all__ = [
    "FactorSVD",
    "FunctionalError",
    "LowRankFactors",
    "RankDiagnostic",
    "RankSelection",
    "apply_factors_to_activation_transpose",
    "as_factors",
    "clipped_rank_candidates",
    "concatenate_factors",
    "factor_frobenius_norm_sq",
    "factor_svd",
    "factors_from_svd",
    "functional_error",
    "merge_and_compress",
    "qr_small_svd",
    "rank_select",
    "select_rank",
    "truncate_factors",
]
