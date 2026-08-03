"""Gradient geometry primitives used by FD-PSC.

The functions in this module operate on *effective-weight* gradients.  They do
not know about LoRA factors and intentionally never project ``A`` and ``B``
separately.  This keeps the geometry identical for Linear layers and flattened
Conv2d logical groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CosineResult:
    value: Optional[float]
    available: bool
    current_norm: float
    reference_norm: float
    reason: str = ""


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("gradient values must be torch.Tensor instances")
    if not (value.is_floating_point() or value.is_complex()):
        value = value.float()
    return value


def frobenius_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return a float32/float64 Frobenius inner product without reshaping copies."""

    left = _as_float_tensor(left)
    right = _as_float_tensor(right)
    if left.shape != right.shape:
        raise ValueError(f"gradient shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    dtype = torch.float64 if left.dtype == torch.float64 or right.dtype == torch.float64 else torch.float32
    return torch.sum(left.to(dtype=dtype) * right.to(device=left.device, dtype=dtype))


def gradient_cosine(
    current: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> CosineResult:
    """Compute a Frobenius cosine, marking zero/non-finite inputs unavailable.

    An absent historical/anchor gradient is not a cosine of zero.  Returning an
    explicit availability flag prevents cold start from being treated as a
    conflict trigger.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    inner = frobenius_inner(current, reference)
    dtype = inner.dtype
    cur = current.to(dtype=dtype)
    ref = reference.to(device=current.device, dtype=dtype)
    cur_norm_t = torch.linalg.vector_norm(cur)
    ref_norm_t = torch.linalg.vector_norm(ref)
    cur_norm = float(cur_norm_t.detach().cpu())
    ref_norm = float(ref_norm_t.detach().cpu())
    if not torch.isfinite(cur_norm_t) or not torch.isfinite(ref_norm_t) or not torch.isfinite(inner):
        return CosineResult(None, False, cur_norm, ref_norm, "non_finite_gradient")
    if cur_norm <= epsilon:
        return CosineResult(None, False, cur_norm, ref_norm, "zero_current_gradient")
    if ref_norm <= epsilon:
        return CosineResult(None, False, cur_norm, ref_norm, "zero_reference_gradient")
    value = inner / (cur_norm_t * ref_norm_t + float(epsilon))
    value = torch.clamp(value, -1.0, 1.0)
    return CosineResult(float(value.detach().cpu()), True, cur_norm, ref_norm)


def cosine_or_none(
    current: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> Optional[float]:
    return gradient_cosine(current, reference, epsilon).value


@dataclass(frozen=True)
class ConflictStatus:
    cosine: Optional[float]
    ema: Optional[float]
    consecutive_conflicts: int
    triggered: bool
    available: bool


class ConflictEMA:
    """Per-logical-layer EMA and consecutive-conflict trigger state."""

    def __init__(
        self,
        beta: float = 0.8,
        threshold: float = -0.1,
        consecutive_required: int = 2,
    ) -> None:
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if consecutive_required <= 0:
            raise ValueError("consecutive_required must be positive")
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.consecutive_required = int(consecutive_required)
        self.ema: Optional[float] = None
        self.consecutive_conflicts = 0

    def reset(self) -> None:
        self.ema = None
        self.consecutive_conflicts = 0

    def update(self, cosine: Optional[float]) -> ConflictStatus:
        if cosine is None or not torch.isfinite(torch.tensor(float(cosine))):
            self.consecutive_conflicts = 0
            return ConflictStatus(None, self.ema, 0, False, False)
        value = float(cosine)
        self.ema = value if self.ema is None else self.beta * self.ema + (1.0 - self.beta) * value
        if self.ema < self.threshold:
            self.consecutive_conflicts += 1
        else:
            self.consecutive_conflicts = 0
        return ConflictStatus(
            value,
            self.ema,
            self.consecutive_conflicts,
            self.consecutive_conflicts >= self.consecutive_required,
            True,
        )

    def state_dict(self) -> Dict[str, object]:
        return {"ema": self.ema, "consecutive_conflicts": self.consecutive_conflicts}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        ema = state.get("ema")
        self.ema = None if ema is None else float(ema)
        self.consecutive_conflicts = int(state.get("consecutive_conflicts", 0))


@dataclass(frozen=True)
class PCGradResult:
    gradient: torch.Tensor
    corrected: bool
    dot_before: float
    correction_norm: float


def c_pcgrad_result(
    current: torch.Tensor,
    reference: torch.Tensor,
    coefficient: float = 1.0,
    epsilon: float = 1.0e-8,
) -> PCGradResult:
    """Apply the faithful one-reference c-PCGrad rule.

    Only a negative inner product is removed.  Positive/aligned gradients and
    unavailable (near-zero) references are returned bit-for-bit unchanged.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(torch.tensor(float(coefficient))):
        raise ValueError("coefficient must be finite")
    dot_t = frobenius_inner(current, reference)
    ref = reference.to(device=current.device, dtype=dot_t.dtype)
    norm_sq = torch.sum(ref * ref)
    dot = float(dot_t.detach().cpu())
    if not torch.isfinite(dot_t) or not torch.isfinite(norm_sq):
        raise ValueError("non-finite gradient supplied to c-PCGrad")
    if dot >= 0.0 or float(norm_sq.detach().cpu()) <= epsilon:
        return PCGradResult(current, False, dot, 0.0)
    correction = float(coefficient) * dot_t / (norm_sq + float(epsilon)) * ref
    projected = current.to(dtype=dot_t.dtype) - correction
    projected = projected.to(dtype=current.dtype)
    return PCGradResult(
        projected,
        True,
        dot,
        float(torch.linalg.vector_norm(correction).detach().cpu()),
    )


def c_pcgrad(
    current: torch.Tensor,
    reference: torch.Tensor,
    coefficient: float = 1.0,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    return c_pcgrad_result(current, reference, coefficient, epsilon).gradient


@dataclass(frozen=True)
class ProjectionResult:
    gradient: torch.Tensor
    feasible: bool
    active_constraints: Tuple[str, ...]
    correction_norm: float
    constraint_values: Dict[str, float] = field(default_factory=dict)
    multipliers: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


def _projection_tolerance(
    gradient: torch.Tensor,
    references: Sequence[torch.Tensor],
    epsilon: float,
) -> float:
    scale = max(1.0, float(torch.linalg.vector_norm(gradient).detach().cpu()))
    for ref in references:
        scale = max(scale, float(torch.linalg.vector_norm(ref).detach().cpu()))
    return max(float(epsilon) * scale * scale * 16.0, 1.0e-12)


def dual_constraint_projection(
    current: torch.Tensor,
    history: Optional[torch.Tensor] = None,
    anchor: Optional[torch.Tensor] = None,
    history_slack: float = 0.0,
    anchor_slack: float = 0.0,
    epsilon: float = 1.0e-8,
) -> ProjectionResult:
    """Minimum-change projection onto history and anchor half-spaces.

    The two-constraint active set is enumerated explicitly.  Small Gram systems
    are solved in float64 and every candidate is checked against the original,
    unregularized inequalities before it can be selected.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if history_slack < 0 or anchor_slack < 0:
        raise ValueError("constraint slacks must be non-negative")
    refs_in = (("history", history, float(history_slack)), ("anchor", anchor, float(anchor_slack)))
    refs = []
    for name, ref, slack in refs_in:
        if ref is None:
            continue
        if ref.shape != current.shape:
            raise ValueError(f"{name} gradient shape mismatch")
        ref64 = ref.detach().to(device=current.device, dtype=torch.float64).reshape(-1)
        if not torch.isfinite(ref64).all():
            return ProjectionResult(current, False, (), 0.0, reason=f"non_finite_{name}_gradient")
        if float(torch.dot(ref64, ref64).detach().cpu()) <= epsilon:
            # 0 >= -slack is always feasible for a legal non-negative slack.
            continue
        refs.append((name, ref64, -slack))

    g = current.detach().to(dtype=torch.float64).reshape(-1)
    if not torch.isfinite(g).all():
        return ProjectionResult(current, False, (), 0.0, reason="non_finite_current_gradient")
    if not refs:
        return ProjectionResult(current, True, (), 0.0, {}, {}, "no_active_references")

    tolerance = _projection_tolerance(g, [item[1] for item in refs], epsilon)

    def constraint_values(candidate: torch.Tensor) -> Dict[str, float]:
        return {name: float(torch.dot(candidate, ref).detach().cpu()) for name, ref, _ in refs}

    def is_feasible(candidate: torch.Tensor) -> bool:
        return all(float(torch.dot(candidate, ref).detach().cpu()) >= bound - tolerance for _, ref, bound in refs)

    candidates = []
    # No active constraint.
    if is_feasible(g):
        candidates.append((g, tuple(), {}))

    # One or both active constraints.  There can be at most two.
    active_sets: Iterable[Tuple[int, ...]] = [(i,) for i in range(len(refs))]
    if len(refs) == 2:
        active_sets = [*active_sets, (0, 1)]
    for active in active_sets:
        rows = torch.stack([refs[i][1] for i in active], dim=0)
        bounds = torch.tensor([refs[i][2] for i in active], device=g.device, dtype=torch.float64)
        rhs = bounds - rows @ g
        gram = rows @ rows.T
        try:
            multipliers = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            multipliers = torch.linalg.pinv(gram, rtol=max(epsilon, 1.0e-12)) @ rhs
        equality_residual = torch.linalg.vector_norm(gram @ multipliers - rhs)
        if not torch.isfinite(multipliers).all() or float(equality_residual.detach().cpu()) > tolerance:
            continue
        if torch.any(multipliers < -tolerance):
            continue
        candidate = g + rows.T @ multipliers
        if not torch.isfinite(candidate).all() or not is_feasible(candidate):
            continue
        names = tuple(refs[i][0] for i in active)
        multiplier_map = {refs[i][0]: max(0.0, float(multipliers[j].detach().cpu())) for j, i in enumerate(active)}
        candidates.append((candidate, names, multiplier_map))

    if not candidates:
        return ProjectionResult(
            current,
            False,
            (),
            0.0,
            constraint_values(g),
            {},
            "active_set_no_feasible_solution",
        )
    candidate, active_names, multiplier_map = min(
        candidates,
        key=lambda item: float(torch.sum((item[0] - g) ** 2).detach().cpu()),
    )
    corrected = candidate.reshape_as(current).to(dtype=current.dtype)
    correction_norm = float(torch.linalg.vector_norm(candidate - g).detach().cpu())
    return ProjectionResult(
        corrected,
        True,
        active_names,
        correction_norm,
        constraint_values(candidate),
        multiplier_map,
    )


dual_constraint_project = dual_constraint_projection


def global_weighted_cosine(
    current_by_layer: Mapping[str, torch.Tensor],
    reference_by_layer: Mapping[str, torch.Tensor],
    weighting: str = "gradient_norm",
    epsilon: float = 1.0e-8,
) -> CosineResult:
    """Aggregate layer cosines without concatenating incompatible matrices."""

    common = sorted(set(current_by_layer) & set(reference_by_layer))
    values = []
    weights = []
    cur_norm_sq = 0.0
    ref_norm_sq = 0.0
    for layer_id in common:
        cur = current_by_layer[layer_id]
        ref = reference_by_layer[layer_id]
        result = gradient_cosine(cur, ref, epsilon)
        if not result.available:
            continue
        if weighting == "gradient_norm":
            weight = result.current_norm
        elif weighting == "parameter_count":
            weight = float(cur.numel())
        elif weighting == "uniform":
            weight = 1.0
        else:
            raise ValueError(f"unknown global cosine weighting: {weighting}")
        values.append(float(result.value))
        weights.append(weight)
        cur_norm_sq += result.current_norm ** 2
        ref_norm_sq += result.reference_norm ** 2
    if not values or sum(weights) <= epsilon:
        return CosineResult(None, False, cur_norm_sq ** 0.5, ref_norm_sq ** 0.5, "no_common_nonzero_layers")
    value = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return CosineResult(value, True, cur_norm_sq ** 0.5, ref_norm_sq ** 0.5)


class GradientGeometryTracker:
    """Own independent history/anchor EMA state for every logical layer."""

    def __init__(self, beta: float, threshold: float, consecutive_required: int) -> None:
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.consecutive_required = int(consecutive_required)
        self._history: Dict[str, ConflictEMA] = {}
        self._anchor: Dict[str, ConflictEMA] = {}
        self._global_history = ConflictEMA(self.beta, self.threshold, self.consecutive_required)
        self._global_anchor = ConflictEMA(self.beta, self.threshold, self.consecutive_required)

    def _tracker(self, table: Dict[str, ConflictEMA], layer_id: str) -> ConflictEMA:
        if layer_id not in table:
            table[layer_id] = ConflictEMA(self.beta, self.threshold, self.consecutive_required)
        return table[layer_id]

    def update(
        self,
        layer_id: str,
        current: torch.Tensor,
        history: Optional[torch.Tensor],
        anchor: Optional[torch.Tensor],
        epsilon: float = 1.0e-8,
    ) -> Tuple[ConflictStatus, ConflictStatus]:
        hist_cos = None if history is None else cosine_or_none(current, history, epsilon)
        anchor_cos = None if anchor is None else cosine_or_none(current, anchor, epsilon)
        return (
            self._tracker(self._history, layer_id).update(hist_cos),
            self._tracker(self._anchor, layer_id).update(anchor_cos),
        )

    def reset_episode(self) -> None:
        self._history.clear()
        self._anchor.clear()
        self._global_history.reset()
        self._global_anchor.reset()

    def update_global(
        self,
        history_cosine: Optional[float],
        anchor_cosine: Optional[float],
    ) -> Tuple[ConflictStatus, ConflictStatus]:
        """Update the two global trigger EMAs without changing layer geometry."""

        return (
            self._global_history.update(history_cosine),
            self._global_anchor.update(anchor_cosine),
        )


compute_gradient_cosine = gradient_cosine


__all__ = [
    "ConflictEMA",
    "ConflictStatus",
    "CosineResult",
    "GradientGeometryTracker",
    "PCGradResult",
    "ProjectionResult",
    "c_pcgrad",
    "c_pcgrad_result",
    "compute_gradient_cosine",
    "cosine_or_none",
    "dual_constraint_project",
    "dual_constraint_projection",
    "frobenius_inner",
    "global_weighted_cosine",
    "gradient_cosine",
]
