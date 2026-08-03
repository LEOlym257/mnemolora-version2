"""Pure, auditable FD-PSC commit gates with one-invocation enforcement."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class CommitGateError(RuntimeError):
    pass


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    reason: str
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitGateInputs:
    proposal_type: str
    before_commit_loss: Optional[float]
    fast_commit_loss: Optional[float]
    candidate_commit_loss: Optional[float]
    historical_replay_exists: bool = False
    before_history_loss: Optional[float] = None
    candidate_history_loss: Optional[float] = None
    before_history_by_context: Mapping[str, float] = field(default_factory=dict)
    candidate_history_by_context: Mapping[str, float] = field(default_factory=dict)
    before_anchor_loss: Optional[float] = None
    candidate_anchor_loss: Optional[float] = None
    plasticity_before_gain: Optional[float] = None
    plasticity_candidate_gain: Optional[float] = None
    functional_error_by_layer: Mapping[str, float] = field(default_factory=dict)
    drift_before_by_layer: Mapping[str, float] = field(default_factory=dict)
    drift_candidate_by_layer: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitGateReport:
    episode_id: str
    proposal_id: str
    query_token_id: str
    results: Tuple[GateResult, ...]
    commit_query_invocations: int = 1

    @property
    def passed(self) -> bool:
        return all(result.status != GateStatus.FAIL for result in self.results)

    def by_name(self, name: str) -> GateResult:
        for result in self.results:
            if result.gate == name:
                return result
        raise KeyError(name)


def _finite(*values: Optional[float]) -> bool:
    return all(value is not None and math.isfinite(float(value)) for value in values)


class CommitGateEvaluator:
    """Evaluate Gates 1-6 once for one proposal-bound query token."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        gates_config: Any,
        *,
        functional_error_threshold: float,
        minimum_exception_commit_fast_gain: float = 0.0,
    ) -> None:
        self.config = gates_config
        self.functional_error_threshold = float(functional_error_threshold)
        self.minimum_exception_commit_fast_gain = float(minimum_exception_commit_fast_gain)
        if self.functional_error_threshold < 0 or not math.isfinite(self.functional_error_threshold):
            raise ValueError("functional_error_threshold must be finite and non-negative")
        self._invocations: Dict[str, Dict[str, str]] = {}

    @property
    def abs_tol(self) -> float:
        return float(self.config.absolute_numerical_tolerance)

    @property
    def rel_tol(self) -> float:
        return float(self.config.relative_numerical_tolerance)

    def _allowed(self, observed: float, limit: float) -> bool:
        return observed <= limit or math.isclose(observed, limit, abs_tol=self.abs_tol, rel_tol=self.rel_tol)

    @staticmethod
    def _disabled(name: str) -> GateResult:
        return GateResult(name, GateStatus.NOT_APPLICABLE, "disabled_unsafe_ablation", {})

    @staticmethod
    def _missing(name: str, reason: str) -> GateResult:
        return GateResult(name, GateStatus.FAIL, reason, {})

    def evaluate_once(
        self,
        *,
        episode_id: str,
        proposal_id: str,
        query_token_id: str,
        inputs: CommitGateInputs,
    ) -> CommitGateReport:
        episode = str(episode_id)
        proposal = str(proposal_id)
        token = str(query_token_id)
        if not episode or not proposal or not token:
            raise CommitGateError("episode, proposal, and consumed query token IDs are required")
        if inputs.proposal_type not in ("global_slow", "new_exception", "replace_exception"):
            raise CommitGateError(f"unknown proposal type: {inputs.proposal_type!r}")
        if episode in self._invocations:
            raise CommitGateError("final commit gates may be invoked only once per episode")
        # Record before computation. A non-finite/missing metric or evaluator
        # exception is terminal and must not permit a second proposal/query.
        self._invocations[episode] = {"proposal_id": proposal, "query_token_id": token}
        results = (
            self._gate_current_gain(inputs),
            self._gate_history(inputs),
            self._gate_anchor(inputs),
            self._gate_plasticity(inputs),
            self._gate_functional_error(inputs),
            self._gate_drift(inputs),
        )
        return CommitGateReport(episode, proposal, token, results)

    def _gate_current_gain(self, inputs: CommitGateInputs) -> GateResult:
        name = "current_gain"
        if not self.config.current_gain_enabled:
            return self._disabled(name)
        if not _finite(inputs.before_commit_loss, inputs.fast_commit_loss, inputs.candidate_commit_loss):
            return self._missing(name, "missing_or_nonfinite_commit_query_loss")
        before = float(inputs.before_commit_loss)  # type: ignore[arg-type]
        fast = float(inputs.fast_commit_loss)  # type: ignore[arg-type]
        candidate = float(inputs.candidate_commit_loss)  # type: ignore[arg-type]
        gain_fast = before - fast
        gain_candidate = before - candidate
        metrics = {"gain_fast_commit": gain_fast, "gain_candidate_commit": gain_candidate}
        if gain_fast <= self.abs_tol:
            return GateResult(name, GateStatus.FAIL, "fast_gain_not_positive", metrics)
        if inputs.proposal_type in ("new_exception", "replace_exception"):
            threshold = max(self.minimum_exception_commit_fast_gain, self.abs_tol)
            if gain_fast < threshold and not math.isclose(
                gain_fast, threshold, abs_tol=self.abs_tol, rel_tol=self.rel_tol
            ):
                return GateResult(name, GateStatus.FAIL, "exception_fast_gain_below_threshold", metrics)
        required = float(self.config.fast_gain_retention) * gain_fast
        if gain_candidate < required and not math.isclose(
            gain_candidate, required, abs_tol=self.abs_tol, rel_tol=self.rel_tol
        ):
            return GateResult(name, GateStatus.FAIL, "candidate_did_not_retain_fast_gain", metrics)
        return GateResult(name, GateStatus.PASS, "gain_retained", metrics)

    def _gate_history(self, inputs: CommitGateInputs) -> GateResult:
        name = "history"
        if not self.config.history_enabled:
            return self._disabled(name)
        if not inputs.historical_replay_exists:
            return GateResult(name, GateStatus.NOT_APPLICABLE, "cold_start_empty_historical_replay", {})
        if not _finite(inputs.before_history_loss, inputs.candidate_history_loss):
            return self._missing(name, "historical_replay_exists_but_metric_unavailable")
        before = float(inputs.before_history_loss)  # type: ignore[arg-type]
        candidate = float(inputs.candidate_history_loss)  # type: ignore[arg-type]
        regression = candidate - before
        metrics: Dict[str, float] = {"history_regression": regression}
        if not self._allowed(regression, float(self.config.history_loss_tolerance)):
            return GateResult(name, GateStatus.FAIL, "mean_history_regression", metrics)
        before_ctx = dict(inputs.before_history_by_context)
        candidate_ctx = dict(inputs.candidate_history_by_context)
        if not before_ctx or set(before_ctx) != set(candidate_ctx):
            return GateResult(name, GateStatus.FAIL, "missing_or_mismatched_context_metrics", metrics)
        regressions = []
        for context in sorted(before_ctx):
            if not _finite(before_ctx[context], candidate_ctx[context]):
                return GateResult(name, GateStatus.FAIL, "nonfinite_context_metric", metrics)
            regressions.append(float(candidate_ctx[context]) - float(before_ctx[context]))
        worst = max(regressions)
        metrics["worst_context_regression"] = worst
        if not self._allowed(worst, float(self.config.worst_context_loss_tolerance)):
            return GateResult(name, GateStatus.FAIL, "worst_context_regression", metrics)
        return GateResult(name, GateStatus.PASS, "history_preserved", metrics)

    def _gate_anchor(self, inputs: CommitGateInputs) -> GateResult:
        name = "anchor"
        if not self.config.anchor_enabled:
            return self._disabled(name)
        if not _finite(inputs.before_anchor_loss, inputs.candidate_anchor_loss):
            return self._missing(name, "missing_or_nonfinite_anchor_metric")
        regression = float(inputs.candidate_anchor_loss) - float(inputs.before_anchor_loss)  # type: ignore[arg-type]
        metrics = {"anchor_regression": regression}
        if not self._allowed(regression, float(self.config.anchor_loss_tolerance)):
            return GateResult(name, GateStatus.FAIL, "anchor_regression", metrics)
        return GateResult(name, GateStatus.PASS, "anchor_preserved", metrics)

    def _gate_plasticity(self, inputs: CommitGateInputs) -> GateResult:
        name = "plasticity"
        if not self.config.plasticity_enabled:
            return self._disabled(name)
        if not _finite(inputs.plasticity_before_gain, inputs.plasticity_candidate_gain):
            return self._missing(name, "missing_or_nonfinite_plasticity_gain")
        before = float(inputs.plasticity_before_gain)  # type: ignore[arg-type]
        candidate = float(inputs.plasticity_candidate_gain)  # type: ignore[arg-type]
        metrics = {"plasticity_before_gain": before, "plasticity_candidate_gain": candidate}
        if before > self.abs_tol:
            required = float(self.config.plasticity_retention) * before
            metrics["plasticity_required_gain"] = required
            if candidate + self.abs_tol < required and not math.isclose(
                candidate, required, abs_tol=self.abs_tol, rel_tol=self.rel_tol
            ):
                return GateResult(name, GateStatus.FAIL, "plasticity_ratio_failed", metrics)
        else:
            if candidate < -self.abs_tol:
                return GateResult(name, GateStatus.FAIL, "near_zero_before_negative_candidate_gain", metrics)
            if before - candidate > self.abs_tol and not math.isclose(
                before, candidate, abs_tol=self.abs_tol, rel_tol=self.rel_tol
            ):
                return GateResult(name, GateStatus.FAIL, "near_zero_absolute_fallback_failed", metrics)
        return GateResult(name, GateStatus.PASS, "plasticity_preserved", metrics)

    def _gate_functional_error(self, inputs: CommitGateInputs) -> GateResult:
        name = "functional_error"
        if not self.config.functional_error_enabled:
            return self._disabled(name)
        errors = dict(inputs.functional_error_by_layer)
        if not errors:
            return self._missing(name, "functional_error_metric_missing")
        if any(not math.isfinite(float(value)) for value in errors.values()):
            return self._missing(name, "nonfinite_functional_error")
        worst = max(float(value) for value in errors.values())
        metrics = {"maximum_functional_error": worst}
        if not self._allowed(worst, self.functional_error_threshold):
            return GateResult(name, GateStatus.FAIL, "functional_error_threshold", metrics)
        return GateResult(name, GateStatus.PASS, "compression_function_preserved", metrics)

    def _gate_drift(self, inputs: CommitGateInputs) -> GateResult:
        name = "spectral_drift"
        if not self.config.spectral_drift_enabled:
            return self._disabled(name)
        before = dict(inputs.drift_before_by_layer)
        candidate = dict(inputs.drift_candidate_by_layer)
        if not before or set(before) != set(candidate):
            return self._missing(name, "missing_or_mismatched_drift_metrics")
        regressions = []
        for layer in sorted(before):
            if not _finite(before[layer], candidate[layer]):
                return self._missing(name, "nonfinite_drift_metric")
            regressions.append(float(candidate[layer]) - float(before[layer]))
        worst = max(regressions)
        metrics = {"maximum_drift_increase": worst}
        if not self._allowed(worst, float(self.config.drift_tolerance)):
            return GateResult(name, GateStatus.FAIL, "spectral_drift_increase", metrics)
        return GateResult(name, GateStatus.PASS, "spectral_drift_bounded", metrics)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "invocations": copy.deepcopy(self._invocations),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise CommitGateError("unsupported commit-gate state schema")
        self._invocations = copy.deepcopy(dict(state.get("invocations", {})))


__all__ = [
    "CommitGateError",
    "CommitGateEvaluator",
    "CommitGateInputs",
    "CommitGateReport",
    "GateResult",
    "GateStatus",
]
