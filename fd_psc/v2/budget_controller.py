"""Closed-loop FSD V2 trust-region budget controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..config import AdaptiveBudgetConfig, RTRCConfig


class AdaptiveBudgetError(RuntimeError):
    """Raised when controller state or operational losses are invalid."""


@dataclass(frozen=True)
class AdaptiveBudgetState:
    u: float
    beta: float
    update_count: int


@dataclass(frozen=True)
class AdaptiveBudgetUpdate:
    beta_before: float
    beta_after: float
    wake_gain: float
    current_loss_cost: float
    operational_plasticity_loss_ratio: float
    historical_replay_regression: float
    history_signal_available: bool
    plasticity_signal_available: bool
    controller_error: float
    update_applied: bool


class AdaptiveBudgetController:
    """Map an unconstrained state through a sigmoid into ``[beta_min,beta_max]``."""

    SCHEMA_VERSION = 1
    _U_LIMIT = 30.0

    def __init__(
        self,
        config: AdaptiveBudgetConfig,
        rtrc: RTRCConfig,
    ) -> None:
        self.config = config
        self.minimum = float(rtrc.budget_fraction_minimum)
        self.maximum = float(rtrc.budget_fraction_maximum)
        initial = float(rtrc.budget_fraction_initial)
        if not (0.0 < self.minimum <= initial <= self.maximum <= 1.0):
            raise AdaptiveBudgetError("adaptive budget bounds are invalid")
        if self.maximum == self.minimum:
            self.u = 0.0
        else:
            fraction = (initial - self.minimum) / (self.maximum - self.minimum)
            fraction = min(max(fraction, 1.0e-12), 1.0 - 1.0e-12)
            self.u = min(
                max(math.log(fraction / (1.0 - fraction)), -self._U_LIMIT),
                self._U_LIMIT,
            )
        self.update_count = 0

    @property
    def beta(self) -> float:
        if self.maximum == self.minimum:
            return self.minimum
        sigmoid = 1.0 / (1.0 + math.exp(-self.u))
        return self.minimum + (self.maximum - self.minimum) * sigmoid

    @property
    def state(self) -> AdaptiveBudgetState:
        return AdaptiveBudgetState(self.u, self.beta, self.update_count)

    @staticmethod
    def _loss(name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            raise AdaptiveBudgetError(f"{name} must be finite when available")
        return result

    def update(
        self,
        *,
        current_before: float,
        current_fast: float,
        current_rtrc: float,
        history_before: Optional[float],
        history_rtrc: Optional[float],
        epsilon: float,
    ) -> AdaptiveBudgetUpdate:
        before = self._loss("current_before", current_before)
        fast = self._loss("current_fast", current_fast)
        accepted = self._loss("current_rtrc", current_rtrc)
        assert before is not None and fast is not None and accepted is not None
        hist_before = self._loss("history_before", history_before)
        hist_rtrc = self._loss("history_rtrc", history_rtrc)
        if (hist_before is None) != (hist_rtrc is None):
            raise AdaptiveBudgetError("history losses must be jointly available")
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise AdaptiveBudgetError("controller epsilon must be finite and positive")

        beta_before = self.beta
        wake_gain = before - fast
        current_cost = accepted - fast
        plasticity_available = wake_gain > float(self.config.minimum_wake_gain)
        plasticity_ratio = (
            max(current_cost, 0.0) / (max(wake_gain, 0.0) + epsilon_value)
            if plasticity_available
            else 0.0
        )
        history_available = hist_before is not None
        history_regression = (
            max(float(hist_rtrc) - float(hist_before), 0.0)
            / (abs(float(hist_before)) + epsilon_value)
            if history_available
            else 0.0
        )
        clip = float(self.config.error_clip)
        plasticity_error = (
            min(
                max(
                    plasticity_ratio - float(self.config.plasticity_loss_target),
                    -clip,
                ),
                clip,
            )
            if plasticity_available
            else 0.0
        )
        history_error = (
            min(
                max(
                    history_regression
                    - float(self.config.history_regression_target),
                    -clip,
                ),
                clip,
            )
            if history_available
            else 0.0
        )
        error = plasticity_error - float(self.config.history_weight) * history_error
        applied = bool(plasticity_available or history_available)
        if applied:
            self.u = min(
                max(
                    self.u
                    + float(self.config.controller_learning_rate) * error,
                    -self._U_LIMIT,
                ),
                self._U_LIMIT,
            )
            self.update_count += 1
        return AdaptiveBudgetUpdate(
            beta_before=beta_before,
            beta_after=self.beta,
            wake_gain=wake_gain,
            current_loss_cost=current_cost,
            operational_plasticity_loss_ratio=plasticity_ratio,
            historical_replay_regression=history_regression,
            history_signal_available=history_available,
            plasticity_signal_available=plasticity_available,
            controller_error=error,
            update_applied=applied,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "u": self.u,
            "beta": self.beta,
            "update_count": self.update_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or int(state.get("schema_version", -1)) != 1:
            raise AdaptiveBudgetError("adaptive budget controller schema is invalid")
        minimum = float(state.get("minimum", float("nan")))
        maximum = float(state.get("maximum", float("nan")))
        if minimum != self.minimum or maximum != self.maximum:
            raise AdaptiveBudgetError("adaptive budget bounds mismatch")
        u = float(state.get("u", float("nan")))
        count = int(state.get("update_count", -1))
        if not math.isfinite(u) or not -self._U_LIMIT <= u <= self._U_LIMIT or count < 0:
            raise AdaptiveBudgetError("adaptive budget state is invalid")
        old_u = self.u
        self.u = u
        expected_beta = self.beta
        stored_beta = float(state.get("beta", float("nan")))
        if not math.isclose(stored_beta, expected_beta, rel_tol=1.0e-12, abs_tol=1.0e-12):
            self.u = old_u
            raise AdaptiveBudgetError("adaptive budget beta/u state is inconsistent")
        self.update_count = count


__all__ = [
    "AdaptiveBudgetController",
    "AdaptiveBudgetError",
    "AdaptiveBudgetState",
    "AdaptiveBudgetUpdate",
]
