"""Generic cumulative repair engine, intentionally blind to commit-query data."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .replay_memory import GRASPBatch, GRASPSampler, ReplayWindow


class RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairBatch:
    current: Tuple[Any, ...]
    replay: Tuple[Any, ...]
    replay_phase: str
    duplicate_rate: float
    normalized_weights: Mapping[str, float]


@dataclass(frozen=True)
class ScreeningResult:
    passed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class RepairCheckpoint:
    cumulative_step: int
    screening: ScreeningResult
    candidate_state: Any
    step_metrics: Tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class RepairResult:
    succeeded: bool
    final_state: Any
    selected_step: Optional[int]
    checkpoints: Tuple[RepairCheckpoint, ...]
    reason: str


class RepairEngine:
    """Run one optimizer trajectory and screen clones at cumulative checkpoints.

    ``train_step`` owns model-specific JEPA, effective-gradient projection, and
    LPR calculations.  The engine owns the hard protocol: exact cumulative step
    count, GRASP phase order, weight renormalization for empty replay, clone-only
    screening, and early stop at the first feasible checkpoint.  There is no
    commit-query argument in this API by design.
    """

    def __init__(
        self,
        *,
        maximum_steps: int,
        candidate_steps: Sequence[int],
        windows_per_batch: int,
        current_weight: float,
        replay_weight: float,
        proximal_enabled: bool,
        proximal_weight: float,
        pcgrad_enabled: bool,
        seed: int,
        sampling: str = "grasp",
    ) -> None:
        self.maximum_steps = int(maximum_steps)
        self.candidate_steps = tuple(int(step) for step in candidate_steps)
        self.windows_per_batch = int(windows_per_batch)
        self.current_weight = float(current_weight)
        self.replay_weight = float(replay_weight)
        self.proximal_enabled = bool(proximal_enabled)
        self.proximal_weight = float(proximal_weight)
        self.pcgrad_enabled = bool(pcgrad_enabled)
        self.sampling = str(sampling)
        if self.sampling not in {"grasp", "balanced_uniform"}:
            raise ValueError("repair sampling must be grasp or balanced_uniform")
        if self.maximum_steps < 0 or self.windows_per_batch <= 0:
            raise ValueError("repair step capacity must be non-negative and batch size positive")
        if (
            not self.candidate_steps
            or self.candidate_steps != tuple(sorted(set(self.candidate_steps)))
            or any(step <= 0 or step > self.maximum_steps for step in self.candidate_steps)
        ):
            raise ValueError("candidate_steps must be positive, unique, cumulative, and <= maximum_steps")
        for name, value in (
            ("current_weight", self.current_weight),
            ("replay_weight", self.replay_weight),
            ("proximal_weight", self.proximal_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.grasp = GRASPSampler(seed)
        self._uniform_rng = random.Random(int(seed) ^ 0x5EEDBEEF)

    def _sample_balanced_uniform(
        self,
        windows: Sequence[ReplayWindow],
        *,
        step_index: int,
    ) -> GRASPBatch:
        if not windows:
            return GRASPBatch("balanced_uniform", (), 0.0)
        buckets: Dict[str, List[ReplayWindow]] = {}
        for window in sorted(windows, key=lambda item: (item.context_identifier, item.window_id)):
            buckets.setdefault(window.context_identifier, []).append(window)
        contexts = sorted(buckets)
        selections: List[ReplayWindow] = []
        available = {key: list(value) for key, value in buckets.items()}
        for offset in range(self.windows_per_batch):
            context = contexts[(int(step_index) + offset) % len(contexts)]
            pool = available[context]
            if pool:
                index = self._uniform_rng.randrange(len(pool))
                selections.append(pool.pop(index).clone())
            else:
                source = buckets[context]
                selections.append(source[self._uniform_rng.randrange(len(source))].clone())
        unique = len({item.window_id for item in selections})
        duplicate_rate = 1.0 - unique / float(len(selections))
        return GRASPBatch("balanced_uniform", tuple(selections), duplicate_rate)

    def _weights(self, replay_available: bool) -> Dict[str, float]:
        enabled = {"current": self.current_weight}
        if replay_available:
            enabled["replay"] = self.replay_weight
        if self.proximal_enabled and replay_available:
            enabled["proximal"] = self.proximal_weight
        total = sum(value for value in enabled.values() if value > 0)
        if total <= 0:
            raise RepairError("all applicable repair loss weights are zero")
        return {name: value / total for name, value in enabled.items() if value > 0}

    def _current_batch(self, windows: Sequence[Any], step_index: int) -> Tuple[Any, ...]:
        if not windows:
            raise RepairError("repair requires current episode support")

        def stable_id(item: Any) -> str:
            window_id = getattr(item, "window_id", None)
            if window_id is not None:
                return str(window_id)
            identity = getattr(item, "identity", None)
            record_id = getattr(identity, "record_id", None)
            if record_id is None:
                record_id = getattr(item, "record_id", None)
            if record_id is None:
                raise RepairError("current repair window lacks a stable record/window ID")
            return str(record_id)

        ordered = sorted(windows, key=stable_id)
        return tuple(
            copy.deepcopy(ordered[(step_index * self.windows_per_batch + offset) % len(ordered)])
            for offset in range(self.windows_per_batch)
        )

    def run(
        self,
        initial_state: Any,
        *,
        current_windows: Sequence[Any],
        replay_windows: Sequence[ReplayWindow],
        train_step: Callable[[Any, RepairBatch, int, bool], Optional[Mapping[str, float]]],
        screen_candidate: Callable[[Any, int], ScreeningResult],
    ) -> RepairResult:
        if not current_windows:
            return RepairResult(False, copy.deepcopy(initial_state), None, (), "empty_current_support")
        live_state = copy.deepcopy(initial_state)
        replay_available = bool(replay_windows)
        weights = self._weights(replay_available)
        checkpoints: List[RepairCheckpoint] = []
        step_metrics: List[Mapping[str, float]] = []
        last_checkpoint = self.candidate_steps[-1]
        for step_index in range(last_checkpoint):
            current = self._current_batch(current_windows, step_index)
            if replay_available and self.sampling == "grasp":
                grasp_batch = self.grasp.sample(
                    replay_windows,
                    step_index=step_index,
                    total_steps=last_checkpoint,
                    batch_size=self.windows_per_batch,
                )
            elif replay_available:
                grasp_batch = self._sample_balanced_uniform(
                    replay_windows,
                    step_index=step_index,
                )
            else:
                grasp_batch = GRASPBatch(
                    phase=(
                        GRASPSampler.phase_for_step(step_index, last_checkpoint)
                        if self.sampling == "grasp"
                        else "balanced_uniform"
                    ),
                    windows=(),
                    duplicate_rate=0.0,
                )
            batch = RepairBatch(
                current=current,
                replay=tuple(grasp_batch.windows),
                replay_phase=grasp_batch.phase,
                duplicate_rate=grasp_batch.duplicate_rate,
                normalized_weights=weights,
            )
            metrics = train_step(live_state, batch, step_index + 1, self.pcgrad_enabled)
            normalized_metrics: Dict[str, float] = {}
            for name, value in dict(metrics or {}).items():
                number = float(value)
                if not math.isfinite(number):
                    raise RepairError(f"repair step returned non-finite metric {name}")
                normalized_metrics[str(name)] = number
            normalized_metrics["grasp_duplicate_rate"] = batch.duplicate_rate
            step_metrics.append(normalized_metrics)
            cumulative_step = step_index + 1
            if cumulative_step not in self.candidate_steps:
                continue
            # Screening owns a detached clone and may recompress that clone.
            # The live optimizer trajectory (including its moments) therefore
            # continues untouched when a checkpoint is rejected.
            candidate_clone = copy.deepcopy(live_state)
            screening = screen_candidate(candidate_clone, cumulative_step)
            if not isinstance(screening, ScreeningResult):
                raise RepairError("screen_candidate must return ScreeningResult")
            checkpoint = RepairCheckpoint(
                cumulative_step=cumulative_step,
                screening=screening,
                candidate_state=copy.deepcopy(candidate_clone),
                step_metrics=tuple(copy.deepcopy(step_metrics)),
            )
            checkpoints.append(checkpoint)
            if screening.passed:
                return RepairResult(
                    True,
                    copy.deepcopy(candidate_clone),
                    cumulative_step,
                    tuple(checkpoints),
                    "first_feasible_cumulative_checkpoint",
                )
        return RepairResult(
            False,
            copy.deepcopy(live_state),
            None,
            tuple(checkpoints),
            "all_cumulative_checkpoints_failed_screening",
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "sampling": self.sampling,
            "grasp": self.grasp.state_dict(),
            "uniform_rng_state": copy.deepcopy(self._uniform_rng.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("sampling", self.sampling)) != self.sampling:
            raise RepairError("repair sampling mode mismatch")
        self.grasp.load_state_dict(state["grasp"])
        if "uniform_rng_state" in state:
            self._uniform_rng.setstate(copy.deepcopy(state["uniform_rng_state"]))


__all__ = [
    "RepairBatch",
    "RepairCheckpoint",
    "RepairEngine",
    "RepairError",
    "RepairResult",
    "ScreeningResult",
]
