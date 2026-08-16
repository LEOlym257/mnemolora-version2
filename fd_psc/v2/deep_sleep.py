"""Capacity-triggered residual distillation and slow-rank recycling.

Deep Sleep is deliberately downstream of RTRC and ordinary slow compression.
It receives only the uncompressed slow candidate plus internal raw replay and
current support.  The teacher is represented by immutable cached residual
targets, so no optimizer step can move the teacher during consolidation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from ..config import DeepSleepConfig
from ..lora_layers import LogicalLoRAAdapter
from ..low_rank_merge import LowRankFactors, as_factors


class DeepSleepError(RuntimeError):
    """Raised when a Deep Sleep invariant is violated."""


@dataclass(frozen=True)
class ModelTrace:
    """One functional output plus a deliberately small hidden-layer trace."""

    output: Tensor
    hidden: Mapping[str, Tensor]


@dataclass(frozen=True)
class ResidualTarget:
    """Fixed teacher-minus-core targets for one internal example."""

    source: str
    payload: Any
    core_output: Tensor
    teacher_output_residual: Tensor
    core_hidden: Mapping[str, Tensor]
    teacher_hidden_residual: Mapping[str, Tensor]


@dataclass(frozen=True)
class CoreDistillationResult:
    steps: int
    output_residual_loss: float
    hidden_residual_loss: float
    current_jepa_loss: float
    total_loss: float
    core_write_frobenius: float
    teacher_frozen: bool


@dataclass(frozen=True)
class ResidualRefitResult:
    factors: Mapping[str, LowRankFactors]
    numerical_rank_before: Mapping[str, int]
    residual_rank_after: Mapping[str, int]


@dataclass(frozen=True)
class DeepSleepSplit:
    """Checkpoint-reproducible historical fit/validation partition."""

    fit_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]


@dataclass(frozen=True)
class FunctionalComparison:
    relative_error: float
    absolute_error: float
    reference_norm: float
    finite: bool

    def passes(self, relative_threshold: float, absolute_tolerance: float) -> bool:
        if not self.finite:
            return False
        if self.reference_norm <= absolute_tolerance:
            return self.absolute_error <= absolute_tolerance
        return self.relative_error <= relative_threshold


class DeepSleepController:
    """Consecutive overflow trigger with a checkpointed sampler RNG."""

    SCHEMA_VERSION = 1

    def __init__(self, config: DeepSleepConfig, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.consecutive_overflows = 0
        self.observation_count = 0
        self.trigger_count = 0
        self.success_count = 0
        self.last_relative_rank_error = 0.0
        self.last_available_windows = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def should_trigger(
        self,
        relative_rank_error: float,
        *,
        available_windows: int,
    ) -> bool:
        error = float(relative_rank_error)
        windows = int(available_windows)
        if not math.isfinite(error) or error < 0.0:
            raise DeepSleepError("relative rank error must be finite and non-negative")
        if windows < 0:
            raise DeepSleepError("available window count must be non-negative")
        self.observation_count += 1
        self.last_relative_rank_error = error
        self.last_available_windows = windows
        if error > float(self.config.trigger_relative_rank_error):
            self.consecutive_overflows += 1
        else:
            self.consecutive_overflows = 0
        triggered = bool(
            self.config.enabled
            and self.consecutive_overflows
            >= int(self.config.trigger_consecutive_commits)
            and windows
            >= max(
                int(self.config.minimum_replay_windows),
                int(self.config.minimum_validation_windows),
            )
        )
        if triggered:
            self.trigger_count += 1
        return triggered

    def mark_success(self) -> None:
        self.success_count += 1
        self.consecutive_overflows = 0

    def sample_indices(self, population: int, batch_size: int) -> Tuple[int, ...]:
        count = int(population)
        batch = int(batch_size)
        if count <= 0 or batch <= 0:
            raise DeepSleepError("deep-sleep sampler requires positive sizes")
        if batch >= count:
            return tuple(
                int(index)
                for index in torch.randperm(count, generator=self.generator).tolist()
            )
        return tuple(
            int(index)
            for index in torch.randperm(count, generator=self.generator)[:batch].tolist()
        )

    def split_fit_validation_indices(self, population: int) -> DeepSleepSplit:
        """Split historical replay with the checkpointed sampler generator.

        Current-episode support is deliberately kept on the fit side by the
        caller.  The validation indices therefore identify held-out historical
        windows that never enter core optimization or the current-task term.
        """

        count = int(population)
        minimum = int(self.config.minimum_validation_windows)
        if count < minimum:
            raise DeepSleepError(
                "deep sleep lacks the minimum held-out validation windows"
            )
        requested = max(
            minimum,
            int(math.ceil(float(self.config.validation_fraction) * count)),
        )
        validation_count = min(requested, count)
        permutation = tuple(
            int(index)
            for index in torch.randperm(count, generator=self.generator).tolist()
        )
        validation = tuple(sorted(permutation[:validation_count]))
        validation_set = set(validation)
        fit = tuple(index for index in range(count) if index not in validation_set)
        return DeepSleepSplit(fit, validation)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "seed": self.seed,
            "consecutive_overflows": self.consecutive_overflows,
            "observation_count": self.observation_count,
            "trigger_count": self.trigger_count,
            "success_count": self.success_count,
            "last_relative_rank_error": self.last_relative_rank_error,
            "last_available_windows": self.last_available_windows,
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or int(state.get("schema_version", -1)) != 1:
            raise DeepSleepError("deep-sleep controller schema is invalid")
        if int(state.get("seed", -1)) != self.seed:
            raise DeepSleepError("deep-sleep controller seed mismatch")
        counters = {}
        for name in (
            "consecutive_overflows",
            "observation_count",
            "trigger_count",
            "success_count",
            "last_available_windows",
        ):
            value = int(state.get(name, -1))
            if value < 0:
                raise DeepSleepError(f"deep-sleep controller {name} is invalid")
            counters[name] = value
        error = float(state.get("last_relative_rank_error", float("nan")))
        if not math.isfinite(error) or error < 0.0:
            raise DeepSleepError("deep-sleep controller rank error is invalid")
        generator_state = state.get("generator_state")
        if (
            not torch.is_tensor(generator_state)
            or generator_state.device.type != "cpu"
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
            or generator_state.numel() == 0
        ):
            raise DeepSleepError("deep-sleep controller RNG state is invalid")
        candidate = torch.Generator(device="cpu")
        candidate.set_state(generator_state.detach().cpu().clone())
        for name, value in counters.items():
            setattr(self, name, value)
        self.last_relative_rank_error = error
        self.generator = candidate


def partition_residual_targets(
    targets: Sequence[ResidualTarget],
    *,
    historical_count: int,
    split: DeepSleepSplit,
) -> Tuple[Tuple[ResidualTarget, ...], Tuple[ResidualTarget, ...]]:
    """Keep validation replay out of every optimizer-facing target batch."""

    count = int(historical_count)
    if count < 0 or count > len(targets):
        raise DeepSleepError("historical target count is invalid")
    expected = set(range(count))
    fit_set = set(split.fit_indices)
    validation_set = set(split.validation_indices)
    if fit_set & validation_set or fit_set | validation_set != expected:
        raise DeepSleepError("deep-sleep split is not a partition of history")
    if any(target.source != "raw_replay" for target in targets[:count]):
        raise DeepSleepError("historical targets must precede current support")
    if any(target.source != "current_support" for target in targets[count:]):
        raise DeepSleepError("current-support targets must follow history")
    current_indices = tuple(range(count, len(targets)))
    fit_indices = split.fit_indices + current_indices
    fit = tuple(targets[index] for index in fit_indices)
    validation = tuple(targets[index] for index in split.validation_indices)
    if not fit or not validation:
        raise DeepSleepError(
            "deep sleep requires non-empty fit and held-out validation subsets"
        )
    return fit, validation


def cache_residual_targets(
    *,
    sources_and_payloads: Sequence[Tuple[str, Any]],
    core_traces: Sequence[ModelTrace],
    teacher_traces: Sequence[ModelTrace],
) -> Tuple[ResidualTarget, ...]:
    """Freeze ``teacher - core_old`` output and hidden residuals."""

    if not (
        len(sources_and_payloads) == len(core_traces) == len(teacher_traces)
    ):
        raise DeepSleepError("residual target inputs have different lengths")
    result = []
    for (source, payload), core, teacher in zip(
        sources_and_payloads, core_traces, teacher_traces
    ):
        if source not in {"raw_replay", "current_support"}:
            raise DeepSleepError(f"illegal distillation source {source!r}")
        if core.output.shape != teacher.output.shape:
            raise DeepSleepError("teacher/core output shapes differ")
        if set(core.hidden) != set(teacher.hidden):
            raise DeepSleepError("teacher/core hidden-layer sets differ")
        hidden_core: Dict[str, Tensor] = {}
        hidden_residual: Dict[str, Tensor] = {}
        for name in sorted(core.hidden):
            if core.hidden[name].shape != teacher.hidden[name].shape:
                raise DeepSleepError(f"teacher/core hidden shape differs for {name}")
            hidden_core[name] = core.hidden[name].detach().clone()
            hidden_residual[name] = (
                teacher.hidden[name].detach() - core.hidden[name].detach()
            ).clone()
        result.append(
            ResidualTarget(
                source=source,
                payload=payload,
                core_output=core.output.detach().clone(),
                teacher_output_residual=(
                    teacher.output.detach() - core.output.detach()
                ).clone(),
                core_hidden=hidden_core,
                teacher_hidden_residual=hidden_residual,
            )
        )
    if not result or not any(item.source == "current_support" for item in result):
        raise DeepSleepError("deep sleep requires current episode support")
    return tuple(result)


def _residual_losses(
    targets: Sequence[ResidualTarget],
    traces: Sequence[ModelTrace],
    *,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    if len(targets) != len(traces) or not targets:
        raise DeepSleepError("student residual batch is empty or misaligned")
    output_terms = []
    hidden_differences: Dict[str, list[Tensor]] = {}
    hidden_targets: Dict[str, list[Tensor]] = {}
    for target, trace in zip(targets, traces):
        if trace.output.shape != target.core_output.shape:
            raise DeepSleepError("student/core output shapes differ")
        student_residual = trace.output - target.core_output
        output_terms.append(
            torch.mean((student_residual - target.teacher_output_residual).square())
        )
        if set(trace.hidden) != set(target.core_hidden):
            raise DeepSleepError("student hidden-layer set changed during deep sleep")
        for name in sorted(trace.hidden):
            student_hidden_residual = trace.hidden[name] - target.core_hidden[name]
            teacher_hidden_residual = target.teacher_hidden_residual[name]
            hidden_differences.setdefault(name, []).append(
                (student_hidden_residual - teacher_hidden_residual).reshape(-1)
            )
            hidden_targets.setdefault(name, []).append(
                teacher_hidden_residual.reshape(-1)
            )
    output_loss = torch.stack(output_terms).mean()
    hidden_terms = []
    for name in sorted(hidden_differences):
        difference = torch.cat(hidden_differences[name])
        target = torch.cat(hidden_targets[name])
        hidden_terms.append(
            torch.mean(difference.square())
            / (torch.mean(target.square()) + float(epsilon))
        )
    hidden_loss = (
        torch.stack(hidden_terms).sum()
        if hidden_terms
        else output_loss.new_zeros(())
    )
    return output_loss, hidden_loss


def distill_core_memory(
    *,
    adapters: Mapping[str, LogicalLoRAAdapter],
    targets: Sequence[ResidualTarget],
    trace_student: Callable[[Any], ModelTrace],
    current_jepa_loss: Callable[[Any], Tensor],
    controller: DeepSleepController,
    config: DeepSleepConfig,
) -> CoreDistillationResult:
    """Optimize dense ``core_delta`` against fixed residual targets."""

    if not targets:
        raise DeepSleepError("deep sleep has no distillation targets")
    core_before = {
        key: adapter.get_core_delta().detach().clone()
        for key, adapter in sorted(adapters.items())
    }
    parameters = []
    for _key, adapter in sorted(adapters.items()):
        core = adapter.get_core_delta()
        if not core.is_leaf:
            raise DeepSleepError("core_delta must remain a leaf buffer")
        core.requires_grad_(True)
        core.grad = None
        parameters.append(core)
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    current = tuple(item for item in targets if item.source == "current_support")
    final_values = (float("inf"), float("inf"), float("inf"), float("inf"))
    try:
        for _step in range(int(config.maximum_steps)):
            indices = controller.sample_indices(len(targets), int(config.batch_size))
            batch = tuple(targets[index] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            traces = tuple(trace_student(item.payload) for item in batch)
            output_loss, hidden_loss = _residual_losses(
                batch, traces, epsilon=float(config.epsilon)
            )
            current_terms = [current_jepa_loss(item.payload) for item in current]
            current_loss = (
                torch.stack(current_terms).mean()
                if current_terms
                else output_loss.new_zeros(())
            )
            total = (
                float(config.output_residual_weight) * output_loss
                + float(config.hidden_residual_weight) * hidden_loss
                + float(config.current_task_weight) * current_loss
            )
            if not torch.isfinite(total):
                raise DeepSleepError("deep-sleep loss became non-finite")
            total.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in parameters
            ):
                raise DeepSleepError("deep-sleep core gradient is missing or non-finite")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise DeepSleepError("deep-sleep core update became non-finite")
            final_values = tuple(
                float(value.detach().cpu())
                for value in (output_loss, hidden_loss, current_loss, total)
            )
    finally:
        for parameter in parameters:
            parameter.requires_grad_(False)
            parameter.grad = None

    write_sq = 0.0
    for key, adapter in sorted(adapters.items()):
        difference = (
            adapter.get_core_delta().detach().to(torch.float64)
            - core_before[key].to(torch.float64)
        )
        write_sq += float(torch.sum(difference.square()).cpu())
    return CoreDistillationResult(
        steps=int(config.maximum_steps),
        output_residual_loss=final_values[0],
        hidden_residual_loss=final_values[1],
        current_jepa_loss=final_values[2],
        total_loss=final_values[3],
        core_write_frobenius=math.sqrt(max(write_sq, 0.0)),
        # The optimizer only sees detached cached targets; no teacher module or
        # teacher parameter is retained in the optimization graph.
        teacher_frozen=True,
    )


def refit_parameter_residual(
    *,
    core_old: Mapping[str, Tensor],
    slow_uncompressed: Mapping[str, LowRankFactors],
    core_new: Mapping[str, Tensor],
    residual_rank: int,
    epsilon: float,
) -> ResidualRefitResult:
    """Fit ``C_old + M_tilde - C_new`` with a deterministic low-rank SVD."""

    keys = set(core_old)
    if keys != set(slow_uncompressed) or keys != set(core_new):
        raise DeepSleepError("residual refit layer manifests differ")
    factors: Dict[str, LowRankFactors] = {}
    numerical: Dict[str, int] = {}
    after: Dict[str, int] = {}
    for key in sorted(keys):
        old = core_old[key]
        new = core_new[key]
        slow = as_factors(slow_uncompressed[key])
        if old.shape != new.shape or tuple(old.shape) != (
            slow.out_features,
            slow.in_features,
        ):
            raise DeepSleepError(f"residual refit dimensions differ for {key}")
        dense = old.to(torch.float64) - new.to(torch.float64)
        if slow.rank:
            dense = dense + slow.b.to(torch.float64) @ slow.a.to(torch.float64)
        u, singular_values, vh = torch.linalg.svd(dense, full_matrices=False)
        if singular_values.numel() and float(singular_values[0]) > 0.0:
            tolerance = (
                float(epsilon)
                * max(int(dense.shape[0]), int(dense.shape[1]))
                * float(singular_values[0])
            )
            numerical_rank = int(torch.count_nonzero(singular_values > tolerance))
        else:
            numerical_rank = 0
        chosen = min(int(residual_rank), numerical_rank)
        if chosen:
            root = torch.sqrt(torch.clamp(singular_values[:chosen], min=0.0))
            b = (u[:, :chosen] * root.unsqueeze(0)).to(
                device=old.device, dtype=old.dtype
            )
            a = (root.unsqueeze(1) * vh[:chosen, :]).to(
                device=old.device, dtype=old.dtype
            )
        else:
            b = old.new_empty((old.shape[0], 0))
            a = old.new_empty((0, old.shape[1]))
        factors[key] = LowRankFactors(b, a)
        numerical[key] = numerical_rank
        after[key] = chosen
    return ResidualRefitResult(factors, numerical, after)


def compare_functional_outputs(
    reference: Sequence[Tensor],
    approximate: Sequence[Tensor],
    *,
    epsilon: float,
) -> FunctionalComparison:
    if len(reference) != len(approximate) or not reference:
        raise DeepSleepError("functional comparison inputs are empty or misaligned")
    error_energy = 0.0
    reference_energy = 0.0
    finite = True
    for expected, actual in zip(reference, approximate):
        if expected.shape != actual.shape:
            raise DeepSleepError("functional comparison output shapes differ")
        difference = actual.to(torch.float64) - expected.to(torch.float64)
        expected64 = expected.to(torch.float64)
        finite = finite and bool(
            torch.isfinite(difference).all() and torch.isfinite(expected64).all()
        )
        error_energy += float(torch.sum(difference.square()).detach().cpu())
        reference_energy += float(torch.sum(expected64.square()).detach().cpu())
    absolute = math.sqrt(max(error_energy, 0.0))
    reference_norm = math.sqrt(max(reference_energy, 0.0))
    relative = error_energy / (reference_energy + float(epsilon))
    finite = finite and all(math.isfinite(value) for value in (absolute, relative))
    return FunctionalComparison(relative, absolute, reference_norm, finite)


__all__ = [
    "CoreDistillationResult",
    "DeepSleepController",
    "DeepSleepError",
    "DeepSleepSplit",
    "FunctionalComparison",
    "ModelTrace",
    "partition_residual_targets",
    "ResidualRefitResult",
    "ResidualTarget",
    "cache_residual_targets",
    "compare_functional_outputs",
    "distill_core_memory",
    "refit_parameter_residual",
]
