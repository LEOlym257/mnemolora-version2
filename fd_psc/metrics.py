"""Structured FD-PSC metrics with JSONL/CSV export and checkpointable counters."""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch


class MetricsError(RuntimeError):
    pass


REQUIRED_METRIC_NAMES = frozenset(
    {
        "current_jepa_loss",
        "fast_adaptation_gain",
        "external_calibration_gain",
        "commit_query_gain",
        "report_test_gain",
        "planning_success",
        "historical_replay_loss",
        "worst_context_regression",
        "anchor_loss",
        "canary_planning_regression",
        "plasticity_gain",
        "plasticity_gate_ratio",
        "next_episode_early_loss_decline",
        "time_to_threshold_replans",
        "per_context_loss",
        "forgetting",
        "backward_transfer",
        "rho_history",
        "rho_anchor",
        "gradient_anchor_cosine",
        "conflict_ema",
        "active_constraints",
        "active_constraint_count",
        "slice_trigger",
        "gradient_correction_norm",
        "spectral_drift",
        "slow_rank",
        "episodic_rank",
        "activation_subspace_rank",
        "lambda_distribution",
        "p_distribution",
        "spectral_energy",
        "functional_error",
        "alpha_shared",
        "alpha_safe",
        "online_update_latency_s",
        "gradient_collection_latency_s",
        "slice_latency_s",
        "sleep_latency_s",
        "replay_memory_bytes",
        "checkpoint_bytes",
        "adapter_parameter_count",
        "exception_count",
        "routed_exception_id",
        "routed_exception_similarity",
        "route_rejection_count",
        "calibration_candidate_count",
        "final_proposal_type",
        "commit_query_gate_invocation_count",
        "rollback_count",
    }
)


METRIC_STATUSES = frozenset(
    {
        "available",
        "unavailable",
        "not_applicable",
        "pending",
        "not_reached",
        "insufficient_observations",
    }
)


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise MetricsError("metric tensors must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricsError("metric value must be finite")
        return value
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"metric value is not scalar: {type(value).__name__}") from exc
    if not math.isfinite(number):
        raise MetricsError("metric value must be finite")
    return number


@dataclass(frozen=True)
class MetricEvent:
    sequence: int
    timestamp_ns: int
    name: str
    value: Any
    episode_id: Optional[str] = None
    replan_index: Optional[int] = None
    logical_layer_id: Optional[str] = None
    context_identifier: Optional[str] = None
    tags: Mapping[str, Any] = field(default_factory=dict)


class StructuredMetrics:
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._events: List[MetricEvent] = []
        self._counters: Dict[str, int] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def record(
        self,
        name: str,
        value: Any,
        *,
        episode_id: Optional[str] = None,
        replan_index: Optional[int] = None,
        logical_layer_id: Optional[str] = None,
        context_identifier: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> MetricEvent:
        metric_name = str(name)
        if not metric_name:
            raise MetricsError("metric name must be non-empty")
        clean_tags = {str(key): _scalar(item) for key, item in dict(tags or {}).items()}
        with self._lock:
            event = MetricEvent(
                sequence=self._sequence,
                timestamp_ns=time.time_ns(),
                name=metric_name,
                value=_scalar(value),
                episode_id=None if episode_id is None else str(episode_id),
                replan_index=None if replan_index is None else int(replan_index),
                logical_layer_id=None if logical_layer_id is None else str(logical_layer_id),
                context_identifier=None if context_identifier is None else str(context_identifier),
                tags=clean_tags,
            )
            self._sequence += 1
            self._events.append(event)
            return event

    def record_nullable(
        self,
        name: str,
        value: Any,
        *,
        status: str,
        reason: Optional[str] = None,
        episode_id: Optional[str] = None,
        replan_index: Optional[int] = None,
        logical_layer_id: Optional[str] = None,
        context_identifier: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> MetricEvent:
        """Record a scalar metric whose availability is explicit.

        Section 28 contains measurements that are legitimately unavailable at
        cold start (for example backward transfer without replay, or a route
        similarity with an empty exception bank).  Encoding those states as a
        fabricated zero makes aggregate reports incorrect.  This helper keeps
        the ordinary scalar event format while requiring a machine-readable
        status and an optional reason for every nullable value.
        """

        clean_status = str(status)
        if clean_status not in METRIC_STATUSES:
            raise MetricsError(
                f"unsupported metric status {clean_status!r}; "
                f"expected one of {sorted(METRIC_STATUSES)}"
            )
        if value is None and clean_status == "available":
            raise MetricsError("an available metric must have a value")
        if value is not None and clean_status != "available":
            raise MetricsError(
                "a non-available metric status must use a null value"
            )
        clean_tags = dict(tags or {})
        if "status" in clean_tags or "reason" in clean_tags:
            raise MetricsError("status/reason are reserved nullable metric tags")
        clean_tags["status"] = clean_status
        if reason is not None:
            clean_tags["reason"] = str(reason)
        return self.record(
            name,
            value,
            episode_id=episode_id,
            replan_index=replan_index,
            logical_layer_id=logical_layer_id,
            context_identifier=context_identifier,
            tags=clean_tags,
        )

    def increment(self, name: str, amount: int = 1, *, episode_id: Optional[str] = None) -> int:
        if not isinstance(amount, int):
            raise MetricsError("counter increment must be an integer")
        with self._lock:
            self._counters[str(name)] = self._counters.get(str(name), 0) + amount
            value = self._counters[str(name)]
        self.record(str(name), value, episode_id=episode_id, tags={"kind": "counter"})
        return value

    @contextmanager
    def timer(self, metric_name: str, **dimensions: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(metric_name, time.perf_counter() - start, **dimensions)

    def events(self) -> Tuple[MetricEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(str(name), 0)

    def write_jsonl(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for event in self.events():
                handle.write(json.dumps(asdict(event), sort_keys=True, allow_nan=False) + "\n")

    def write_csv(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "sequence",
            "timestamp_ns",
            "name",
            "value",
            "episode_id",
            "replan_index",
            "logical_layer_id",
            "context_identifier",
            "tags_json",
        ]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event in self.events():
                row = asdict(event)
                row["tags_json"] = json.dumps(row.pop("tags"), sort_keys=True, allow_nan=False)
                writer.writerow(row)

    def state_dict(self) -> Dict[str, Any]:
        # Events are diagnostic output, not algorithm state. Persistent counters
        # and sequence are enough to keep IDs reproducible across resume.
        with self._lock:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "sequence": self._sequence,
                "counters": dict(self._counters),
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise MetricsError("unsupported metrics schema")
        with self._lock:
            self._sequence = int(state.get("sequence", 0))
            self._counters = {str(key): int(value) for key, value in dict(state.get("counters", {})).items()}
            self._events = []


__all__ = [
    "METRIC_STATUSES",
    "MetricEvent",
    "MetricsError",
    "REQUIRED_METRIC_NAMES",
    "StructuredMetrics",
]
