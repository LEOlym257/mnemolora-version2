"""Bounded nearest-prototype exception adapters and local replay state."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .replay_memory import ReplayWindow


class ExceptionRouterError(RuntimeError):
    pass


def _seed(seed: int, stable_id: str) -> int:
    value = hashlib.sha256(f"{int(seed)}\0{stable_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big", signed=False)


def _safe_unit(value: Any, expected_dim: Optional[int] = None) -> Tuple[Optional[torch.Tensor], str]:
    try:
        vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten().clone()
    except Exception:
        return None, "invalid_context_type"
    if vector.numel() == 0:
        return None, "empty_context"
    if expected_dim is not None and vector.numel() != expected_dim:
        return None, "context_schema_mismatch"
    if not torch.isfinite(vector).all():
        return None, "nonfinite_context"
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) <= 1.0e-12:
        return None, "zero_context"
    return vector / norm, "ok"


@dataclass(frozen=True)
class RouteDecision:
    adapter_id: Optional[str]
    similarity: Optional[float]
    reason: str

    @property
    def matched(self) -> bool:
        return self.adapter_id is not None


class BoundedLocalReplay:
    """Per-exception deterministic reservoir, mutated only on commit."""

    SCHEMA_VERSION = 1

    def __init__(self, capacity: int, seed: int) -> None:
        if int(capacity) < 0:
            raise ValueError("local replay capacity must be non-negative")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.seen_count = 0
        self.admission_count = 0
        self._windows: List[ReplayWindow] = []
        self._seen_ids: set[str] = set()
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._windows)

    def add(self, window: ReplayWindow) -> bool:
        if window.window_id in self._seen_ids:
            return False
        self._seen_ids.add(window.window_id)
        self.seen_count += 1
        if self.capacity == 0:
            return False
        stored = window.clone()
        if len(self._windows) < self.capacity:
            self._windows.append(stored)
            self.admission_count += 1
            return True
        index = self._rng.randrange(self.seen_count)
        if index < self.capacity:
            self._windows[index] = stored
            self.admission_count += 1
            return True
        return False

    def add_many(self, windows: Iterable[ReplayWindow]) -> int:
        return sum(int(self.add(window)) for window in windows)

    def windows(self) -> Tuple[ReplayWindow, ...]:
        return tuple(window.clone() for window in sorted(self._windows, key=lambda item: item.window_id))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "seed": self.seed,
            "seen_count": self.seen_count,
            "admission_count": self.admission_count,
            "windows": copy.deepcopy(self._windows),
            "seen_ids": sorted(self._seen_ids),
            "rng_state": copy.deepcopy(self._rng.getstate()),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BoundedLocalReplay":
        if int(state.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise ExceptionRouterError("unsupported local replay schema")
        result = cls(int(state["capacity"]), int(state["seed"]))
        result.seen_count = int(state["seen_count"])
        result.admission_count = int(state["admission_count"])
        result._windows = copy.deepcopy(list(state["windows"]))
        result._seen_ids = set(str(item) for item in state["seen_ids"])
        result._rng.setstate(copy.deepcopy(state["rng_state"]))
        if len(result._windows) > result.capacity:
            raise ExceptionRouterError("local replay state exceeds capacity")
        return result


@dataclass
class ExceptionAdapterRecord:
    adapter_id: str
    adapter_state: Any
    context_prototype: torch.Tensor
    context_descriptor_sum: torch.Tensor
    descriptor_count: int
    residual_prototype: Optional[torch.Tensor]
    residual_available: bool
    residual_descriptor_sum: Optional[torch.Tensor]
    residual_descriptor_count: int
    local_replay: BoundedLocalReplay
    usage_count: int = 0
    last_used_clock: int = 0
    cumulative_gain: float = 0.0
    commit_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ExceptionAdapterRecord":
        return copy.deepcopy(self)


class ExceptionRouter:
    """Nearest context prototype router with a route fixed for each episode."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        maximum_adapters: int,
        minimum_route_similarity: float,
        local_replay_windows: int,
        seed: int,
        no_match_behavior: str = "slow_only",
    ) -> None:
        if int(maximum_adapters) < 0 or int(local_replay_windows) < 0:
            raise ValueError("exception capacities must be non-negative")
        if not -1.0 <= float(minimum_route_similarity) <= 1.0:
            raise ValueError("minimum_route_similarity must be in [-1,1]")
        if no_match_behavior != "slow_only":
            raise ValueError("the compliant no-match behavior is slow_only")
        self.maximum_adapters = int(maximum_adapters)
        self.minimum_route_similarity = float(minimum_route_similarity)
        self.local_replay_windows = int(local_replay_windows)
        self.seed = int(seed)
        self.no_match_behavior = no_match_behavior
        self._records: Dict[str, ExceptionAdapterRecord] = {}
        self._next_adapter_index = 0
        self._usage_clock = 0
        self._active_episode_id: Optional[str] = None
        self._active_route: Optional[RouteDecision] = None

    @property
    def active_adapter_id(self) -> Optional[str]:
        return None if self._active_route is None else self._active_route.adapter_id

    def __len__(self) -> int:
        return len(self._records)

    def adapter_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._records))

    def get(self, adapter_id: str) -> ExceptionAdapterRecord:
        try:
            return self._records[str(adapter_id)].clone()
        except KeyError as exc:
            raise ExceptionRouterError(f"unknown exception adapter {adapter_id!r}") from exc

    def _prototype_dim(self) -> Optional[int]:
        if not self._records:
            return None
        return next(iter(self._records.values())).context_prototype.numel()

    def route(self, context: Any, *, production: bool = False) -> RouteDecision:
        vector, reason = _safe_unit(context, self._prototype_dim())
        if vector is None:
            return RouteDecision(None, None, reason)
        best_id: Optional[str] = None
        best_similarity = -float("inf")
        for adapter_id in sorted(self._records):
            prototype = self._records[adapter_id].context_prototype
            similarity = float(torch.dot(prototype, vector).item())
            if similarity > best_similarity:
                best_id, best_similarity = adapter_id, similarity
        if best_id is None:
            return RouteDecision(None, None, "empty_exception_bank")
        if not math.isfinite(best_similarity):
            return RouteDecision(None, None, "nonfinite_similarity")
        if best_similarity < self.minimum_route_similarity:
            return RouteDecision(None, best_similarity, "below_route_threshold")
        decision = RouteDecision(best_id, best_similarity, "matched")
        if production:
            self._usage_clock += 1
            record = self._records[best_id]
            record.usage_count += 1
            record.last_used_clock = self._usage_clock
        return decision

    def begin_episode(self, episode_id: str, context: Any) -> RouteDecision:
        if self._active_episode_id is not None:
            raise ExceptionRouterError("exception route is already fixed for an active episode")
        if not str(episode_id):
            raise ExceptionRouterError("episode_id must be non-empty")
        decision = self.route(context, production=True)
        self._active_episode_id = str(episode_id)
        self._active_route = decision
        return decision

    def route_for_episode(self, episode_id: str) -> RouteDecision:
        if self._active_episode_id != str(episode_id) or self._active_route is None:
            raise ExceptionRouterError("no fixed route for this episode")
        return self._active_route

    def end_episode(self, episode_id: str) -> None:
        if self._active_episode_id != str(episode_id):
            raise ExceptionRouterError("cannot end a different router episode")
        self._active_episode_id = None
        self._active_route = None

    @staticmethod
    def _raw_descriptor_sum(
        descriptors: Sequence[Any],
        *,
        label: str,
        expected_dim: Optional[int] = None,
        require_nonzero: bool,
    ) -> torch.Tensor:
        if not descriptors:
            raise ExceptionRouterError(f"at least one {label} descriptor is required")
        vectors: List[torch.Tensor] = []
        dimension = expected_dim
        for descriptor in descriptors:
            try:
                vector = (
                    torch.as_tensor(descriptor, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .flatten()
                    .clone()
                )
            except Exception as exc:
                raise ExceptionRouterError(
                    f"invalid {label} descriptor type"
                ) from exc
            if vector.numel() == 0:
                raise ExceptionRouterError(f"empty {label} descriptor")
            if dimension is not None and vector.numel() != dimension:
                raise ExceptionRouterError(f"{label} descriptor schema mismatch")
            if not torch.isfinite(vector).all():
                raise ExceptionRouterError(f"non-finite {label} descriptor")
            if require_nonzero and float(torch.linalg.vector_norm(vector)) <= 1.0e-12:
                raise ExceptionRouterError(f"zero {label} descriptor")
            dimension = vector.numel()
            vectors.append(vector)
        result = torch.stack(vectors, dim=0).sum(dim=0)
        if not torch.isfinite(result).all():
            raise ExceptionRouterError(f"non-finite {label} descriptor sum")
        return result

    @staticmethod
    def _prototype_from_sum(
        descriptor_sum: torch.Tensor,
        *,
        label: str,
        allow_zero: bool,
    ) -> Tuple[torch.Tensor, bool]:
        vector = torch.as_tensor(
            descriptor_sum, dtype=torch.float32
        ).detach().cpu().flatten().clone()
        if vector.numel() == 0 or not torch.isfinite(vector).all():
            raise ExceptionRouterError(f"invalid {label} descriptor sum")
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm):
            raise ExceptionRouterError(f"non-finite {label} descriptor norm")
        if float(norm) <= 1.0e-12:
            if allow_zero:
                return torch.zeros_like(vector), False
            raise ExceptionRouterError(f"{label} prototype is undefined from a zero sum")
        return vector / norm, True

    @classmethod
    def _context_statistics(
        cls,
        descriptors: Sequence[Any],
        *,
        expected_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        descriptor_sum = cls._raw_descriptor_sum(
            descriptors,
            label="context",
            expected_dim=expected_dim,
            require_nonzero=True,
        )
        prototype, _ = cls._prototype_from_sum(
            descriptor_sum,
            label="context",
            allow_zero=False,
        )
        return descriptor_sum, prototype

    @classmethod
    def _residual_statistics(
        cls,
        descriptors: Sequence[Any],
        *,
        expected_dim: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        if not descriptors:
            return None, None, False
        descriptor_sum = cls._raw_descriptor_sum(
            descriptors,
            label="residual",
            expected_dim=expected_dim,
            require_nonzero=False,
        )
        prototype, available = cls._prototype_from_sum(
            descriptor_sum,
            label="residual",
            allow_zero=True,
        )
        return descriptor_sum, prototype, available

    def preview_next_adapter_id(self) -> str:
        return f"exception-{self._next_adapter_index:08d}"

    def _eviction_candidate(self) -> Optional[str]:
        candidates = [
            record
            for record in self._records.values()
            if record.adapter_id != self.active_adapter_id
        ]
        if not candidates:
            return None
        record = min(
            candidates,
            key=lambda item: (item.last_used_clock, item.cumulative_gain, item.adapter_id),
        )
        return record.adapter_id

    def commit_new(
        self,
        *,
        adapter_state: Any,
        context_descriptors: Sequence[Any],
        residual_descriptors: Sequence[Any] = (),
        local_windows: Iterable[ReplayWindow] = (),
        validation_gain: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
        adapter_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        if self.maximum_adapters <= 0:
            raise ExceptionRouterError("exception adapters are disabled")
        if not math.isfinite(float(validation_gain)):
            raise ExceptionRouterError("validation_gain must be finite")
        proposed_id = str(adapter_id or self.preview_next_adapter_id())
        if proposed_id in self._records:
            raise ExceptionRouterError(f"adapter id already exists: {proposed_id}")
        expected_preview = self.preview_next_adapter_id()
        if adapter_id is not None and proposed_id != expected_preview:
            raise ExceptionRouterError("new adapter id must follow the deterministic sequence")
        context_descriptor_sum, context_prototype = self._context_statistics(
            context_descriptors,
            expected_dim=self._prototype_dim(),
        )
        if self._prototype_dim() is not None and context_prototype.numel() != self._prototype_dim():
            raise ExceptionRouterError("new exception context schema differs from the bank")
        (
            residual_descriptor_sum,
            residual_prototype,
            residual_available,
        ) = self._residual_statistics(residual_descriptors)
        evicted: Optional[str] = None
        if len(self._records) >= self.maximum_adapters:
            evicted = self._eviction_candidate()
            if evicted is None:
                raise ExceptionRouterError("exception bank is full and no adapter is safely evictable")
        local = BoundedLocalReplay(self.local_replay_windows, _seed(self.seed, proposed_id))
        local.add_many(local_windows)
        # A newly committed exception was the successful outcome of this
        # production episode even though there was no matching route at its
        # start.  Account for that use inside the surrounding persistent
        # transaction so a failed commit restores the clock and the record
        # together.  Replacement commits are already counted by the fixed
        # production route in ``begin_episode`` and must not count twice.
        next_usage_clock = self._usage_clock + 1
        record = ExceptionAdapterRecord(
            adapter_id=proposed_id,
            adapter_state=copy.deepcopy(adapter_state),
            context_prototype=context_prototype,
            context_descriptor_sum=context_descriptor_sum,
            descriptor_count=len(context_descriptors),
            residual_prototype=residual_prototype,
            residual_available=residual_available,
            residual_descriptor_sum=residual_descriptor_sum,
            residual_descriptor_count=len(residual_descriptors),
            local_replay=local,
            usage_count=1,
            last_used_clock=next_usage_clock,
            cumulative_gain=float(validation_gain),
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        if evicted is not None:
            del self._records[evicted]
        self._records[proposed_id] = record
        self._next_adapter_index += 1
        self._usage_clock = next_usage_clock
        return proposed_id, evicted

    def commit_replace(
        self,
        adapter_id: str,
        *,
        adapter_state: Any,
        context_descriptors: Sequence[Any],
        residual_descriptors: Sequence[Any] = (),
        local_windows: Iterable[ReplayWindow] = (),
        validation_gain: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        adapter_id = str(adapter_id)
        if adapter_id not in self._records:
            raise ExceptionRouterError(f"cannot replace unknown adapter {adapter_id!r}")
        if not context_descriptors:
            raise ExceptionRouterError("replacement requires accepted context descriptors")
        if not math.isfinite(float(validation_gain)):
            raise ExceptionRouterError("validation_gain must be finite")
        old = self._records[adapter_id]
        new_context_sum, _ = self._context_statistics(
            context_descriptors,
            expected_dim=old.context_descriptor_sum.numel(),
        )
        context_descriptor_sum = old.context_descriptor_sum + new_context_sum
        new_prototype, _ = self._prototype_from_sum(
            context_descriptor_sum,
            label="context",
            allow_zero=False,
        )
        new_residual_count = len(residual_descriptors)
        if not residual_descriptors:
            residual_descriptor_sum = old.residual_descriptor_sum
            residual_prototype = old.residual_prototype
            residual_available = old.residual_available
        else:
            expected_residual_dim = (
                None
                if old.residual_descriptor_sum is None
                else old.residual_descriptor_sum.numel()
            )
            new_residual_sum, _, _ = self._residual_statistics(
                residual_descriptors,
                expected_dim=expected_residual_dim,
            )
            assert new_residual_sum is not None
            residual_descriptor_sum = (
                new_residual_sum
                if old.residual_descriptor_sum is None
                else old.residual_descriptor_sum + new_residual_sum
            )
            residual_prototype, residual_available = self._prototype_from_sum(
                residual_descriptor_sum,
                label="residual",
                allow_zero=True,
            )
        residual_count = old.residual_descriptor_count + new_residual_count
        local = BoundedLocalReplay.from_state_dict(old.local_replay.state_dict())
        local.add_many(local_windows)
        updated_metadata = copy.deepcopy(old.metadata)
        updated_metadata.update(dict(metadata or {}))
        self._records[adapter_id] = ExceptionAdapterRecord(
            adapter_id=adapter_id,
            adapter_state=copy.deepcopy(adapter_state),
            context_prototype=new_prototype,
            context_descriptor_sum=context_descriptor_sum,
            descriptor_count=old.descriptor_count + len(context_descriptors),
            residual_prototype=copy.deepcopy(residual_prototype),
            residual_available=residual_available,
            residual_descriptor_sum=copy.deepcopy(residual_descriptor_sum),
            residual_descriptor_count=residual_count,
            local_replay=local,
            usage_count=old.usage_count,
            last_used_clock=old.last_used_clock,
            cumulative_gain=old.cumulative_gain + float(validation_gain),
            commit_count=old.commit_count + 1,
            metadata=updated_metadata,
        )

    def state_dict(self) -> Dict[str, Any]:
        records = {}
        for adapter_id, record in sorted(self._records.items()):
            records[adapter_id] = {
                "adapter_state": copy.deepcopy(record.adapter_state),
                "context_prototype": record.context_prototype.clone(),
                "context_descriptor_sum": record.context_descriptor_sum.clone(),
                "descriptor_count": record.descriptor_count,
                "residual_prototype": copy.deepcopy(record.residual_prototype),
                "residual_available": record.residual_available,
                "residual_descriptor_sum": copy.deepcopy(
                    record.residual_descriptor_sum
                ),
                "residual_descriptor_count": record.residual_descriptor_count,
                "local_replay": record.local_replay.state_dict(),
                "usage_count": record.usage_count,
                "last_used_clock": record.last_used_clock,
                "cumulative_gain": record.cumulative_gain,
                "commit_count": record.commit_count,
                "metadata": copy.deepcopy(record.metadata),
            }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": {
                "maximum_adapters": self.maximum_adapters,
                "minimum_route_similarity": self.minimum_route_similarity,
                "local_replay_windows": self.local_replay_windows,
                "seed": self.seed,
                "no_match_behavior": self.no_match_behavior,
            },
            "records": records,
            "next_adapter_index": self._next_adapter_index,
            "usage_clock": self._usage_clock,
            "active_episode_id": self._active_episode_id,
            "active_route": copy.deepcopy(self._active_route),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("schema_version", -1))
        if version not in {1, self.SCHEMA_VERSION}:
            raise ExceptionRouterError("unsupported exception-router schema")
        expected = {
            "maximum_adapters": self.maximum_adapters,
            "minimum_route_similarity": self.minimum_route_similarity,
            "local_replay_windows": self.local_replay_windows,
            "seed": self.seed,
            "no_match_behavior": self.no_match_behavior,
        }
        if dict(state.get("config", {})) != expected:
            raise ExceptionRouterError("exception-router checkpoint/config mismatch")
        restored: Dict[str, ExceptionAdapterRecord] = {}
        context_dimension: Optional[int] = None
        for adapter_id, raw in sorted(dict(state.get("records", {})).items()):
            prototype, reason = _safe_unit(
                raw["context_prototype"], context_dimension
            )
            if prototype is None:
                raise ExceptionRouterError(f"invalid restored prototype: {reason}")
            context_dimension = prototype.numel()
            descriptor_count = int(raw["descriptor_count"])
            if descriptor_count <= 0:
                raise ExceptionRouterError("restored prototype counts are invalid")
            if version >= 2:
                if "context_descriptor_sum" not in raw:
                    raise ExceptionRouterError(
                        "schema-2 router record is missing context sufficient statistics"
                    )
                context_sum = torch.as_tensor(
                    raw["context_descriptor_sum"], dtype=torch.float32
                ).detach().cpu().flatten().clone()
                if (
                    context_sum.numel() != prototype.numel()
                    or not torch.isfinite(context_sum).all()
                ):
                    raise ExceptionRouterError(
                        "restored context descriptor sum is invalid"
                    )
                sum_prototype, _ = self._prototype_from_sum(
                    context_sum,
                    label="context",
                    allow_zero=False,
                )
                if not torch.allclose(
                    sum_prototype, prototype, atol=1.0e-6, rtol=1.0e-5
                ):
                    raise ExceptionRouterError(
                        "restored context prototype disagrees with descriptor sum"
                    )
            else:
                # Schema 1 persisted only a normalized prototype and count.
                # Exact magnitudes are unrecoverable; prototype*count is the
                # explicit compatibility statistic and reproduces all legacy
                # equal-unit-vector updates before accepting new raw vectors.
                context_sum = prototype * float(descriptor_count)

            residual = copy.deepcopy(raw.get("residual_prototype"))
            residual_available = bool(raw["residual_available"])
            if "residual_descriptor_count" in raw:
                residual_count = int(raw["residual_descriptor_count"])
            else:
                residual_count = descriptor_count if residual is not None else 0
            if residual_count < 0:
                raise ExceptionRouterError("restored residual count is invalid")
            if residual_available:
                residual, residual_reason = _safe_unit(residual, None)
                if residual is None or residual_count == 0:
                    raise ExceptionRouterError(
                        f"invalid restored residual prototype: {residual_reason}"
                    )
            elif residual is not None:
                residual = torch.as_tensor(residual, dtype=torch.float32).detach().cpu().flatten()
                if not torch.isfinite(residual).all() or bool(torch.any(residual != 0)):
                    raise ExceptionRouterError(
                        "unavailable residual prototype must be an explicit finite zero"
                    )
            if version >= 2:
                if "residual_descriptor_sum" not in raw:
                    raise ExceptionRouterError(
                        "schema-2 router record is missing residual sufficient statistics"
                    )
                raw_residual_sum = raw.get("residual_descriptor_sum")
                residual_sum = (
                    None
                    if raw_residual_sum is None
                    else torch.as_tensor(raw_residual_sum, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .flatten()
                    .clone()
                )
            else:
                residual_sum = (
                    None
                    if residual is None
                    else residual * float(residual_count)
                )
            if residual_count == 0:
                if residual_sum is not None or residual is not None or residual_available:
                    raise ExceptionRouterError(
                        "zero residual count must have no residual statistics"
                    )
            else:
                if (
                    residual_sum is None
                    or residual_sum.numel() == 0
                    or not torch.isfinite(residual_sum).all()
                ):
                    raise ExceptionRouterError(
                        "restored residual descriptor sum is invalid"
                    )
                sum_residual, sum_available = self._prototype_from_sum(
                    residual_sum,
                    label="residual",
                    allow_zero=True,
                )
                if sum_available != residual_available:
                    raise ExceptionRouterError(
                        "restored residual availability disagrees with descriptor sum"
                    )
                if residual is None or not torch.allclose(
                    sum_residual, residual, atol=1.0e-6, rtol=1.0e-5
                ):
                    raise ExceptionRouterError(
                        "restored residual prototype disagrees with descriptor sum"
                    )
            restored[str(adapter_id)] = ExceptionAdapterRecord(
                adapter_id=str(adapter_id),
                adapter_state=copy.deepcopy(raw["adapter_state"]),
                context_prototype=prototype,
                context_descriptor_sum=context_sum,
                descriptor_count=descriptor_count,
                residual_prototype=residual,
                residual_available=residual_available,
                residual_descriptor_sum=residual_sum,
                residual_descriptor_count=residual_count,
                local_replay=BoundedLocalReplay.from_state_dict(raw["local_replay"]),
                usage_count=int(raw["usage_count"]),
                last_used_clock=int(raw["last_used_clock"]),
                cumulative_gain=float(raw["cumulative_gain"]),
                commit_count=int(raw["commit_count"]),
                metadata=copy.deepcopy(dict(raw.get("metadata", {}))),
            )
        if len(restored) > self.maximum_adapters:
            raise ExceptionRouterError("restored exception bank exceeds configured maximum")
        dims = {record.context_prototype.numel() for record in restored.values()}
        if len(dims) > 1:
            raise ExceptionRouterError("restored exception bank has mixed context schemas")
        self._records = restored
        self._next_adapter_index = int(state.get("next_adapter_index", 0))
        self._usage_clock = int(state.get("usage_clock", 0))
        self._active_episode_id = state.get("active_episode_id")
        self._active_route = copy.deepcopy(state.get("active_route"))


__all__ = [
    "BoundedLocalReplay",
    "ExceptionAdapterRecord",
    "ExceptionRouter",
    "ExceptionRouterError",
    "RouteDecision",
]
