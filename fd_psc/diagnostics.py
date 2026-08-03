"""Diagnostics and invariant helpers; fallbacks are always explicit events."""

from __future__ import annotations

import dataclasses
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .checkpoint import state_content_hash


class DiagnosticError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiagnosticEvent:
    sequence: int
    timestamp_ns: int
    level: str
    code: str
    message: str
    episode_id: Optional[str] = None
    logical_layer_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)


class Diagnostics:
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._events: List[DiagnosticEvent] = []
        self._sequence = 0

    def record(
        self,
        level: str,
        code: str,
        message: str,
        *,
        episode_id: Optional[str] = None,
        logical_layer_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticEvent:
        normalized_level = str(level).lower()
        if normalized_level not in ("info", "warning", "error"):
            raise ValueError("diagnostic level must be info/warning/error")
        if not str(code) or not str(message):
            raise ValueError("diagnostic code and message must be non-empty")
        event = DiagnosticEvent(
            sequence=self._sequence,
            timestamp_ns=time.time_ns(),
            level=normalized_level,
            code=str(code),
            message=str(message),
            episode_id=None if episode_id is None else str(episode_id),
            logical_layer_id=None if logical_layer_id is None else str(logical_layer_id),
            details=dict(details or {}),
        )
        self._sequence += 1
        self._events.append(event)
        return event

    def fallback(self, code: str, reason: str, **dimensions: Any) -> DiagnosticEvent:
        return self.record("warning", code, f"fallback: {reason}", details=dimensions)

    def events(self) -> Tuple[DiagnosticEvent, ...]:
        return tuple(self._events)

    def state_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "sequence": self._sequence}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise DiagnosticError("unsupported diagnostics schema")
        self._sequence = int(state.get("sequence", 0))
        self._events = []


def assert_finite_tree(value: Any, *, path: str = "root") -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise DiagnosticError(f"non-finite tensor at {path}")
        return
    if np is not None and isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise DiagnosticError(f"non-finite array at {path}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticError(f"non-finite float at {path}")
    if dataclasses.is_dataclass(value):
        assert_finite_tree(dataclasses.asdict(value), path=path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_finite_tree(item, path=f"{path}[{index}]")


def tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if np is not None and isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    if dataclasses.is_dataclass(value):
        return tree_nbytes(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return sum(tree_nbytes(key) + tree_nbytes(item) for key, item in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(tree_nbytes(item) for item in value)
    return sys.getsizeof(value)


def bitwise_state_equal(left: Any, right: Any) -> bool:
    try:
        return state_content_hash(left) == state_content_hash(right)
    except Exception:
        return False


def deterministic_environment_report() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


__all__ = [
    "DiagnosticError",
    "DiagnosticEvent",
    "Diagnostics",
    "assert_finite_tree",
    "bitwise_state_equal",
    "deterministic_environment_report",
    "tree_nbytes",
]
