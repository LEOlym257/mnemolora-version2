"""All-state transactional snapshots for FD-PSC commits and rollback."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import torch

try:  # NumPy is part of AdaJEPA, but keep the transaction helper import-safe.
    import numpy as np
except Exception:  # pragma: no cover - only for unusually minimal installations
    np = None  # type: ignore[assignment]


class TransactionError(RuntimeError):
    pass


class TransactionRollbackError(TransactionError):
    pass


@dataclass
class Participant:
    snapshot: Callable[[], Any]
    restore: Callable[[Any], None]

    @classmethod
    def from_stateful(cls, value: Any) -> "Participant":
        if not callable(getattr(value, "state_dict", None)) or not callable(
            getattr(value, "load_state_dict", None)
        ):
            raise TypeError("transaction participant requires state_dict/load_state_dict")
        return cls(value.state_dict, value.load_state_dict)


@dataclass
class RNGSnapshot:
    python_state: object
    numpy_state: object
    torch_cpu_state: torch.Tensor
    torch_cuda_states: Optional[list]
    generator_states: Dict[str, torch.Tensor]

    @classmethod
    def capture(cls, generators: Optional[Mapping[str, torch.Generator]] = None) -> "RNGSnapshot":
        numpy_state = copy.deepcopy(np.random.get_state()) if np is not None else None
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        return cls(
            python_state=copy.deepcopy(random.getstate()),
            numpy_state=numpy_state,
            torch_cpu_state=torch.random.get_rng_state().clone(),
            torch_cuda_states=cuda_states,
            generator_states={
                str(name): generator.get_state().clone()
                for name, generator in sorted((generators or {}).items())
            },
        )

    def restore(self, generators: Optional[Mapping[str, torch.Generator]] = None) -> None:
        random.setstate(copy.deepcopy(self.python_state))
        if np is not None and self.numpy_state is not None:
            np.random.set_state(copy.deepcopy(self.numpy_state))
        torch.random.set_rng_state(self.torch_cpu_state.clone())
        if self.torch_cuda_states is not None:
            if not torch.cuda.is_available():
                raise TransactionRollbackError("CUDA RNG state cannot be restored because CUDA is unavailable")
            if len(self.torch_cuda_states) != torch.cuda.device_count():
                raise TransactionRollbackError("CUDA device count changed during transaction")
            torch.cuda.set_rng_state_all([state.clone() for state in self.torch_cuda_states])
        supplied = dict(generators or {})
        if set(supplied) != set(self.generator_states):
            raise TransactionRollbackError("dedicated generator set changed during transaction")
        for name, state in self.generator_states.items():
            supplied[name].set_state(state.clone())


@dataclass
class TransactionSnapshot:
    participant_states: Dict[str, Any]
    rng: RNGSnapshot


class StateTransaction:
    """Snapshot all registered persistent components and every relevant RNG.

    The context rolls back unless :meth:`commit` is called.  This makes an
    early return, gate failure, checkpoint failure, or raised exception safe by
    default.
    """

    def __init__(
        self,
        participants: Mapping[str, Any],
        *,
        generators: Optional[Mapping[str, torch.Generator]] = None,
        name: str = "fd_psc_commit",
    ) -> None:
        if not participants:
            raise ValueError("a state transaction requires at least one participant")
        self.name = str(name)
        self._participants: Dict[str, Participant] = {}
        for key, value in sorted(participants.items()):
            participant = value if isinstance(value, Participant) else Participant.from_stateful(value)
            self._participants[str(key)] = participant
        self._generators = dict(generators or {})
        self._snapshot: Optional[TransactionSnapshot] = None
        self._active = False
        self._committed = False
        self.rollback_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def committed(self) -> bool:
        return self._committed

    def begin(self) -> "StateTransaction":
        if self._active or self._snapshot is not None:
            raise TransactionError("transaction objects are single-use")
        states: Dict[str, Any] = {}
        for name, participant in self._participants.items():
            states[name] = copy.deepcopy(participant.snapshot())
        self._snapshot = TransactionSnapshot(states, RNGSnapshot.capture(self._generators))
        self._active = True
        return self

    def commit(self) -> None:
        if not self._active or self._snapshot is None:
            raise TransactionError("cannot commit an inactive transaction")
        self._committed = True
        self._active = False

    def rollback(self, reason: str = "explicit_rollback") -> None:
        if self._snapshot is None:
            raise TransactionError("cannot rollback before begin")
        if self._committed:
            raise TransactionError("cannot rollback an already committed transaction")
        failures = []
        # Restore participants in reverse registration order in case a later
        # component references an earlier one.
        for name in reversed(list(self._participants)):
            try:
                self._participants[name].restore(copy.deepcopy(self._snapshot.participant_states[name]))
            except Exception as exc:  # collect all failures; do not leave later state unrestored
                failures.append(f"{name}: {exc}")
        try:
            self._snapshot.rng.restore(self._generators)
        except Exception as exc:
            failures.append(f"rng: {exc}")
        self.rollback_reason = str(reason)
        self._active = False
        if failures:
            raise TransactionRollbackError("rollback was incomplete: " + "; ".join(failures))

    def __enter__(self) -> "StateTransaction":
        return self.begin()

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], traceback: Any) -> bool:
        if self._committed:
            return False
        reason = f"exception:{type(exc).__name__}:{exc}" if exc is not None else "context_exited_without_commit"
        try:
            self.rollback(reason)
        except TransactionRollbackError:
            if exc is not None:
                raise
            raise
        return False


__all__ = [
    "Participant",
    "RNGSnapshot",
    "StateTransaction",
    "TransactionError",
    "TransactionRollbackError",
    "TransactionSnapshot",
]
