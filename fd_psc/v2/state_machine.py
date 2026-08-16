"""Small, auditable state machine for the FSD V2 wake/sleep lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class FSDV2StateError(RuntimeError):
    """Raised when an FSD V2 lifecycle transition is not legal."""


class FSDV2State(str, Enum):
    IDLE = "idle"
    EPISODE_WAKE = "episode_wake"
    SLEEP_GEOMETRY = "sleep_geometry"
    SLEEP_RTRC = "sleep_rtrc"
    SLEEP_COMPRESS = "sleep_compress"
    DEEP_SLEEP = "deep_sleep"
    COMMIT = "commit"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class FSDV2Transition:
    sequence: int
    source: FSDV2State
    target: FSDV2State
    reason: str
    episode_id: Optional[str]


class FSDV2StateMachine:
    """Enforce the single-candidate FSD V2 wake-to-commit path."""

    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.state = FSDV2State.IDLE
        self.episode_id: Optional[str] = None
        self._sequence = 0
        self._transitions: list[FSDV2Transition] = []

    @property
    def active(self) -> bool:
        return self.state is not FSDV2State.IDLE

    def transitions(self) -> Tuple[FSDV2Transition, ...]:
        return tuple(self._transitions)

    def _transition(
        self,
        expected: FSDV2State,
        target: FSDV2State,
        reason: str,
    ) -> None:
        if self.state is not expected:
            raise FSDV2StateError(
                f"illegal FSD V2 transition {self.state.value} -> {target.value}; "
                f"expected {expected.value}"
            )
        record = FSDV2Transition(
            sequence=self._sequence,
            source=self.state,
            target=target,
            reason=str(reason),
            episode_id=self.episode_id,
        )
        self._sequence += 1
        self._transitions.append(record)
        self.state = target

    def begin_episode(self, episode_id: str) -> None:
        value = str(episode_id)
        if not value:
            raise FSDV2StateError("episode_id must be non-empty")
        if self.state is not FSDV2State.IDLE:
            raise FSDV2StateError("cannot begin an episode while FSD V2 is active")
        self.episode_id = value
        self._transition(FSDV2State.IDLE, FSDV2State.EPISODE_WAKE, "begin_episode")

    def enter_sleep_geometry(self) -> None:
        self._transition(
            FSDV2State.EPISODE_WAKE,
            FSDV2State.SLEEP_GEOMETRY,
            "build_fresh_replay_geometry",
        )

    def enter_sleep_rtrc(self) -> None:
        self._transition(
            FSDV2State.SLEEP_GEOMETRY,
            FSDV2State.SLEEP_RTRC,
            "project_shared_rtrc",
        )

    def enter_sleep_compress(self) -> None:
        self._transition(
            FSDV2State.SLEEP_RTRC,
            FSDV2State.SLEEP_COMPRESS,
            "fixed_rank_compression",
        )

    def enter_deep_sleep(self) -> None:
        self._transition(
            FSDV2State.SLEEP_COMPRESS,
            FSDV2State.DEEP_SLEEP,
            "residual_distillation_rank_recycling",
        )

    def enter_commit(self) -> None:
        if self.state not in {FSDV2State.SLEEP_COMPRESS, FSDV2State.DEEP_SLEEP}:
            raise FSDV2StateError(
                f"illegal FSD V2 transition {self.state.value} -> commit; "
                "expected sleep_compress or deep_sleep"
            )
        self._transition(self.state, FSDV2State.COMMIT, "commit")

    def finish_commit(self) -> None:
        self._transition(FSDV2State.COMMIT, FSDV2State.IDLE, "commit_complete")
        self.episode_id = None

    def finish_without_sleep(self, reason: str) -> None:
        self._transition(FSDV2State.EPISODE_WAKE, FSDV2State.IDLE, reason)
        self.episode_id = None

    def rollback(self, reason: str) -> None:
        if self.state is FSDV2State.IDLE:
            return
        source = self.state
        record = FSDV2Transition(
            sequence=self._sequence,
            source=source,
            target=FSDV2State.ROLLBACK,
            reason=str(reason),
            episode_id=self.episode_id,
        )
        self._sequence += 1
        self._transitions.append(record)
        self.state = FSDV2State.ROLLBACK
        self._transition(FSDV2State.ROLLBACK, FSDV2State.IDLE, "rollback_complete")
        self.episode_id = None

    def state_dict(self) -> Dict[str, Any]:
        if self.state is not FSDV2State.IDLE:
            raise FSDV2StateError("FSD V2 may only checkpoint between episodes")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "state": self.state.value,
            "sequence": self._sequence,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("schema_version", -1))
        if version == 1:
            raise FSDV2StateError(
                "V1 sidecar cannot be loaded as FSD V2 without explicit migration."
            )
        if version != self.SCHEMA_VERSION:
            raise FSDV2StateError(f"unsupported FSD V2 state-machine schema {version}")
        if str(state.get("state")) != FSDV2State.IDLE.value:
            raise FSDV2StateError("FSD V2 checkpoints must contain an idle state")
        sequence = int(state.get("sequence", 0))
        if sequence < 0:
            raise FSDV2StateError("state-machine sequence must be non-negative")
        self.state = FSDV2State.IDLE
        self.episode_id = None
        self._sequence = sequence
        self._transitions = []


__all__ = [
    "FSDV2State",
    "FSDV2StateError",
    "FSDV2StateMachine",
    "FSDV2Transition",
]
