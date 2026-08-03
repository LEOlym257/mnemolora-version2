"""Strict FD-PSC episode and single-proposal state machine.

The state machine deliberately owns no model tensors.  It is the small,
serializable authority that prevents a sleep implementation from accidentally
evaluating two proposals on commit-query data or committing twice.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class StateMachineError(RuntimeError):
    """Raised for an illegal FD-PSC lifecycle transition."""


class FDPSCState(str, Enum):
    IDLE = "IDLE"
    EPISODE_PILOT = "EPISODE_PILOT"
    EPISODE_CENTERED = "EPISODE_CENTERED"
    SLEEP_CALIBRATION = "SLEEP_CALIBRATION"
    REPAIR = "REPAIR"
    FINAL_PROPOSAL_READY = "FINAL_PROPOSAL_READY"
    REJECT_NO_PROPOSAL = "REJECT_NO_PROPOSAL"
    FINAL_GATE = "FINAL_GATE"
    COMMIT_SLOW = "COMMIT_SLOW"
    COMMIT_EXCEPTION = "COMMIT_EXCEPTION"
    REJECT_QUERY = "REJECT_QUERY"
    ROLLBACK = "ROLLBACK"


class ProposalType(str, Enum):
    GLOBAL_SLOW = "global_slow"
    NEW_EXCEPTION = "new_exception"
    REPLACE_EXCEPTION = "replace_exception"


@dataclass(frozen=True)
class FinalProposal:
    proposal_id: str
    proposal_type: ProposalType
    payload: Any
    calibration_metrics: Mapping[str, Any] = field(default_factory=dict)
    candidate_count: int = 1

    def __post_init__(self) -> None:
        if not str(self.proposal_id):
            raise ValueError("proposal_id must be non-empty")
        if int(self.candidate_count) < 1:
            raise ValueError("candidate_count must be positive")


@dataclass(frozen=True)
class TransitionRecord:
    index: int
    episode_id: Optional[str]
    old_state: FDPSCState
    new_state: FDPSCState
    reason: str


class FDPSCStateMachine:
    """Validate one Pilot/Centered/sleep/final-query lifecycle at a time."""

    SCHEMA_VERSION = 1
    _TERMINAL = {
        FDPSCState.REJECT_NO_PROPOSAL,
        FDPSCState.COMMIT_SLOW,
        FDPSCState.COMMIT_EXCEPTION,
        FDPSCState.REJECT_QUERY,
    }

    def __init__(self) -> None:
        self.state = FDPSCState.IDLE
        self.episode_id: Optional[str] = None
        self.context_identifier: Optional[str] = None
        self.online_update_count = 0
        self.support_window_count = 0
        self.sleep_entry_count = 0
        self.final_gate_count = 0
        self.final_proposal_count = 0
        self.persistent_commit_count = 0
        self.rollback_count = 0
        self.centered_activated = False
        self.final_proposal: Optional[FinalProposal] = None
        self.query_token_id: Optional[str] = None
        self.last_outcome: Optional[str] = None
        self.last_reason: Optional[str] = None
        self._transition_index = 0
        self.transition_log: List[TransitionRecord] = []

    @property
    def active(self) -> bool:
        return self.state != FDPSCState.IDLE

    def _transition(self, expected: object, new_state: FDPSCState, reason: str) -> None:
        allowed = {expected} if isinstance(expected, FDPSCState) else set(expected)  # type: ignore[arg-type]
        if self.state not in allowed:
            names = ", ".join(sorted(item.value for item in allowed))
            raise StateMachineError(
                f"illegal FD-PSC transition {self.state.value}->{new_state.value}; expected one of {names}"
            )
        old = self.state
        self.state = new_state
        self.transition_log.append(
            TransitionRecord(
                index=self._transition_index,
                episode_id=self.episode_id,
                old_state=old,
                new_state=new_state,
                reason=str(reason),
            )
        )
        self._transition_index += 1

    def begin_episode(self, episode_id: str, context_identifier: str) -> None:
        if not str(episode_id) or not str(context_identifier):
            raise StateMachineError("episode_id and context_identifier must be known before begin_episode")
        if self.state != FDPSCState.IDLE:
            raise StateMachineError(f"cannot begin episode while state is {self.state.value}")
        self.episode_id = str(episode_id)
        self.context_identifier = str(context_identifier)
        self.online_update_count = 0
        self.support_window_count = 0
        self.sleep_entry_count = 0
        self.final_gate_count = 0
        self.final_proposal_count = 0
        self.centered_activated = False
        self.final_proposal = None
        self.query_token_id = None
        self.last_outcome = None
        self.last_reason = None
        self._transition(FDPSCState.IDLE, FDPSCState.EPISODE_PILOT, "begin_episode")

    def note_support_window(self, count: int = 1) -> None:
        if self.state not in (FDPSCState.EPISODE_PILOT, FDPSCState.EPISODE_CENTERED):
            raise StateMachineError("support may only be added during an active episode")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("support window count must be a positive integer")
        self.support_window_count += count

    def note_online_update(self, count: int = 1) -> None:
        if self.state not in (FDPSCState.EPISODE_PILOT, FDPSCState.EPISODE_CENTERED):
            raise StateMachineError("online updates may only occur in Pilot/Centered state")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("online update count must be a positive integer")
        self.online_update_count += count

    def activate_centered(self, reason: str = "triggered_slice") -> None:
        if self.centered_activated:
            raise StateMachineError("Pilot->Centered may happen at most once per episode")
        self._transition(FDPSCState.EPISODE_PILOT, FDPSCState.EPISODE_CENTERED, reason)
        self.centered_activated = True

    def enter_sleep(self) -> None:
        if self.sleep_entry_count:
            raise StateMachineError("sleep may be entered at most once per episode")
        self._transition(
            (FDPSCState.EPISODE_PILOT, FDPSCState.EPISODE_CENTERED),
            FDPSCState.SLEEP_CALIBRATION,
            "episode_returned_normally",
        )
        self.sleep_entry_count = 1

    def finish_without_sleep(self, reason: str) -> None:
        """Close a normal episode that produced no eligible support window.

        This is deliberately *not* represented as SLEEP_CALIBRATION: the
        planner contract says sleep is entered exactly once only after a
        normal return with a non-empty support buffer.  It is also not an
        abort/rollback because the planner itself completed normally.
        """

        self._transition(
            (FDPSCState.EPISODE_PILOT, FDPSCState.EPISODE_CENTERED),
            FDPSCState.IDLE,
            str(reason),
        )
        self.last_outcome = "NO_SLEEP"
        self.last_reason = str(reason)
        self._clear_active_fields()

    def enter_repair(self, reason: str) -> None:
        self._transition(FDPSCState.SLEEP_CALIBRATION, FDPSCState.REPAIR, reason)

    def set_final_proposal(self, proposal: FinalProposal, reason: str = "calibration_selected") -> None:
        if self.final_proposal is not None:
            raise StateMachineError("an episode may select at most one final proposal")
        self._transition(
            (FDPSCState.SLEEP_CALIBRATION, FDPSCState.REPAIR),
            FDPSCState.FINAL_PROPOSAL_READY,
            reason,
        )
        self.final_proposal = proposal
        self.final_proposal_count += 1

    def reject_no_proposal(self, reason: str) -> None:
        if self.final_proposal is not None:
            raise StateMachineError("cannot reject as no-proposal after selecting a proposal")
        self._transition(
            (FDPSCState.SLEEP_CALIBRATION, FDPSCState.REPAIR),
            FDPSCState.REJECT_NO_PROPOSAL,
            reason,
        )
        self.last_outcome = FDPSCState.REJECT_NO_PROPOSAL.value
        self.last_reason = str(reason)

    def begin_final_gate(self, proposal_id: str, query_token_id: str) -> None:
        if self.final_gate_count:
            raise StateMachineError("commit-query final gate may run at most once per episode")
        if self.final_proposal is None or self.final_proposal.proposal_id != str(proposal_id):
            raise StateMachineError("final gate authorization does not match the selected proposal")
        if not str(query_token_id):
            raise StateMachineError("final gate requires a commit-query access token")
        self._transition(FDPSCState.FINAL_PROPOSAL_READY, FDPSCState.FINAL_GATE, "commit_query_consumed")
        self.final_gate_count = 1
        self.query_token_id = str(query_token_id)

    def commit_slow(self, reason: str = "all_gates_passed") -> None:
        if self.final_proposal is None or self.final_proposal.proposal_type != ProposalType.GLOBAL_SLOW:
            raise StateMachineError("COMMIT_SLOW requires a global-slow final proposal")
        self._transition(FDPSCState.FINAL_GATE, FDPSCState.COMMIT_SLOW, reason)
        self.persistent_commit_count += 1
        self.last_outcome = FDPSCState.COMMIT_SLOW.value
        self.last_reason = reason

    def commit_exception(self, reason: str = "all_gates_passed") -> None:
        if self.final_proposal is None or self.final_proposal.proposal_type not in (
            ProposalType.NEW_EXCEPTION,
            ProposalType.REPLACE_EXCEPTION,
        ):
            raise StateMachineError("COMMIT_EXCEPTION requires an exception final proposal")
        self._transition(FDPSCState.FINAL_GATE, FDPSCState.COMMIT_EXCEPTION, reason)
        self.persistent_commit_count += 1
        self.last_outcome = FDPSCState.COMMIT_EXCEPTION.value
        self.last_reason = reason

    def commit_baseline_slow(self, proposal: FinalProposal, reason: str) -> None:
        """Record an explicitly labelled no-gate SVD baseline commit."""

        if proposal.proposal_type != ProposalType.GLOBAL_SLOW:
            raise StateMachineError("baseline slow commit requires a global-slow proposal")
        if self.final_proposal is not None:
            raise StateMachineError("baseline episode already has a proposal")
        self.final_proposal = proposal
        self.final_proposal_count += 1
        self._transition(FDPSCState.SLEEP_CALIBRATION, FDPSCState.COMMIT_SLOW, reason)
        self.persistent_commit_count += 1
        self.last_outcome = FDPSCState.COMMIT_SLOW.value
        self.last_reason = str(reason)

    def reject_query(self, reason: str) -> None:
        self._transition(FDPSCState.FINAL_GATE, FDPSCState.REJECT_QUERY, reason)
        self.last_outcome = FDPSCState.REJECT_QUERY.value
        self.last_reason = str(reason)

    def finish_episode(self) -> None:
        if self.state not in self._TERMINAL:
            raise StateMachineError(f"cannot finish episode from {self.state.value}")
        self._transition(self._TERMINAL, FDPSCState.IDLE, "episodic_cleanup_complete")
        self._clear_active_fields()

    def abort(self, reason: str) -> None:
        if self.state == FDPSCState.IDLE:
            raise StateMachineError("cannot abort without an active episode")
        self._transition(tuple(state for state in FDPSCState if state != FDPSCState.IDLE), FDPSCState.ROLLBACK, reason)
        self.rollback_count += 1
        self.last_outcome = FDPSCState.ROLLBACK.value
        self.last_reason = str(reason)
        self._transition(FDPSCState.ROLLBACK, FDPSCState.IDLE, "rollback_complete")
        self._clear_active_fields()

    def _clear_active_fields(self) -> None:
        self.episode_id = None
        self.context_identifier = None
        # Per-episode counters remain available for post-terminal audit until
        # the next begin_episode call resets them.
        self.final_proposal = None
        self.query_token_id = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "state": self.state.value,
            "episode_id": self.episode_id,
            "context_identifier": self.context_identifier,
            "online_update_count": self.online_update_count,
            "support_window_count": self.support_window_count,
            "sleep_entry_count": self.sleep_entry_count,
            "final_gate_count": self.final_gate_count,
            "final_proposal_count": self.final_proposal_count,
            "persistent_commit_count": self.persistent_commit_count,
            "rollback_count": self.rollback_count,
            "centered_activated": self.centered_activated,
            "final_proposal": copy.deepcopy(self.final_proposal),
            "query_token_id": self.query_token_id,
            "last_outcome": self.last_outcome,
            "last_reason": self.last_reason,
            "transition_index": self._transition_index,
            "transition_log": copy.deepcopy(self.transition_log),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise StateMachineError("unsupported state-machine schema")
        restored_state = FDPSCState(str(state["state"]))
        self.state = restored_state
        self.episode_id = state.get("episode_id")
        self.context_identifier = state.get("context_identifier")
        self.online_update_count = int(state.get("online_update_count", 0))
        self.support_window_count = int(state.get("support_window_count", 0))
        self.sleep_entry_count = int(state.get("sleep_entry_count", 0))
        self.final_gate_count = int(state.get("final_gate_count", 0))
        self.final_proposal_count = int(state.get("final_proposal_count", 0))
        self.persistent_commit_count = int(state.get("persistent_commit_count", 0))
        self.rollback_count = int(state.get("rollback_count", 0))
        self.centered_activated = bool(state.get("centered_activated", False))
        self.final_proposal = copy.deepcopy(state.get("final_proposal"))
        self.query_token_id = state.get("query_token_id")
        self.last_outcome = state.get("last_outcome")
        self.last_reason = state.get("last_reason")
        self._transition_index = int(state.get("transition_index", 0))
        self.transition_log = copy.deepcopy(list(state.get("transition_log", [])))


__all__ = [
    "FDPSCState",
    "FDPSCStateMachine",
    "FinalProposal",
    "ProposalType",
    "StateMachineError",
    "TransitionRecord",
]
