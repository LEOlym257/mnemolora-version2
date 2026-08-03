"""Isolated Gate-7 planning canaries for FD-PSC.

The canary layer is deliberately independent from external calibration and
commit-query data.  Its callback receives only a fixed canary scenario, a
fixed seed, and a disposable clone of either the pre-commit or candidate
persistent state.  This module never receives an ``ExternalDataRegistry`` (or
any commit-query token), and it verifies that the caller-owned states are
bitwise/content unchanged after evaluation.

Canary manifest schema version 1::

    {
      "schema_version": 1,
      "base_checkpoint_hash": "<sha256>",
      "preprocess_hash": "<sha256>",
      "environment_id": "robot-eval-v1",
      "deterministic_reset": true,
      "scenarios": [
        {
          "scenario_id": "pick-01",
          "context_identifier": "pick",
          "seed": 17,
          "payload": {"task": "pick", "reset_state": 3},
          "content_hash": "<sha256-of-payload>",
          "metadata": {}
        }
      ]
    }

Instead of inline ``payload``, a scenario may name a JSON ``payload_path`` and
its file ``sha256``.  File payloads are read and retained as an in-memory JSON
snapshot at manifest load time, so a later file replacement cannot silently
change the rollout request.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .checkpoint import CheckpointValidationError, state_content_hash
from .external_data import canonical_json_hash, sha256_file


class CanaryError(RuntimeError):
    """Base class for Gate-7 canary errors."""


class CanaryManifestError(CanaryError):
    """The immutable canary manifest or a referenced payload is invalid."""


class CanaryUnavailableError(CanaryError):
    """A rollout worker cannot provide the requested deterministic rollout."""


class CanaryEvaluationError(CanaryError):
    """A rollout result or evaluator contract is invalid."""


class CanaryPhase(str, enum.Enum):
    PRE_COMMIT = "pre_commit"
    POST_COMMIT = "post_commit"


class CanaryStatus(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    UNRUN = "unrun"
    NOT_APPLICABLE = "not_applicable"


def _require_sha256(name: str, value: Any) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CanaryManifestError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _read_json(path: Path, *, label: str) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryManifestError(f"cannot read {label} JSON {path}: {exc}") from exc


def _assert_jsonable(
    value: Any,
    *,
    label: str,
    error_type: type = CanaryEvaluationError,
) -> Any:
    """Return a detached JSON value and reject NaN/Inf or opaque objects."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{label} must be finite JSON data: {exc}") from exc


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _mutable_object_ids(value: Any, seen: Optional[set] = None) -> set:
    """Collect mutable descendants to reject shallow/aliasing clone factories."""

    visited = seen if seen is not None else set()
    identity = id(value)
    if identity in visited:
        return set()
    visited.add(identity)
    result = set()
    if isinstance(value, Mapping):
        result.add(identity)
        for key, item in value.items():
            result.update(_mutable_object_ids(key, visited))
            result.update(_mutable_object_ids(item, visited))
    elif isinstance(value, (list, set, bytearray)):
        result.add(identity)
        for item in value:
            result.update(_mutable_object_ids(item, visited))
    elif hasattr(value, "data_ptr") and callable(getattr(value, "data_ptr", None)):
        # Tensor-like values are mutable even when contained by an immutable tuple.
        result.add(identity)
    elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, int, float, bool, type(None))):
        result.add(identity)
        result.update(_mutable_object_ids(vars(value), visited))
    elif isinstance(value, tuple):
        for item in value:
            result.update(_mutable_object_ids(item, visited))
    return result


def _reject_commit_query_references(value: Any, *, path: str = "manifest") -> None:
    """Keep Gate-7 inputs structurally separate from commit-query material."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if "commit_query" in normalized or normalized in {"query_token", "commitquery"}:
                raise CanaryManifestError(
                    f"{path}.{key} is forbidden: canary manifests cannot reference commit-query data"
                )
            _reject_commit_query_references(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_commit_query_references(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class CanaryScenario:
    scenario_id: str
    context_identifier: str
    seed: int
    content_hash: str
    payload: Any = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def detached_payload(self) -> Any:
        return copy.deepcopy(self.payload)


@dataclass(frozen=True)
class CanaryManifest:
    path: Path
    manifest_hash: str
    file_hash: str
    base_checkpoint_hash: str
    preprocess_hash: str
    environment_id: str
    deterministic_reset: bool
    scenarios: Tuple[CanaryScenario, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION = 1

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_base_checkpoint_hash: Optional[str] = None,
        expected_preprocess_hash: Optional[str] = None,
        expected_manifest_hash: Optional[str] = None,
        verify_checksums: bool = True,
    ) -> "CanaryManifest":
        manifest_path = Path(path).expanduser().resolve()
        raw = _read_json(manifest_path, label="canary manifest")
        if not isinstance(raw, Mapping):
            raise CanaryManifestError("canary manifest must be a JSON object")
        _reject_commit_query_references(raw)
        if int(raw.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise CanaryManifestError("unsupported canary manifest schema_version")

        base_hash = _require_sha256("base_checkpoint_hash", raw.get("base_checkpoint_hash", ""))
        preprocess_hash = _require_sha256("preprocess_hash", raw.get("preprocess_hash", ""))
        if expected_base_checkpoint_hash is not None:
            expected_base = _require_sha256(
                "expected_base_checkpoint_hash", expected_base_checkpoint_hash
            )
            if not hmac.compare_digest(base_hash, expected_base):
                raise CanaryManifestError("canary manifest base checkpoint hash mismatch")
        if expected_preprocess_hash is not None:
            expected_preprocess = _require_sha256(
                "expected_preprocess_hash", expected_preprocess_hash
            )
            if not hmac.compare_digest(preprocess_hash, expected_preprocess):
                raise CanaryManifestError("canary manifest preprocess hash mismatch")

        environment_id = str(raw.get("environment_id", "")).strip()
        if not environment_id:
            raise CanaryManifestError("environment_id must be non-empty")
        deterministic_reset = raw.get("deterministic_reset")
        if not isinstance(deterministic_reset, bool):
            raise CanaryManifestError("deterministic_reset must be a boolean")

        raw_scenarios = raw.get("scenarios")
        if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)):
            raise CanaryManifestError("manifest.scenarios must be a JSON array")
        if not raw_scenarios:
            raise CanaryManifestError("manifest.scenarios must be non-empty")

        scenarios: List[CanaryScenario] = []
        scenario_ids = set()
        for index, raw_scenario in enumerate(raw_scenarios):
            label = f"scenarios[{index}]"
            if not isinstance(raw_scenario, Mapping):
                raise CanaryManifestError(f"{label} must be an object")
            scenario_id = str(raw_scenario.get("scenario_id", "")).strip()
            context = str(raw_scenario.get("context_identifier", "")).strip()
            if not scenario_id or not context:
                raise CanaryManifestError(
                    f"{label} requires non-empty scenario_id and context_identifier"
                )
            if scenario_id in scenario_ids:
                raise CanaryManifestError(f"duplicate scenario_id: {scenario_id}")
            scenario_ids.add(scenario_id)
            seed = raw_scenario.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise CanaryManifestError(f"{label}.seed must be a non-negative integer")

            has_inline = "payload" in raw_scenario
            has_path = raw_scenario.get("payload_path") is not None
            if has_inline == has_path:
                raise CanaryManifestError(
                    f"{label} must provide exactly one of payload or payload_path"
                )
            if has_inline:
                payload = _assert_jsonable(
                    raw_scenario["payload"],
                    label=f"{label}.payload",
                    error_type=CanaryManifestError,
                )
                actual_content_hash = canonical_json_hash(payload)
                expected_content_hash = _require_sha256(
                    f"{label}.content_hash", raw_scenario.get("content_hash", "")
                )
                if verify_checksums and not hmac.compare_digest(
                    actual_content_hash, expected_content_hash
                ):
                    raise CanaryManifestError(
                        f"{label} payload checksum mismatch: expected "
                        f"{expected_content_hash}, got {actual_content_hash}"
                    )
                content_hash = expected_content_hash
            else:
                payload_path = Path(str(raw_scenario["payload_path"])).expanduser()
                if not payload_path.is_absolute():
                    payload_path = manifest_path.parent / payload_path
                payload_path = payload_path.resolve()
                expected_file_hash = _require_sha256(
                    f"{label}.sha256", raw_scenario.get("sha256", "")
                )
                if not payload_path.is_file():
                    raise CanaryManifestError(f"{label} payload file is not readable: {payload_path}")
                actual_file_hash = sha256_file(payload_path)
                if verify_checksums and not hmac.compare_digest(actual_file_hash, expected_file_hash):
                    raise CanaryManifestError(
                        f"{label} file checksum mismatch: expected "
                        f"{expected_file_hash}, got {actual_file_hash}"
                    )
                payload = _read_json(payload_path, label=f"{label} payload")
                payload = _assert_jsonable(
                    payload,
                    label=f"{label}.payload",
                    error_type=CanaryManifestError,
                )
                # The file digest is the fixed identity for an external scenario.
                content_hash = expected_file_hash

            metadata = _assert_jsonable(
                raw_scenario.get("metadata", {}),
                label=f"{label}.metadata",
                error_type=CanaryManifestError,
            )
            if not isinstance(metadata, Mapping):
                raise CanaryManifestError(f"{label}.metadata must be an object")
            scenarios.append(
                CanaryScenario(
                    scenario_id=scenario_id,
                    context_identifier=context,
                    seed=int(seed),
                    content_hash=content_hash,
                    payload=payload,
                    metadata=dict(metadata),
                )
            )

        metadata = _assert_jsonable(
            raw.get("metadata", {}),
            label="manifest.metadata",
            error_type=CanaryManifestError,
        )
        if not isinstance(metadata, Mapping):
            raise CanaryManifestError("manifest.metadata must be an object")
        manifest_hash = canonical_json_hash(raw)
        if expected_manifest_hash is not None:
            expected = _require_sha256("expected_manifest_hash", expected_manifest_hash)
            if not hmac.compare_digest(manifest_hash, expected):
                raise CanaryManifestError("canary manifest identity hash mismatch")
        return cls(
            path=manifest_path,
            manifest_hash=manifest_hash,
            file_hash=sha256_file(manifest_path),
            base_checkpoint_hash=base_hash,
            preprocess_hash=preprocess_hash,
            environment_id=environment_id,
            deterministic_reset=deterministic_reset,
            scenarios=tuple(scenarios),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class CanaryTriggerDecision:
    should_run: bool
    phase: CanaryPhase
    episode_count: int
    commit_sequence: int
    reasons: Tuple[str, ...] = ()
    rank_expanded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_run": self.should_run,
            "phase": self.phase.value,
            "episode_count": self.episode_count,
            "commit_sequence": self.commit_sequence,
            "reasons": list(self.reasons),
            "rank_expanded": self.rank_expanded,
        }


class CanaryScheduler:
    """Decide pre-commit high-risk and periodic post-commit canaries."""

    def __init__(
        self,
        *,
        enabled: bool,
        every_episodes: int,
        high_risk_rank_expansion: bool = True,
    ) -> None:
        if int(every_episodes) <= 0:
            raise ValueError("every_episodes must be positive")
        self.enabled = bool(enabled)
        self.every_episodes = int(every_episodes)
        self.high_risk_rank_expansion = bool(high_risk_rank_expansion)

    @classmethod
    def from_config(cls, config: Any) -> "CanaryScheduler":
        return cls(
            enabled=bool(config.enabled),
            every_episodes=int(config.every_episodes),
            high_risk_rank_expansion=bool(config.high_risk_rank_expansion),
        )

    @staticmethod
    def rank_expanded(
        before_ranks: Optional[Mapping[str, int]],
        candidate_ranks: Optional[Mapping[str, int]],
    ) -> bool:
        before = {str(key): int(value) for key, value in dict(before_ranks or {}).items()}
        candidate = {str(key): int(value) for key, value in dict(candidate_ranks or {}).items()}
        if any(value < 0 for value in tuple(before.values()) + tuple(candidate.values())):
            raise ValueError("canary ranks must be non-negative")
        return any(candidate.get(key, 0) > before.get(key, 0) for key in set(before) | set(candidate))

    def decide(
        self,
        *,
        phase: CanaryPhase,
        episode_count: int,
        commit_sequence: int,
        before_ranks: Optional[Mapping[str, int]] = None,
        candidate_ranks: Optional[Mapping[str, int]] = None,
        high_risk_commit: bool = False,
        exception_merge: bool = False,
    ) -> CanaryTriggerDecision:
        phase = CanaryPhase(phase)
        episode_count = int(episode_count)
        commit_sequence = int(commit_sequence)
        if episode_count < 0 or commit_sequence < 0:
            raise ValueError("episode_count and commit_sequence must be non-negative")
        expansion = self.rank_expanded(before_ranks, candidate_ranks)
        reasons: List[str] = []
        if self.enabled:
            if phase is CanaryPhase.PRE_COMMIT:
                if high_risk_commit:
                    reasons.append("high_risk_commit")
                if exception_merge:
                    reasons.append("exception_merge")
                if self.high_risk_rank_expansion and expansion:
                    reasons.append("rank_expansion")
            elif episode_count > 0 and episode_count % self.every_episodes == 0:
                reasons.append("periodic")
        return CanaryTriggerDecision(
            should_run=bool(reasons),
            phase=phase,
            episode_count=episode_count,
            commit_sequence=commit_sequence,
            reasons=tuple(reasons),
            rank_expanded=expansion,
        )


@dataclass(frozen=True)
class CanaryRolloutRequest:
    manifest_hash: str
    environment_id: str
    scenario_id: str
    context_identifier: str
    seed: int
    pair_index: int
    pair_id: str
    state_label: str
    state: Any = field(repr=False)
    payload: Any = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requires_deterministic_reset: bool = True


@dataclass(frozen=True)
class CanaryRolloutOutcome:
    success: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "metrics": _assert_jsonable(dict(self.metrics), label="rollout metrics"),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CanaryPairResult:
    pair_index: int
    pair_id: str
    scenario_id: str
    context_identifier: str
    seed: int
    before: CanaryRolloutOutcome
    candidate: CanaryRolloutOutcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "context_identifier": self.context_identifier,
            "seed": self.seed,
            "before": self.before.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


@dataclass(frozen=True)
class CanaryResult:
    status: CanaryStatus
    reason: str
    trigger: CanaryTriggerDecision
    manifest_hash: Optional[str]
    manifest_file_hash: Optional[str]
    environment_id: Optional[str]
    requested_rollouts_per_state: int
    completed_pairs: int
    before_successes: Optional[int]
    candidate_successes: Optional[int]
    before_success_rate: Optional[float]
    candidate_success_rate: Optional[float]
    pairs: Tuple[CanaryPairResult, ...] = ()
    limitations: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is CanaryStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "status": self.status.value,
            "reason": self.reason,
            "trigger": self.trigger.to_dict(),
            "manifest_hash": self.manifest_hash,
            "manifest_file_hash": self.manifest_file_hash,
            "environment_id": self.environment_id,
            "requested_rollouts_per_state": self.requested_rollouts_per_state,
            "completed_pairs": self.completed_pairs,
            "before_successes": self.before_successes,
            "candidate_successes": self.candidate_successes,
            "before_success_rate": self.before_success_rate,
            "candidate_success_rate": self.candidate_success_rate,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "limitations": list(self.limitations),
        }
        return _assert_jsonable(value, label="canary result")


RolloutEvaluator = Callable[[CanaryRolloutRequest], Any]
CloneState = Callable[[Any], Any]
StateHasher = Callable[[Any], str]
DecisionFunction = Callable[
    [Tuple[CanaryRolloutOutcome, ...], Tuple[CanaryRolloutOutcome, ...]],
    Tuple[bool, str],
]


def _default_decision(
    before: Tuple[CanaryRolloutOutcome, ...],
    candidate: Tuple[CanaryRolloutOutcome, ...],
) -> Tuple[bool, str]:
    before_successes = sum(int(item.success) for item in before)
    candidate_successes = sum(int(item.success) for item in candidate)
    if candidate_successes < before_successes:
        return False, "candidate planning success regressed against paired before state"
    return True, "candidate planning success did not regress against paired before state"


class CanaryRunner:
    """Run paired rollouts on disposable state clones without touching live state."""

    def __init__(
        self,
        manifest: CanaryManifest,
        *,
        rollout_count: int,
        evaluator: Optional[RolloutEvaluator],
        unavailable_policy: str = "report_unrun",
        clone_state: CloneState = copy.deepcopy,
        state_hasher: StateHasher = state_content_hash,
        decision_function: DecisionFunction = _default_decision,
    ) -> None:
        if int(rollout_count) <= 0:
            raise ValueError("rollout_count must be positive")
        if int(rollout_count) > len(manifest.scenarios):
            raise CanaryManifestError(
                "canary manifest has fewer fixed scenarios than rollout_count"
            )
        if unavailable_policy not in {"report_unrun", "error"}:
            raise ValueError("unavailable_policy must be 'report_unrun' or 'error'")
        self.manifest = manifest
        self.rollout_count = int(rollout_count)
        self.evaluator = evaluator
        self.unavailable_policy = unavailable_policy
        self.clone_state = clone_state
        self.state_hasher = state_hasher
        self.decision_function = decision_function

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        expected_base_checkpoint_hash: str,
        evaluator: Optional[RolloutEvaluator],
        clone_state: CloneState = copy.deepcopy,
        state_hasher: StateHasher = state_content_hash,
        decision_function: DecisionFunction = _default_decision,
    ) -> "CanaryRunner":
        if not config.manifest_path:
            raise CanaryManifestError("enabled canary requires manifest_path")
        manifest = CanaryManifest.load(
            Path(config.manifest_path),
            expected_base_checkpoint_hash=expected_base_checkpoint_hash,
        )
        return cls(
            manifest,
            rollout_count=int(config.rollout_count),
            evaluator=evaluator,
            unavailable_policy=str(config.unavailable_policy),
            clone_state=clone_state,
            state_hasher=state_hasher,
            decision_function=decision_function,
        )

    def _unrun(
        self,
        trigger: CanaryTriggerDecision,
        reason: str,
        *,
        limitation: Optional[str] = None,
    ) -> CanaryResult:
        if self.unavailable_policy == "error":
            raise CanaryUnavailableError(reason)
        limitations = (limitation or reason,)
        return CanaryResult(
            status=CanaryStatus.UNRUN,
            reason=reason,
            trigger=trigger,
            manifest_hash=self.manifest.manifest_hash,
            manifest_file_hash=self.manifest.file_hash,
            environment_id=self.manifest.environment_id,
            requested_rollouts_per_state=self.rollout_count,
            completed_pairs=0,
            before_successes=None,
            candidate_successes=None,
            before_success_rate=None,
            candidate_success_rate=None,
            pairs=(),
            limitations=limitations,
        )

    @staticmethod
    def _normalize_outcome(value: Any) -> CanaryRolloutOutcome:
        if isinstance(value, CanaryRolloutOutcome):
            outcome = value
        elif isinstance(value, bool):
            outcome = CanaryRolloutOutcome(success=value)
        elif isinstance(value, Mapping):
            if not isinstance(value.get("success"), bool):
                raise CanaryEvaluationError("rollout result requires boolean success")
            metrics = value.get("metrics", {})
            if not isinstance(metrics, Mapping):
                raise CanaryEvaluationError("rollout result metrics must be an object")
            outcome = CanaryRolloutOutcome(
                success=bool(value["success"]),
                metrics=dict(metrics),
                reason=str(value.get("reason", "")),
            )
        else:
            raise CanaryEvaluationError(
                "rollout evaluator must return bool, mapping, or CanaryRolloutOutcome"
            )
        # Validate now rather than discovering non-finite metrics during logging.
        outcome.to_dict()
        return outcome

    def _request(
        self,
        *,
        scenario: CanaryScenario,
        pair_index: int,
        state_label: str,
        state: Any,
    ) -> CanaryRolloutRequest:
        pair_id = hashlib.sha256(
            (
                f"{self.manifest.manifest_hash}\0{pair_index}\0{scenario.scenario_id}"
                f"\0{scenario.seed}"
            ).encode("utf-8")
        ).hexdigest()
        cloned_state = self.clone_state(state)
        shared_mutable = _mutable_object_ids(state) & _mutable_object_ids(cloned_state)
        if shared_mutable:
            raise CanaryEvaluationError(
                "clone_state returned a clone sharing mutable objects with caller-owned state"
            )
        return CanaryRolloutRequest(
            manifest_hash=self.manifest.manifest_hash,
            environment_id=self.manifest.environment_id,
            scenario_id=scenario.scenario_id,
            context_identifier=scenario.context_identifier,
            seed=scenario.seed,
            pair_index=pair_index,
            pair_id=pair_id,
            state_label=state_label,
            state=cloned_state,
            payload=scenario.detached_payload(),
            metadata=copy.deepcopy(dict(scenario.metadata)),
            requires_deterministic_reset=True,
        )

    def run(
        self,
        trigger: CanaryTriggerDecision,
        *,
        before_state: Any,
        candidate_state: Any,
    ) -> CanaryResult:
        if not trigger.should_run:
            return CanaryResult(
                status=CanaryStatus.NOT_APPLICABLE,
                reason="canary schedule did not trigger",
                trigger=trigger,
                manifest_hash=self.manifest.manifest_hash,
                manifest_file_hash=self.manifest.file_hash,
                environment_id=self.manifest.environment_id,
                requested_rollouts_per_state=0,
                completed_pairs=0,
                before_successes=None,
                candidate_successes=None,
                before_success_rate=None,
                candidate_success_rate=None,
            )
        if not self.manifest.deterministic_reset:
            return self._unrun(
                trigger,
                "canary environment does not support deterministic reset",
                limitation="manifest declares deterministic_reset=false",
            )
        if self.evaluator is None:
            return self._unrun(trigger, "canary rollout evaluator is unavailable")

        try:
            before_identity = self.state_hasher(before_state)
            candidate_identity = self.state_hasher(candidate_state)
        except (CheckpointValidationError, TypeError, ValueError) as exc:
            raise CanaryEvaluationError(f"cannot fingerprint caller-owned canary state: {exc}") from exc

        pairs: List[CanaryPairResult] = []
        try:
            for pair_index, scenario in enumerate(self.manifest.scenarios[: self.rollout_count]):
                before_request = self._request(
                    scenario=scenario,
                    pair_index=pair_index,
                    state_label="before",
                    state=before_state,
                )
                candidate_request = self._request(
                    scenario=scenario,
                    pair_index=pair_index,
                    state_label="candidate",
                    state=candidate_state,
                )
                before_outcome = self._normalize_outcome(self.evaluator(before_request))
                candidate_outcome = self._normalize_outcome(self.evaluator(candidate_request))
                pairs.append(
                    CanaryPairResult(
                        pair_index=pair_index,
                        pair_id=before_request.pair_id,
                        scenario_id=scenario.scenario_id,
                        context_identifier=scenario.context_identifier,
                        seed=scenario.seed,
                        before=before_outcome,
                        candidate=candidate_outcome,
                    )
                )
        except CanaryUnavailableError as exc:
            return self._unrun(trigger, str(exc))
        except CanaryEvaluationError:
            raise
        except Exception as exc:
            raise CanaryEvaluationError(f"canary rollout evaluator failed: {exc}") from exc
        finally:
            try:
                before_after = self.state_hasher(before_state)
                candidate_after = self.state_hasher(candidate_state)
            except Exception as exc:
                raise CanaryEvaluationError(
                    f"cannot verify caller-owned canary state after rollout: {exc}"
                ) from exc
            if before_after != before_identity or candidate_after != candidate_identity:
                raise CanaryEvaluationError(
                    "canary evaluator changed caller-owned model or persistent state"
                )

        before_outcomes = tuple(pair.before for pair in pairs)
        candidate_outcomes = tuple(pair.candidate for pair in pairs)
        try:
            passed, reason = self.decision_function(before_outcomes, candidate_outcomes)
        except Exception as exc:
            raise CanaryEvaluationError(f"canary decision function failed: {exc}") from exc
        if not isinstance(passed, bool) or not str(reason):
            raise CanaryEvaluationError("canary decision function must return (bool, non-empty reason)")

        before_successes = sum(int(item.success) for item in before_outcomes)
        candidate_successes = sum(int(item.success) for item in candidate_outcomes)
        count = len(pairs)
        return CanaryResult(
            status=CanaryStatus.PASS if passed else CanaryStatus.FAIL,
            reason=str(reason),
            trigger=trigger,
            manifest_hash=self.manifest.manifest_hash,
            manifest_file_hash=self.manifest.file_hash,
            environment_id=self.manifest.environment_id,
            requested_rollouts_per_state=self.rollout_count,
            completed_pairs=count,
            before_successes=before_successes,
            candidate_successes=candidate_successes,
            before_success_rate=before_successes / count,
            candidate_success_rate=candidate_successes / count,
            pairs=tuple(pairs),
        )


__all__ = [
    "CanaryError",
    "CanaryEvaluationError",
    "CanaryManifest",
    "CanaryManifestError",
    "CanaryPairResult",
    "CanaryPhase",
    "CanaryResult",
    "CanaryRolloutOutcome",
    "CanaryRolloutRequest",
    "CanaryRunner",
    "CanaryScenario",
    "CanaryScheduler",
    "CanaryStatus",
    "CanaryTriggerDecision",
    "CanaryUnavailableError",
]
