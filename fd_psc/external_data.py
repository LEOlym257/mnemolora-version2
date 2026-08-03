"""Versioned external split registry with leakage and query-access guards.

Manifest schema version 1::

    {
      "schema_version": 1,
      "base_checkpoint_hash": "<sha256>",
      "preprocess_hash": "<sha256>",
      "latent_adapter_schema": "name/version",
      "splits": {
        "calibration": {"path": "calibration.json", "sha256": "<sha256>"},
        ...
      },
      "contexts": {                         # optional integrity index
        "context-a": {"calibration": ["record-id", ...], ...}
      }
    }

Each split file is either a JSON list or ``{"schema_version": 1,
"records": [...]}``.  A record has stable trajectory/transition/frame IDs and
a content hash.  Those four namespaces are audited pairwise across every
external split.  Commit-query records cannot be obtained through a general
getter: a proposal-bound, single-use token is mandatory.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXTERNAL_SPLITS: Tuple[str, ...] = (
    "calibration",
    "commit_query",
    "plasticity_support",
    "plasticity_query",
    "report_test",
    "anchor",
)


class ExternalDataError(RuntimeError):
    pass


class ManifestSchemaError(ExternalDataError):
    pass


class DataLeakageError(ExternalDataError):
    pass


class CommitQueryAccessError(ExternalDataError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestSchemaError(f"payload is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def external_payload_file_hash(path: Path) -> str:
    """Load an external payload file and return its execution-time hash.

    JSON payload identities are defined by canonical decoded content, while
    PyTorch payload identities are defined by the exact file bytes.  Loading
    the value here is intentional: the startup leakage audit must also prove
    that every referenced payload is readable and has the mapping schema that
    the planner will consume later.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ManifestSchemaError(f"external payload file is not readable: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestSchemaError(f"cannot read external payload JSON {resolved}: {exc}") from exc
        actual = canonical_json_hash(value)
    elif suffix in {".pt", ".pth"}:
        try:
            import torch

            try:
                value = torch.load(resolved, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(resolved, map_location="cpu")
        except Exception as exc:
            raise ManifestSchemaError(f"cannot read external payload {resolved}: {exc}") from exc
        actual = sha256_file(resolved)
    else:
        raise ManifestSchemaError(
            f"unsupported external payload format {resolved.suffix!r}: {resolved}"
        )
    if not isinstance(value, Mapping):
        raise ManifestSchemaError(f"external payload file must contain a mapping: {resolved}")
    return actual


def _require_sha256(name: str, value: Any) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ManifestSchemaError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _stable_ids(name: str, value: Any, *, nonempty: bool = True) -> Tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(str(item) for item in value)
    else:
        raise ManifestSchemaError(f"{name} must be a string or sequence of strings")
    if nonempty and (not values or any(not item for item in values)):
        raise ManifestSchemaError(f"{name} must contain stable non-empty IDs")
    if len(values) != len(set(values)):
        raise ManifestSchemaError(f"{name} contains duplicate IDs")
    return values


@dataclass(frozen=True)
class DataIdentity:
    """Identity fields used for split and online-support leakage auditing."""

    record_id: str
    context_identifier: str
    trajectory_ids: Tuple[str, ...]
    transition_ids: Tuple[str, ...]
    frame_ids: Tuple[str, ...]
    content_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, split_name: str = "support") -> "DataIdentity":
        record_id = str(value.get("record_id", ""))
        context_identifier = str(value.get("context_identifier", ""))
        if not record_id or not context_identifier:
            raise ManifestSchemaError(f"{split_name} record requires record_id and context_identifier")
        trajectory_raw = value.get("trajectory_ids", value.get("trajectory_id"))
        if trajectory_raw is None:
            raise ManifestSchemaError(f"{split_name}/{record_id} is missing trajectory_id(s)")
        transitions = value.get("transition_ids")
        frames = value.get("frame_ids")
        if transitions is None or frames is None:
            raise ManifestSchemaError(
                f"{split_name}/{record_id} must provide transition_ids and frame_ids for leakage audit"
            )
        payload = value.get("payload", None)
        supplied_hash = value.get("content_hash", value.get("content_sha256"))
        if supplied_hash is None:
            if payload is None:
                raise ManifestSchemaError(f"{split_name}/{record_id} requires content_hash")
            supplied_hash = canonical_json_hash(payload)
        content_hash = _require_sha256(f"{split_name}/{record_id}.content_hash", supplied_hash)
        if payload is not None:
            actual = canonical_json_hash(payload)
            if not hmac.compare_digest(actual, content_hash):
                raise ManifestSchemaError(
                    f"{split_name}/{record_id} payload hash mismatch: expected {content_hash}, got {actual}"
                )
        return cls(
            record_id=record_id,
            context_identifier=context_identifier,
            trajectory_ids=_stable_ids("trajectory_ids", trajectory_raw),
            transition_ids=_stable_ids("transition_ids", transitions),
            frame_ids=_stable_ids("frame_ids", frames),
            content_hash=content_hash,
        )


@dataclass(frozen=True)
class ExternalRecord:
    identity: DataIdentity
    split_name: str
    payload: Any = None
    payload_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return self.identity.record_id

    @property
    def context_identifier(self) -> str:
        return self.identity.context_identifier

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], split_name: str) -> "ExternalRecord":
        has_inline_payload = value.get("payload") is not None
        has_payload_path = value.get("payload_path") is not None
        if has_inline_payload == has_payload_path:
            raise ManifestSchemaError(
                f"{split_name} record must provide exactly one of payload or payload_path"
            )
        if has_inline_payload and not isinstance(value["payload"], Mapping):
            raise ManifestSchemaError(f"{split_name} record payload must be a mapping")
        if has_payload_path and not str(value["payload_path"]):
            raise ManifestSchemaError(f"{split_name} record payload_path must be non-empty")
        return cls(
            identity=DataIdentity.from_mapping(value, split_name=split_name),
            split_name=split_name,
            payload=copy.deepcopy(value.get("payload")),
            payload_path=str(value["payload_path"]) if value.get("payload_path") is not None else None,
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
        )


class _IdentityIndex:
    def __init__(self) -> None:
        self.trajectory_ids: Dict[str, str] = {}
        self.transition_ids: Dict[str, str] = {}
        self.frame_ids: Dict[str, str] = {}
        self.content_hashes: Dict[str, str] = {}

    def add(self, identity: DataIdentity, owner: str, *, allow_same_owner: bool = False) -> None:
        collisions: List[str] = []
        fields = (
            ("trajectory", identity.trajectory_ids, self.trajectory_ids),
            ("transition", identity.transition_ids, self.transition_ids),
            ("frame", identity.frame_ids, self.frame_ids),
            ("content_hash", (identity.content_hash,), self.content_hashes),
        )
        for kind, values, index in fields:
            for item in values:
                previous = index.get(item)
                if previous is not None and (not allow_same_owner or previous != owner):
                    collisions.append(f"{kind}={item!r} already owned by {previous}")
        if collisions:
            raise DataLeakageError(f"data leakage for {owner}: " + "; ".join(collisions))
        for _, values, index in fields:
            for item in values:
                index[item] = owner

    def check_against(self, identity: DataIdentity, owner: str) -> None:
        # Use a throw-away copy so the external index is never mutated by a
        # failed or successful online support audit.
        clone = copy.deepcopy(self)
        clone.add(identity, owner)


@dataclass(frozen=True)
class CommitQueryToken:
    token_id: str
    episode_id: str
    proposal_id: str
    manifest_hash: str


class ExternalDataRegistry:
    """Load fixed splits, audit isolation, and guard commit-query access."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        manifest_path: Path,
        *,
        verify_checksums: bool = True,
        expected_base_checkpoint_hash: Optional[str] = None,
        expected_preprocess_hash: Optional[str] = None,
        require_all_splits: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.verify_checksums = bool(verify_checksums)
        self._manifest_raw = self._read_json(self.manifest_path)
        if not isinstance(self._manifest_raw, Mapping):
            raise ManifestSchemaError("external manifest must be a JSON object")
        if int(self._manifest_raw.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ManifestSchemaError("unsupported external manifest schema_version")
        declared_manifest_hash = self._manifest_raw.get("manifest_content_hash")
        if declared_manifest_hash is None:
            self.manifest_hash = canonical_json_hash(self._manifest_raw)
        else:
            expected_manifest_hash = _require_sha256(
                "manifest_content_hash", declared_manifest_hash
            )
            hash_payload = dict(self._manifest_raw)
            del hash_payload["manifest_content_hash"]
            actual_manifest_hash = canonical_json_hash(hash_payload)
            if not hmac.compare_digest(expected_manifest_hash, actual_manifest_hash):
                raise ManifestSchemaError(
                    "external manifest content hash mismatch: "
                    f"expected {expected_manifest_hash}, got {actual_manifest_hash}"
                )
            self.manifest_hash = actual_manifest_hash
        self.base_checkpoint_hash = _require_sha256(
            "base_checkpoint_hash", self._manifest_raw.get("base_checkpoint_hash", "")
        )
        self.preprocess_hash = _require_sha256(
            "preprocess_hash", self._manifest_raw.get("preprocess_hash", "")
        )
        self.latent_adapter_schema = str(self._manifest_raw.get("latent_adapter_schema", ""))
        if not self.latent_adapter_schema:
            raise ManifestSchemaError("latent_adapter_schema must be non-empty")
        if expected_base_checkpoint_hash is not None and not hmac.compare_digest(
            self.base_checkpoint_hash, _require_sha256("expected_base_checkpoint_hash", expected_base_checkpoint_hash)
        ):
            raise ManifestSchemaError("external manifest base checkpoint hash mismatch")
        if expected_preprocess_hash is not None and not hmac.compare_digest(
            self.preprocess_hash,
            _require_sha256("expected_preprocess_hash", expected_preprocess_hash),
        ):
            raise ManifestSchemaError("external manifest preprocess hash mismatch")
        split_specs = self._manifest_raw.get("splits")
        if not isinstance(split_specs, Mapping):
            raise ManifestSchemaError("manifest.splits must be an object")
        missing = sorted(set(EXTERNAL_SPLITS) - set(split_specs))
        if require_all_splits and missing:
            raise ManifestSchemaError(f"manifest is missing required splits: {missing}")
        unknown = sorted(set(split_specs) - set(EXTERNAL_SPLITS))
        if unknown:
            raise ManifestSchemaError(f"manifest has unknown splits: {unknown}")

        self._records: Dict[str, Tuple[ExternalRecord, ...]] = {}
        self._by_context: Dict[str, Dict[str, Tuple[ExternalRecord, ...]]] = {}
        self._external_index = _IdentityIndex()
        for split_name in EXTERNAL_SPLITS:
            if split_name not in split_specs:
                self._records[split_name] = ()
                self._by_context[split_name] = {}
                continue
            records = self._load_split(split_name, split_specs[split_name])
            self._records[split_name] = records
            grouped: Dict[str, List[ExternalRecord]] = {}
            for record in records:
                self._external_index.add(record.identity, f"{split_name}/{record.record_id}")
                grouped.setdefault(record.context_identifier, []).append(record)
            self._by_context[split_name] = {
                context: tuple(sorted(items, key=lambda item: item.record_id))
                for context, items in sorted(grouped.items())
            }
        self._validate_context_index(self._manifest_raw.get("contexts"))
        self.episode_contexts: Mapping[str, str] = MappingProxyType(
            self._validate_episode_contexts(self._manifest_raw.get("episode_contexts"))
        )

        self._active_episode_id: Optional[str] = None
        self._active_context: Optional[str] = None
        self._support_sealed = False
        self._support_records: Dict[str, DataIdentity] = {}
        self._query_issued: Dict[str, str] = {}
        self._query_consumed: Dict[str, str] = {}

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestSchemaError(f"cannot read JSON {path}: {exc}") from exc

    def _load_split(self, split_name: str, spec: Any) -> Tuple[ExternalRecord, ...]:
        if not isinstance(spec, Mapping):
            raise ManifestSchemaError(f"manifest.splits.{split_name} must be an object")
        if "records" in spec:
            raw_records = spec["records"]
        else:
            raw_path = spec.get("path")
            if not raw_path:
                raise ManifestSchemaError(f"split {split_name} requires path or inline records")
            path = Path(str(raw_path)).expanduser()
            path = (path if path.is_absolute() else self.manifest_path.parent / path).resolve()
            if not path.is_file():
                raise ManifestSchemaError(f"split file is not readable: {path}")
            expected = spec.get("sha256")
            if self.verify_checksums:
                expected_hash = _require_sha256(f"splits.{split_name}.sha256", expected or "")
                actual = sha256_file(path)
                if not hmac.compare_digest(expected_hash, actual):
                    raise ManifestSchemaError(
                        f"split {split_name} checksum mismatch: expected {expected_hash}, got {actual}"
                    )
            payload = self._read_json(path)
            if isinstance(payload, Mapping):
                if int(payload.get("schema_version", self.SCHEMA_VERSION)) != self.SCHEMA_VERSION:
                    raise ManifestSchemaError(f"split {split_name} has unsupported schema")
                raw_records = payload.get("records")
            else:
                raw_records = payload
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise ManifestSchemaError(f"split {split_name} records must be a JSON array")
        parsed: List[ExternalRecord] = []
        seen_record_ids = set()
        for value in raw_records:
            if not isinstance(value, Mapping):
                raise ManifestSchemaError(f"split {split_name} contains a non-object record")
            record = ExternalRecord.from_mapping(value, split_name)
            if record.payload_path is not None:
                payload_path = Path(record.payload_path).expanduser()
                payload_path = (
                    payload_path
                    if payload_path.is_absolute()
                    else self.manifest_path.parent / payload_path
                ).resolve()
                actual = external_payload_file_hash(payload_path)
                if not hmac.compare_digest(actual, record.identity.content_hash):
                    raise ManifestSchemaError(
                        f"{split_name}/{record.record_id} payload hash mismatch: "
                        f"expected {record.identity.content_hash}, got {actual}"
                    )
            if record.record_id in seen_record_ids:
                raise ManifestSchemaError(f"duplicate record_id in {split_name}: {record.record_id}")
            seen_record_ids.add(record.record_id)
            parsed.append(record)
        return tuple(sorted(parsed, key=lambda item: item.record_id))

    def _validate_context_index(self, contexts: Any) -> None:
        if contexts is None:
            return
        if not isinstance(contexts, Mapping):
            raise ManifestSchemaError("manifest.contexts must be an object")
        actual: Dict[str, Dict[str, List[str]]] = {}
        for split_name, grouped in self._by_context.items():
            for context, records in grouped.items():
                actual.setdefault(context, {})[split_name] = [record.record_id for record in records]
        normalized: Dict[str, Dict[str, List[str]]] = {}
        for context, split_map in contexts.items():
            if not isinstance(split_map, Mapping):
                raise ManifestSchemaError(f"contexts.{context} must be an object")
            normalized[str(context)] = {}
            for split_name, record_ids in split_map.items():
                if split_name not in EXTERNAL_SPLITS:
                    raise ManifestSchemaError(f"contexts.{context} names unknown split {split_name}")
                normalized[str(context)][str(split_name)] = sorted(_stable_ids("record_ids", record_ids, nonempty=False))
        # Omitted empty split/context entries are equivalent.
        clean_actual = {
            context: {split: sorted(ids) for split, ids in splits.items() if ids}
            for context, splits in actual.items()
        }
        clean_normalized = {
            context: {split: ids for split, ids in splits.items() if ids}
            for context, splits in normalized.items()
        }
        if clean_actual != clean_normalized:
            raise ManifestSchemaError("manifest.contexts does not match split record context mapping")

    def _validate_episode_contexts(self, episode_contexts: Any) -> Dict[str, str]:
        """Validate the optional evaluation-episode routing table once at startup."""

        if episode_contexts is None:
            return {}
        if not isinstance(episode_contexts, Mapping):
            raise ManifestSchemaError("manifest.episode_contexts must be an object")
        known_contexts = {
            context
            for grouped in self._by_context.values()
            for context in grouped
        }
        normalized: Dict[str, str] = {}
        for raw_key, raw_context in episode_contexts.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ManifestSchemaError(
                    "manifest.episode_contexts keys must be non-empty strings"
                )
            if not isinstance(raw_context, str) or not raw_context.strip():
                raise ManifestSchemaError(
                    "manifest.episode_contexts values must be non-empty strings"
                )
            key = raw_key.strip()
            context = raw_context.strip()
            if key in normalized:
                raise ManifestSchemaError(
                    f"manifest.episode_contexts has duplicate normalized key {key!r}"
                )
            if context not in known_contexts:
                raise ManifestSchemaError(
                    f"manifest.episode_contexts key {key!r} names unknown context {context!r}"
                )
            normalized[key] = context
        return normalized

    def validate_context_splits(
        self,
        context_identifier: str,
        required_splits: Sequence[str],
    ) -> None:
        """Prove all episode-required fixed splits exist before online use."""

        context = str(context_identifier)
        if not context:
            raise ExternalDataError("context_identifier must be non-empty")
        normalized = tuple(dict.fromkeys(str(name) for name in required_splits))
        unknown = sorted(set(normalized) - set(EXTERNAL_SPLITS))
        if unknown:
            raise ExternalDataError(f"unknown required external splits: {unknown}")
        missing = [
            split_name
            for split_name in normalized
            if not self._by_context[split_name].get(context)
        ]
        if missing:
            raise ExternalDataError(
                f"context {context!r} has no records in required external splits: {missing}"
            )

    def begin_episode(
        self,
        episode_id: str,
        context_identifier: str,
        *,
        required_splits: Sequence[str] = ("calibration", "commit_query"),
    ) -> None:
        if self._active_episode_id is not None:
            raise ExternalDataError("an external-data episode is already active")
        if not episode_id or not context_identifier:
            raise ExternalDataError("episode_id and context_identifier must be non-empty")
        self.validate_context_splits(str(context_identifier), required_splits)
        self._active_episode_id = str(episode_id)
        self._active_context = str(context_identifier)
        self._support_sealed = False
        self._support_records = {}

    def audit_and_register_support(self, value: Any) -> DataIdentity:
        if self._active_episode_id is None:
            raise ExternalDataError("begin_episode must precede support registration")
        if self._support_sealed:
            raise ExternalDataError("support identity cannot be added after the first online update")
        identity = value if isinstance(value, DataIdentity) else DataIdentity.from_mapping(value, split_name="support")
        if identity.context_identifier != self._active_context:
            raise DataLeakageError("support context does not match the episode context")
        if identity.record_id in self._support_records:
            raise DataLeakageError(f"duplicate support record_id: {identity.record_id}")
        self._external_index.check_against(identity, f"support/{self._active_episode_id}/{identity.record_id}")
        # Adjacent/sliding support windows may legitimately share boundary frames
        # and transitions. Only an exact content duplicate is rejected here;
        # external-vs-support isolation above remains strict at every ID level.
        for previous in self._support_records.values():
            if previous.content_hash == identity.content_hash:
                raise DataLeakageError("duplicate support window content")
        self._support_records[identity.record_id] = identity
        return identity

    def audit_composed_support(
        self,
        value: Any,
        source_record_ids: Sequence[str],
    ) -> DataIdentity:
        """Read-only audit for a replay window composed from online support.

        The source segments were each admitted before their first optimizer
        use.  Temporal composition creates a new content hash, however, so the
        derived identity must itself be checked against every immutable
        external split.  This method deliberately does not register another
        online sample and is therefore valid after support admission is
        sealed; it only verifies provenance and the external isolation index.
        """

        if self._active_episode_id is None:
            raise ExternalDataError("begin_episode must precede composed support audit")
        identity = (
            value
            if isinstance(value, DataIdentity)
            else DataIdentity.from_mapping(value, split_name="composed_support")
        )
        if identity.context_identifier != self._active_context:
            raise DataLeakageError("composed support context does not match the episode")
        ordered_ids = tuple(str(record_id) for record_id in source_record_ids)
        if len(ordered_ids) < 2 or len(ordered_ids) != len(set(ordered_ids)):
            raise DataLeakageError(
                "composed support requires at least two unique source record IDs"
            )
        try:
            sources = tuple(self._support_records[record_id] for record_id in ordered_ids)
        except KeyError as exc:
            raise DataLeakageError(
                f"composed support source is absent from the episode ledger: {exc.args[0]}"
            ) from exc
        if any(source.context_identifier != identity.context_identifier for source in sources):
            raise DataLeakageError("composed support source contexts differ")
        trajectories = {source.trajectory_ids for source in sources}
        if len(trajectories) != 1 or identity.trajectory_ids != sources[0].trajectory_ids:
            raise DataLeakageError("composed support source trajectories differ")
        expected_transitions = tuple(
            transition
            for source in sources
            for transition in source.transition_ids
        )
        expected_frames = tuple(sources[0].frame_ids) + tuple(
            frame
            for source in sources[1:]
            for frame in source.frame_ids[1:]
        )
        if identity.transition_ids != expected_transitions:
            raise DataLeakageError(
                "composed support transition provenance differs from its sources"
            )
        if identity.frame_ids != expected_frames:
            raise DataLeakageError(
                "composed support frame provenance differs from its sources"
            )
        self._external_index.check_against(
            identity,
            f"composed_support/{self._active_episode_id}/{identity.record_id}",
        )
        return identity

    def open_support_registration(self) -> None:
        """Open the next incremental support-registration phase.

        AdaJEPA receives a new environment segment after every replan.  Each
        segment must be identity-audited before the optimizer is allowed to
        consume the enlarged buffer, so sealing is per update boundary rather
        than a permanent one-way episode latch.
        """

        if self._active_episode_id is None:
            raise ExternalDataError("no active episode")
        self._support_sealed = False

    def seal_support_for_online_update(self) -> None:
        if self._active_episode_id is None:
            raise ExternalDataError("no active episode")
        self._support_sealed = True

    def end_episode(self) -> None:
        self._active_episode_id = None
        self._active_context = None
        self._support_sealed = False
        self._support_records = {}

    def _context_records(self, split_name: str, context_identifier: Optional[str]) -> Tuple[ExternalRecord, ...]:
        if split_name == "commit_query":
            raise CommitQueryAccessError("commit-query requires consume_commit_query(token)")
        context = str(context_identifier if context_identifier is not None else self._active_context or "")
        if not context:
            raise ExternalDataError("context_identifier is required")
        records = self._by_context.get(split_name, {}).get(context, ())
        if not records:
            raise ExternalDataError(f"split {split_name} has no records for context {context!r}")
        return records

    def calibration(self, context_identifier: Optional[str] = None) -> Tuple[ExternalRecord, ...]:
        return self._context_records("calibration", context_identifier)

    def plasticity_support(self, context_identifier: Optional[str] = None) -> Tuple[ExternalRecord, ...]:
        return self._context_records("plasticity_support", context_identifier)

    def plasticity_query(self, context_identifier: Optional[str] = None) -> Tuple[ExternalRecord, ...]:
        return self._context_records("plasticity_query", context_identifier)

    def anchor(self, context_identifier: Optional[str] = None) -> Tuple[ExternalRecord, ...]:
        context = str(context_identifier if context_identifier is not None else self._active_context or "")
        if context and self._by_context["anchor"].get(context):
            return self._by_context["anchor"][context]
        if self._records["anchor"]:
            return self._records["anchor"]
        raise ExternalDataError("anchor split is empty")

    def report_test(self, context_identifier: Optional[str] = None) -> Tuple[ExternalRecord, ...]:
        # This is a pure lookup: no access ledger or router/system state changes.
        return self._context_records("report_test", context_identifier)

    def issue_commit_query_token(self, episode_id: str, proposal_id: str) -> CommitQueryToken:
        episode = str(episode_id)
        proposal = str(proposal_id)
        if self._active_episode_id != episode:
            raise CommitQueryAccessError("token episode does not match the active episode")
        if not proposal:
            raise CommitQueryAccessError("proposal_id must be non-empty")
        if episode in self._query_issued or episode in self._query_consumed:
            raise CommitQueryAccessError("commit-query authorization may be issued only once per episode")
        token_id = hashlib.sha256(
            f"{self.manifest_hash}\0{episode}\0{proposal}\0single_final_proposal".encode("utf-8")
        ).hexdigest()
        self._query_issued[episode] = token_id
        return CommitQueryToken(token_id, episode, proposal, self.manifest_hash)

    def consume_commit_query(
        self,
        token: CommitQueryToken,
        *,
        proposal_id: str,
        context_identifier: Optional[str] = None,
    ) -> Tuple[ExternalRecord, ...]:
        episode = token.episode_id
        expected = self._query_issued.get(episode)
        if expected is None or not hmac.compare_digest(expected, token.token_id):
            raise CommitQueryAccessError("unknown or already consumed commit-query token")
        if token.manifest_hash != self.manifest_hash:
            raise CommitQueryAccessError("commit-query token belongs to another manifest")
        if token.proposal_id != str(proposal_id):
            raise CommitQueryAccessError("commit-query token is bound to another proposal")
        if self._active_episode_id != episode:
            raise CommitQueryAccessError("commit-query token episode is no longer active")
        # Consume before lookup. Missing/corrupt context is a final-gate failure,
        # not permission to issue a replacement token or try another proposal.
        del self._query_issued[episode]
        self._query_consumed[episode] = token.token_id
        context = str(context_identifier if context_identifier is not None else self._active_context or "")
        records = self._by_context["commit_query"].get(context, ())
        if not records:
            raise CommitQueryAccessError(f"no commit-query records for context {context!r}")
        return records

    def commit_query_access_count(self, episode_id: str) -> int:
        return int(str(episode_id) in self._query_consumed)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "active_episode_id": self._active_episode_id,
            "active_context": self._active_context,
            "support_sealed": self._support_sealed,
            "support_records": copy.deepcopy(self._support_records),
            "query_issued": dict(self._query_issued),
            "query_consumed": dict(self._query_consumed),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ExternalDataError("unsupported external-data state schema")
        if state.get("manifest_hash") != self.manifest_hash:
            raise ExternalDataError("external-data state manifest hash mismatch")
        self._active_episode_id = state.get("active_episode_id")
        self._active_context = state.get("active_context")
        self._support_sealed = bool(state.get("support_sealed", False))
        self._support_records = copy.deepcopy(dict(state.get("support_records", {})))
        self._query_issued = {str(k): str(v) for k, v in dict(state.get("query_issued", {})).items()}
        self._query_consumed = {str(k): str(v) for k, v in dict(state.get("query_consumed", {})).items()}


__all__ = [
    "CommitQueryAccessError",
    "CommitQueryToken",
    "DataIdentity",
    "DataLeakageError",
    "EXTERNAL_SPLITS",
    "ExternalDataError",
    "ExternalDataRegistry",
    "ExternalRecord",
    "ManifestSchemaError",
    "canonical_json_hash",
    "sha256_file",
]
