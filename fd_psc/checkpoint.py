"""Atomic immutable sidecar checkpoints for persistent FD-PSC state."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .external_data import sha256_file


class CheckpointError(RuntimeError):
    pass


class CheckpointValidationError(CheckpointError):
    pass


def _hash_update(digest: "hashlib._Hash", value: Any) -> None:
    """Deterministically hash nested state, including tensor bytes and dtype."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif isinstance(value, float):
        if not math.isfinite(value):
            digest.update(b"F" + repr(value).encode("ascii") + b";")
        else:
            digest.update(b"F" + struct.pack("!d", value))
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        digest.update(b"S" + len(raw).to_bytes(8, "big") + raw)
    elif isinstance(value, bytes):
        digest.update(b"Y" + len(value).to_bytes(8, "big") + value)
    elif isinstance(value, Path):
        _hash_update(digest, str(value))
    elif isinstance(value, (torch.dtype, torch.device)):
        _hash_update(digest, str(value))
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _hash_update(digest, str(tensor.dtype))
        _hash_update(digest, tuple(tensor.shape))
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _hash_update(digest, str(array.dtype))
        _hash_update(digest, tuple(array.shape))
        digest.update(array.tobytes())
    elif np is not None and isinstance(value, np.generic):
        _hash_update(digest, value.item())
    elif dataclasses.is_dataclass(value):
        digest.update(b"D")
        _hash_update(digest, value.__class__.__module__ + "." + value.__class__.__qualname__)
        _hash_update(digest, dataclasses.asdict(value))
    elif isinstance(value, Mapping):
        digest.update(b"M" + len(value).to_bytes(8, "big"))
        keyed = []
        for key, item in value.items():
            key_digest = hashlib.sha256()
            _hash_update(key_digest, key)
            keyed.append((key_digest.digest(), key, item))
        for _, key, item in sorted(keyed, key=lambda entry: entry[0]):
            _hash_update(digest, key)
            _hash_update(digest, item)
    elif isinstance(value, (tuple, list)):
        digest.update((b"Q" if isinstance(value, tuple) else b"L") + len(value).to_bytes(8, "big"))
        for item in value:
            _hash_update(digest, item)
    elif isinstance(value, (set, frozenset)):
        digest.update(b"E" + len(value).to_bytes(8, "big"))
        children = []
        for item in value:
            child = hashlib.sha256()
            _hash_update(child, item)
            children.append(child.digest())
        for child in sorted(children):
            digest.update(child)
    else:
        # Checkpoint payloads should be structural, not arbitrary live objects.
        raise CheckpointValidationError(
            f"unsupported checkpoint value type for deterministic hashing: {type(value).__name__}"
        )


def state_content_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was exposed
        return torch.load(path, map_location="cpu")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic_bytes(path, data)


@dataclass(frozen=True)
class CheckpointReference:
    commit_id: str
    commit_sequence: int
    version_file: str
    file_hash: str
    state_hash: str
    schema_version: int


class SidecarCheckpointManager:
    """Write verified immutable versions, then atomically update ``latest``."""

    def __init__(
        self,
        *,
        state_directory: Path,
        latest_pointer_path: Path,
        base_checkpoint_hash: str,
        manifest_hash: str,
        schema_version: int = 1,
        retention_versions: int = 20,
        keep_commit_journal: bool = True,
        state_validator: Optional[Callable[[Any], None]] = None,
        migrations: Optional[Mapping[int, Callable[[Any], Any]]] = None,
    ) -> None:
        self.state_directory = Path(state_directory).expanduser().resolve()
        self.latest_pointer_path = Path(latest_pointer_path).expanduser().resolve()
        self.base_checkpoint_hash = str(base_checkpoint_hash).lower()
        self.manifest_hash = str(manifest_hash).lower()
        self.schema_version = int(schema_version)
        self.retention_versions = int(retention_versions)
        self.keep_commit_journal = bool(keep_commit_journal)
        self.state_validator = state_validator
        self.migrations = dict(migrations or {})
        if self.schema_version <= 0 or self.retention_versions <= 0:
            raise ValueError("schema_version and retention_versions must be positive")
        for name, value in (
            ("base_checkpoint_hash", self.base_checkpoint_hash),
            ("manifest_hash", self.manifest_hash),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.latest_pointer_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_commit_id(commit_id: str) -> str:
        value = str(commit_id)
        if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise CheckpointError("commit_id must use only [A-Za-z0-9_.-]")
        return value

    def _journal_path(self, commit_id: str) -> Path:
        return self.state_directory / f"journal-{self._safe_commit_id(commit_id)}.json"

    def next_available_id(self, base_id: str) -> str:
        """Return a deterministic journal ID without ever reusing an old one.

        A failed atomic save deliberately retains an ``aborted`` journal.  The
        unsuffixed ID is attempt one; retries use a fixed-width suffix so both
        commit and episode-snapshot IDs remain deterministic and sortable.
        """

        base_id = self._safe_commit_id(base_id)
        if not self._journal_path(base_id).exists():
            return base_id
        for attempt in range(2, 100_000_000):
            candidate = f"{base_id}-attempt-{attempt:08d}"
            if not self._journal_path(candidate).exists():
                return candidate
        raise CheckpointError("checkpoint journal attempt space is exhausted")

    def _write_journal(self, commit_id: str, value: Mapping[str, Any]) -> None:
        _atomic_json(self._journal_path(commit_id), value)

    def read_journal(self, commit_id: str) -> Mapping[str, Any]:
        """Read one audit journal without treating it as a loadable version."""

        path = self._journal_path(commit_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(
                f"commit journal is unreadable for {commit_id!r}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise CheckpointValidationError("commit journal must be a JSON object")
        return value

    def unresolved_episode_journals(
        self,
        restored_episode_sequence: int,
    ) -> Tuple[str, ...]:
        """Return unfinished system journals newer than a restored boundary.

        Such a journal is a durability tombstone: production work for that
        episode occurred, but the selected checkpoint does not cover its
        episode high-water mark.  Resuming it could re-open a consumed query.
        Generic manager journals without deterministic episode metadata are
        intentionally outside this system-level check.
        """

        restored_episode_sequence = int(restored_episode_sequence)
        if restored_episode_sequence < 0:
            raise CheckpointError("restored_episode_sequence must be non-negative")
        unresolved = []
        for path in sorted(self.state_directory.glob("journal-*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping) or value.get("status") not in {
                    "prepared",
                    "aborted",
                }:
                    continue
                commit_id = str(value.get("commit_id", ""))
                if path != self._journal_path(commit_id):
                    continue
                metadata = value.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                match = re.fullmatch(r"episode-(\d+)", str(metadata.get("episode_id", "")))
                if match is None:
                    continue
                # Episode IDs are zero-based while episode_sequence is the
                # number of admitted episodes. A restored count N covers only
                # IDs [0, N); an aborted journal for ID N is unresolved.
                if int(match.group(1)) >= restored_episode_sequence:
                    unresolved.append(commit_id)
            except Exception:
                # A malformed unrelated journal cannot prove an unresolved
                # episode. Identity/version corruption is handled separately
                # by normal load and recovery validation.
                continue
        return tuple(unresolved)

    def mark_rolled_back(
        self,
        commit_ids: Sequence[str],
        *,
        rollback_id: str,
        rollback_sequence: int,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Invalidate versions superseded by one committed rollback version.

        The rollback checkpoint must already be the verified ``latest``
        version.  Existing immutable version files are deliberately retained
        as audit evidence, but changing their journal status makes them
        ineligible for explicit load and recovery scans.  A failed commit that
        never reached checkpoint preparation still receives a terminal audit
        journal, so its ID cannot later be mistaken for an unused ID.
        """

        rollback_id = self._safe_commit_id(rollback_id)
        rollback_sequence = int(rollback_sequence)
        if rollback_sequence < 0:
            raise CheckpointError("rollback_sequence must be non-negative")
        rollback_journal = self.read_journal(rollback_id)
        if (
            rollback_journal.get("status") != "committed"
            or int(rollback_journal.get("commit_sequence", -1)) != rollback_sequence
        ):
            raise CheckpointError(
                "rollback journal must be committed before superseded journals are invalidated"
            )
        pointer = self._read_pointer()
        if pointer.get("commit_id") != rollback_id:
            raise CheckpointError("latest pointer does not reference the rollback checkpoint")

        if isinstance(commit_ids, (str, bytes)):
            raise CheckpointError("commit_ids must be a sequence of IDs, not one string")
        stable_ids = tuple(dict.fromkeys(self._safe_commit_id(value) for value in commit_ids))
        if not stable_ids:
            raise CheckpointError("periodic rollback must supersede at least one commit ID")
        if rollback_id in stable_ids:
            raise CheckpointError("rollback checkpoint cannot invalidate itself")
        rolled_back_at_ns = time.time_ns()
        for commit_id in stable_ids:
            path = self._journal_path(commit_id)
            existing: Dict[str, Any]
            if path.exists():
                value = self.read_journal(commit_id)
                status = str(value.get("status", ""))
                if status == "rolled_back":
                    if value.get("rollback_id") != rollback_id:
                        raise CheckpointError(
                            f"commit {commit_id!r} was already rolled back by another journal"
                        )
                    continue
                if status != "committed":
                    raise CheckpointError(
                        f"cannot roll back non-committed journal {commit_id!r} ({status!r})"
                    )
                existing = dict(value)
            else:
                match = re.fullmatch(
                    r"commit-(\d+)(?:-attempt-\d{8})?",
                    commit_id,
                )
                sequence = int(match.group(1)) if match is not None else rollback_sequence
                existing = {
                    "journal_schema_version": 1,
                    "commit_id": commit_id,
                    "commit_sequence": sequence,
                    "created_at_ns": rolled_back_at_ns,
                    "base_checkpoint_hash": self.base_checkpoint_hash,
                    "manifest_hash": self.manifest_hash,
                    "metadata": {},
                }
            existing.update(
                {
                    "status": "rolled_back",
                    "rolled_back_at_ns": rolled_back_at_ns,
                    "rollback_id": rollback_id,
                    "rollback_sequence": rollback_sequence,
                    "rollback_reason": str(reason),
                    "rollback_metadata": dict(metadata or {}),
                }
            )
            self._write_journal(commit_id, existing)
        _fsync_directory(self.state_directory)

    def _read_pointer_bytes(self) -> Optional[bytes]:
        try:
            return self.latest_pointer_path.read_bytes()
        except FileNotFoundError:
            return None

    def _restore_pointer(self, old: Optional[bytes]) -> None:
        if old is None:
            if self.latest_pointer_path.exists():
                self.latest_pointer_path.unlink()
                _fsync_directory(self.latest_pointer_path.parent)
        else:
            _atomic_bytes(self.latest_pointer_path, old)

    def save_committed(
        self,
        state: Any,
        *,
        commit_id: str,
        commit_sequence: int,
        config_identity: str,
        journal_metadata: Optional[Mapping[str, Any]] = None,
    ) -> CheckpointReference:
        commit_id = self._safe_commit_id(commit_id)
        if int(commit_sequence) < 0:
            raise CheckpointError("commit_sequence must be non-negative")
        if self._journal_path(commit_id).exists():
            raise CheckpointError(f"commit_id is immutable and already has a journal: {commit_id}")
        created_at_ns = time.time_ns()
        state_hash = state_content_hash(state)
        envelope = {
            "schema_version": self.schema_version,
            "commit_id": commit_id,
            "commit_sequence": int(commit_sequence),
            "created_at_ns": created_at_ns,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "manifest_hash": self.manifest_hash,
            "config_identity": str(config_identity),
            "state_hash": state_hash,
            "state": state,
        }
        prepared = {
            "journal_schema_version": 1,
            "schema_version": self.schema_version,
            "status": "prepared",
            "commit_id": commit_id,
            "commit_sequence": int(commit_sequence),
            "created_at_ns": created_at_ns,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "manifest_hash": self.manifest_hash,
            "state_hash": state_hash,
            "metadata": dict(journal_metadata or {}),
        }
        old_pointer = self._read_pointer_bytes()
        pointer_updated = False
        temp_path: Optional[Path] = None
        final_path: Optional[Path] = None
        self._write_journal(commit_id, prepared)
        try:
            fd, raw_temp = tempfile.mkstemp(prefix=f".state-{commit_id}.", suffix=".tmp", dir=str(self.state_directory))
            temp_path = Path(raw_temp)
            with os.fdopen(fd, "wb") as handle:
                torch.save(envelope, handle)
                handle.flush()
                os.fsync(handle.fileno())
            file_hash = sha256_file(temp_path)
            loaded = self._load_version(temp_path, expected_file_hash=file_hash)
            if loaded["state_hash"] != state_hash:
                raise CheckpointValidationError("reloaded temporary version changed state hash")
            final_name = f"state-{commit_id}-{file_hash[:16]}.pt"
            final_path = self.state_directory / final_name
            if final_path.exists():
                if sha256_file(final_path) != file_hash:
                    raise CheckpointValidationError("immutable version filename collision")
                temp_path.unlink()
                temp_path = None
            else:
                os.replace(temp_path, final_path)
                temp_path = None
                _fsync_directory(self.state_directory)
            # Reload the immutable final path, not merely the temporary file.
            self._load_version(final_path, expected_file_hash=file_hash)
            reference = CheckpointReference(
                commit_id=commit_id,
                commit_sequence=int(commit_sequence),
                version_file=final_name,
                file_hash=file_hash,
                state_hash=state_hash,
                schema_version=self.schema_version,
            )
            pointer = dataclasses.asdict(reference)
            pointer["base_checkpoint_hash"] = self.base_checkpoint_hash
            pointer["manifest_hash"] = self.manifest_hash
            _atomic_json(self.latest_pointer_path, pointer)
            pointer_updated = True
            committed = dict(prepared)
            committed.update(
                {
                    "status": "committed",
                    "version_file": final_name,
                    "file_hash": file_hash,
                    "committed_at_ns": time.time_ns(),
                }
            )
            self._write_journal(commit_id, committed)
            self._prune_retention(exclude={final_name})
            return reference
        except Exception as exc:
            if pointer_updated:
                try:
                    self._restore_pointer(old_pointer)
                except Exception as restore_exc:
                    raise CheckpointError(
                        f"checkpoint failed and latest pointer restoration failed: {restore_exc}"
                    ) from exc
            aborted = dict(prepared)
            aborted.update({"status": "aborted", "reason": f"{type(exc).__name__}: {exc}"})
            try:
                self._write_journal(commit_id, aborted)
            except Exception:
                pass
            raise CheckpointError(f"sidecar checkpoint commit failed: {exc}") from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _load_version(self, path: Path, *, expected_file_hash: Optional[str] = None) -> Dict[str, Any]:
        path = Path(path).resolve()
        if not path.is_file() or path.parent != self.state_directory:
            raise CheckpointValidationError("version must be an immutable file inside state_directory")
        if expected_file_hash is not None:
            actual_file_hash = sha256_file(path)
            if actual_file_hash != expected_file_hash:
                raise CheckpointValidationError("version file content hash mismatch")
        try:
            envelope = _torch_load(path)
        except Exception as exc:
            raise CheckpointValidationError(f"cannot reload checkpoint version: {exc}") from exc
        if not isinstance(envelope, Mapping):
            raise CheckpointValidationError("checkpoint version is not an envelope mapping")
        version = int(envelope.get("schema_version", -1))
        if version > self.schema_version or version <= 0:
            raise CheckpointValidationError(f"unsupported checkpoint schema {version}")
        if envelope.get("base_checkpoint_hash") != self.base_checkpoint_hash:
            raise CheckpointValidationError("base checkpoint hash mismatch")
        if envelope.get("manifest_hash") != self.manifest_hash:
            raise CheckpointValidationError("module manifest hash mismatch")
        expected_state_hash = str(envelope.get("state_hash", ""))
        actual_state_hash = state_content_hash(envelope.get("state"))
        if expected_state_hash != actual_state_hash:
            raise CheckpointValidationError("checkpoint state content hash mismatch")
        result = dict(envelope)
        if version < self.schema_version:
            current_state = result["state"]
            for old_version in range(version, self.schema_version):
                migration = self.migrations.get(old_version)
                if migration is None:
                    raise CheckpointValidationError(
                        f"no explicit migration from checkpoint schema {old_version}"
                    )
                current_state = migration(current_state)
            result["state"] = current_state
            result["schema_version"] = self.schema_version
        if self.state_validator is not None:
            self.state_validator(result["state"])
        return result

    def _read_pointer(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.latest_pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(f"latest pointer is unreadable: {exc}") from exc
        if not isinstance(value, Mapping):
            raise CheckpointValidationError("latest pointer must be a JSON object")
        if value.get("base_checkpoint_hash") != self.base_checkpoint_hash:
            raise CheckpointValidationError("latest pointer base hash mismatch")
        if value.get("manifest_hash") != self.manifest_hash:
            raise CheckpointValidationError("latest pointer manifest hash mismatch")
        return value

    def _journal_is_committed(self, commit_id: str, version_file: str, file_hash: str) -> bool:
        try:
            value = json.loads(self._journal_path(commit_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(value, Mapping)
            and value.get("status") == "committed"
            and value.get("commit_id") == commit_id
            and value.get("version_file") == version_file
            and value.get("file_hash") == file_hash
            and value.get("base_checkpoint_hash") == self.base_checkpoint_hash
            and value.get("manifest_hash") == self.manifest_hash
        )

    def _validate_journal_envelope(
        self,
        journal: Mapping[str, Any],
        envelope: Mapping[str, Any],
        *,
        commit_id: str,
    ) -> None:
        """Fail closed unless one committed journal names this exact envelope."""

        if journal.get("commit_id") != commit_id or envelope.get("commit_id") != commit_id:
            raise CheckpointValidationError(
                "checkpoint commit_id differs across journal and envelope"
            )
        try:
            journal_sequence = int(journal["commit_sequence"])
            envelope_sequence = int(envelope["commit_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointValidationError(
                "checkpoint commit_sequence is missing or invalid"
            ) from exc
        if journal_sequence != envelope_sequence:
            raise CheckpointValidationError(
                "checkpoint commit_sequence differs across journal and envelope"
            )
        if journal.get("state_hash") != envelope.get("state_hash"):
            raise CheckpointValidationError(
                "checkpoint state_hash differs across journal and envelope"
            )
        journal_schema = journal.get("schema_version")
        if journal_schema is not None and int(journal_schema) != int(
            envelope.get("schema_version", -1)
        ):
            raise CheckpointValidationError(
                "checkpoint schema differs across journal and envelope"
            )

    def load_latest(self, *, recover_if_needed: bool = False) -> Tuple[Any, CheckpointReference]:
        try:
            pointer = self._read_pointer()
            version_file = str(pointer["version_file"])
            file_hash = str(pointer["file_hash"])
            commit_id = str(pointer["commit_id"])
            if Path(version_file).name != version_file:
                raise CheckpointValidationError("latest pointer contains an unsafe version path")
            if not self._journal_is_committed(commit_id, version_file, file_hash):
                raise CheckpointValidationError("latest pointer does not have a matching committed journal")
            journal = self.read_journal(commit_id)
            envelope = self._load_version(
                self.state_directory / version_file,
                expected_file_hash=file_hash,
            )
            self._validate_journal_envelope(journal, envelope, commit_id=commit_id)
            if int(pointer.get("commit_sequence", -1)) != int(
                envelope["commit_sequence"]
            ):
                raise CheckpointValidationError(
                    "latest pointer commit_sequence differs from its version"
                )
            if pointer.get("state_hash") != envelope.get("state_hash"):
                raise CheckpointValidationError(
                    "latest pointer state_hash differs from its version"
                )
            if int(pointer.get("schema_version", -1)) != int(
                envelope.get("schema_version", -1)
            ):
                raise CheckpointValidationError(
                    "latest pointer schema differs from its version"
                )
            reference = CheckpointReference(
                commit_id=commit_id,
                commit_sequence=int(pointer["commit_sequence"]),
                version_file=version_file,
                file_hash=file_hash,
                state_hash=str(envelope["state_hash"]),
                schema_version=int(envelope["schema_version"]),
            )
            return envelope["state"], reference
        except Exception:
            if not recover_if_needed:
                raise
            return self.recover_latest(repair_pointer=True)

    def load_version(self, path: Path) -> Tuple[Any, CheckpointReference]:
        """Load one immutable version only if its committed journal matches."""

        resolved = Path(path).expanduser().resolve()
        if resolved.parent != self.state_directory or resolved.name != Path(path).name:
            raise CheckpointValidationError(
                "explicit version must be a file directly inside state_directory"
            )
        if not resolved.is_file():
            raise CheckpointValidationError("explicit checkpoint version does not exist")
        file_hash = sha256_file(resolved)
        envelope = self._load_version(resolved, expected_file_hash=file_hash)
        commit_id = str(envelope.get("commit_id", ""))
        if not self._journal_is_committed(commit_id, resolved.name, file_hash):
            raise CheckpointValidationError(
                "explicit version does not have a matching committed journal"
            )
        journal = self.read_journal(commit_id)
        self._validate_journal_envelope(journal, envelope, commit_id=commit_id)
        reference = CheckpointReference(
            commit_id=commit_id,
            commit_sequence=int(envelope["commit_sequence"]),
            version_file=resolved.name,
            file_hash=file_hash,
            state_hash=str(envelope["state_hash"]),
            schema_version=int(envelope["schema_version"]),
        )
        return envelope["state"], reference

    def recover_latest(self, *, repair_pointer: bool = False) -> Tuple[Any, CheckpointReference]:
        candidates = []
        for journal_path in sorted(self.state_directory.glob("journal-*.json")):
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                if not isinstance(journal, Mapping) or journal.get("status") != "committed":
                    continue
                if journal.get("base_checkpoint_hash") != self.base_checkpoint_hash:
                    continue
                if journal.get("manifest_hash") != self.manifest_hash:
                    continue
                version_file = str(journal["version_file"])
                file_hash = str(journal["file_hash"])
                commit_id = str(journal["commit_id"])
                if journal_path != self._journal_path(commit_id):
                    continue
                if Path(version_file).name != version_file:
                    continue
                envelope = self._load_version(
                    self.state_directory / version_file,
                    expected_file_hash=file_hash,
                )
                self._validate_journal_envelope(
                    journal,
                    envelope,
                    commit_id=commit_id,
                )
                candidates.append((int(envelope["commit_sequence"]), commit_id, journal, envelope))
            except Exception:
                continue
        if not candidates:
            raise CheckpointValidationError("no complete committed sidecar version can be recovered")
        _, commit_id, journal, envelope = max(candidates, key=lambda item: (item[0], item[1]))
        reference = CheckpointReference(
            commit_id=commit_id,
            commit_sequence=int(journal["commit_sequence"]),
            version_file=str(journal["version_file"]),
            file_hash=str(journal["file_hash"]),
            state_hash=str(envelope["state_hash"]),
            schema_version=int(envelope["schema_version"]),
        )
        if repair_pointer:
            pointer = dataclasses.asdict(reference)
            pointer["base_checkpoint_hash"] = self.base_checkpoint_hash
            pointer["manifest_hash"] = self.manifest_hash
            _atomic_json(self.latest_pointer_path, pointer)
        return envelope["state"], reference

    def _prune_retention(self, *, exclude: set[str]) -> None:
        committed = []
        rolled_back = []
        for path in self.state_directory.glob("journal-*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(value, Mapping) or value.get("status") not in {
                "committed",
                "rolled_back",
            }:
                continue
            # Corrupt journals are audit evidence, not retention inputs.  Never
            # trust their claimed sequence or version path to evict a verified
            # fallback.  Invalid entries and their files are left untouched.
            try:
                commit_id = str(value["commit_id"])
                if path != self._journal_path(commit_id):
                    continue
                version_file = str(value["version_file"])
                file_hash = str(value["file_hash"])
                if Path(version_file).name != version_file:
                    continue
                envelope = self._load_version(
                    self.state_directory / version_file,
                    expected_file_hash=file_hash,
                )
                self._validate_journal_envelope(
                    value,
                    envelope,
                    commit_id=commit_id,
                )
                sequence = int(envelope["commit_sequence"])
            except Exception:
                continue
            if value.get("status") == "committed":
                if not self._journal_is_committed(
                    commit_id,
                    version_file,
                    file_hash,
                ):
                    continue
                committed.append((sequence, commit_id, path, value))
            else:
                rolled_back.append((path, value))
        committed.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, journal_path, value in committed[self.retention_versions :]:
            version_file = str(value.get("version_file", ""))
            if version_file in exclude or Path(version_file).name != version_file:
                continue
            version_path = self.state_directory / version_file
            try:
                if version_path.exists():
                    version_path.unlink()
                if not self.keep_commit_journal:
                    journal_path.unlink()
            except OSError:
                # Retention failure cannot invalidate an otherwise atomic commit;
                # the next save will retry and disk usage is observable in metrics.
                continue
        # Once a later verified checkpoint is being published, superseded
        # period versions have already outlived their canary boundary.  Their
        # terminal journals retain hashes and rollback provenance, while the
        # large tensor payloads can now follow the retention policy.
        for _, value in rolled_back:
            version_file = str(value.get("version_file", ""))
            if (
                not version_file
                or version_file in exclude
                or Path(version_file).name != version_file
            ):
                continue
            version_path = self.state_directory / version_file
            try:
                if version_path.exists():
                    version_path.unlink()
            except OSError:
                continue
        _fsync_directory(self.state_directory)


__all__ = [
    "CheckpointError",
    "CheckpointReference",
    "CheckpointValidationError",
    "SidecarCheckpointManager",
    "state_content_hash",
]
