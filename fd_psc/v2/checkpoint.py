"""Schema-2 atomic sidecars for FSD V2.

This module intentionally has no dependency on ``fd_psc.external_data``.  A
V2 checkpoint binds the internal algorithm state to the official base hash,
the injected target manifest, and the V2-only configuration identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch


class FSDV2CheckpointError(RuntimeError):
    """Raised when an FSD V2 sidecar cannot be saved or verified."""


_V1_MIGRATION_ERROR = (
    "V1 sidecar cannot be loaded as FSD V2 without explicit migration."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported PyTorch
        return torch.load(path, map_location="cpu")


def _digest(name: str, value: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return result


@dataclass(frozen=True)
class FSDV2CheckpointReference:
    commit_sequence: int
    version_file: str
    file_hash: str


class FSDV2CheckpointStore:
    """Write immutable state files and an atomically replaced latest pointer."""

    SCHEMA_VERSION = 2
    ALGORITHM_VERSION = "fsd_v2"

    def __init__(
        self,
        *,
        state_directory: Path,
        latest_pointer_path: Path,
        base_checkpoint_hash: str,
        runtime_base_state_hash: str,
        target_manifest_hash: str,
        config_identity: str,
        retention_versions: int = 20,
    ) -> None:
        self.state_directory = Path(state_directory).expanduser().resolve()
        self.latest_pointer_path = Path(latest_pointer_path).expanduser().resolve()
        self.base_checkpoint_hash = _digest(
            "base_checkpoint_hash", base_checkpoint_hash
        )
        self.runtime_base_state_hash = _digest(
            "runtime_base_state_hash", runtime_base_state_hash
        )
        self.target_manifest_hash = _digest(
            "target_manifest_hash", target_manifest_hash
        )
        self.config_identity = _digest("config_identity", config_identity)
        self.retention_versions = int(retention_versions)
        if self.retention_versions <= 0:
            raise ValueError("retention_versions must be positive")
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.latest_pointer_path.parent.mkdir(parents=True, exist_ok=True)

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "algorithm_version": self.ALGORITHM_VERSION,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "runtime_base_state_hash": self.runtime_base_state_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "config_identity": self.config_identity,
        }

    def _validate_envelope(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise FSDV2CheckpointError("FSD V2 checkpoint is not a mapping")
        version = int(value.get("schema_version", -1))
        if version == 1:
            raise FSDV2CheckpointError(_V1_MIGRATION_ERROR)
        if version != self.SCHEMA_VERSION:
            raise FSDV2CheckpointError(
                f"unsupported FSD V2 checkpoint schema {version}"
            )
        if str(value.get("algorithm_version", "")) != self.ALGORITHM_VERSION:
            raise FSDV2CheckpointError("checkpoint algorithm identity is not fsd_v2")
        for key, expected in (
            ("base_checkpoint_hash", self.base_checkpoint_hash),
            ("runtime_base_state_hash", self.runtime_base_state_hash),
            ("target_manifest_hash", self.target_manifest_hash),
            ("config_identity", self.config_identity),
        ):
            if str(value.get(key, "")).lower() != expected:
                raise FSDV2CheckpointError(f"checkpoint {key} mismatch")
        state = value.get("state")
        if not isinstance(state, Mapping):
            raise FSDV2CheckpointError("checkpoint state is not a mapping")
        state_version = int(state.get("schema_version", -1))
        if state_version == 1:
            raise FSDV2CheckpointError(_V1_MIGRATION_ERROR)
        if state_version != self.SCHEMA_VERSION:
            raise FSDV2CheckpointError(
                f"unsupported FSD V2 algorithm-state schema {state_version}"
            )
        if str(state.get("algorithm_version", "")) != self.ALGORITHM_VERSION:
            raise FSDV2CheckpointError("algorithm state is not fsd_v2")
        envelope_sequence = int(value.get("commit_sequence", -1))
        state_sequence = int(state.get("commit_sequence", -1))
        if envelope_sequence <= 0 or state_sequence != envelope_sequence:
            raise FSDV2CheckpointError(
                "checkpoint envelope/state commit_sequence mismatch"
            )
        return value

    def save(
        self,
        state: Mapping[str, Any],
        *,
        commit_sequence: int,
    ) -> FSDV2CheckpointReference:
        sequence = int(commit_sequence)
        if sequence <= 0:
            raise FSDV2CheckpointError("commit_sequence must be positive")
        envelope = {
            **self._identity(),
            "commit_sequence": sequence,
            "state": dict(state),
        }
        self._validate_envelope(envelope)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".state-commit-{sequence:08d}.",
            suffix=".tmp",
            dir=str(self.state_directory),
        )
        temporary = Path(raw_temp)
        final_path: Optional[Path] = None
        created_final = False
        pointer_published = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(envelope, handle)
                handle.flush()
                os.fsync(handle.fileno())
            file_hash = _sha256_file(temporary)
            # Verify the bytes before they become eligible for publication.
            self._validate_envelope(_torch_load(temporary))
            final_name = f"state-commit-{sequence:08d}-{file_hash[:16]}.pt"
            final_path = self.state_directory / final_name
            if final_path.exists():
                if _sha256_file(final_path) != file_hash:
                    raise FSDV2CheckpointError("immutable checkpoint filename collision")
                temporary.unlink()
            else:
                os.replace(temporary, final_path)
                created_final = True
            reference = FSDV2CheckpointReference(sequence, final_name, file_hash)
            pointer = {
                **self._identity(),
                "commit_sequence": sequence,
                "version_file": final_name,
                "file_hash": file_hash,
            }
            try:
                _atomic_json(self.latest_pointer_path, pointer)
                pointer_published = True
            except Exception:
                # If a filesystem wrapper reports an error after the atomic
                # replace actually completed, the exact pointer bytes remain
                # the authoritative logical-publication result.
                try:
                    published_value = json.loads(
                        self.latest_pointer_path.read_text(encoding="utf-8")
                    )
                    pointer_published = published_value == pointer
                except Exception:
                    pointer_published = False
                if not pointer_published:
                    raise
            # The pointer swap above is the logical publication point.  No
            # best-effort housekeeping performed afterwards may turn that
            # committed state into an apparent save failure (which would make
            # the live transaction roll back while the pointer stayed ahead).
            try:
                self._prune(exclude=final_name)
            except Exception:
                pass
            return reference
        except FSDV2CheckpointError:
            raise
        except Exception as exc:
            raise FSDV2CheckpointError(
                f"FSD V2 checkpoint save failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()
            # A version file whose pointer was never published is not a
            # committed state and must not remain selectable as an explicit
            # resume target.
            if created_final and not pointer_published and final_path is not None:
                try:
                    final_path.unlink()
                except OSError:
                    pass

    def _load_path(
        self,
        path: Path,
        *,
        expected_hash: Optional[str] = None,
    ) -> Tuple[Mapping[str, Any], FSDV2CheckpointReference]:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FSDV2CheckpointError(f"checkpoint file is not readable: {target}")
        actual_hash = _sha256_file(target)
        if expected_hash is not None and actual_hash != str(expected_hash).lower():
            raise FSDV2CheckpointError("checkpoint file hash mismatch")
        try:
            envelope = self._validate_envelope(_torch_load(target))
        except FSDV2CheckpointError:
            raise
        except Exception as exc:
            raise FSDV2CheckpointError(
                f"checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        sequence = int(envelope.get("commit_sequence", -1))
        if sequence <= 0:
            raise FSDV2CheckpointError("checkpoint commit_sequence is invalid")
        return envelope["state"], FSDV2CheckpointReference(
            sequence, target.name, actual_hash
        )

    def load_pointer(
        self, pointer_path: Optional[Path] = None
    ) -> Tuple[Mapping[str, Any], FSDV2CheckpointReference]:
        pointer_target = Path(pointer_path or self.latest_pointer_path).expanduser().resolve()
        try:
            pointer = json.loads(pointer_target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FSDV2CheckpointError(f"latest pointer is unreadable: {exc}") from exc
        if not isinstance(pointer, Mapping):
            raise FSDV2CheckpointError("latest pointer is not a mapping")
        # A V1 latest pointer may identify its schema before any tensor file is read.
        pointer_version = int(pointer.get("schema_version", -1))
        if pointer_version == 1:
            raise FSDV2CheckpointError(_V1_MIGRATION_ERROR)
        for key, expected in self._identity().items():
            actual = pointer.get(key)
            if key.endswith("_hash") or key == "config_identity":
                actual = str(actual or "").lower()
            if actual != expected:
                raise FSDV2CheckpointError(f"latest pointer {key} mismatch")
        version_file = str(pointer.get("version_file", ""))
        if not version_file or Path(version_file).name != version_file:
            raise FSDV2CheckpointError("latest pointer contains an unsafe version path")
        state, reference = self._load_path(
            self.state_directory / version_file,
            expected_hash=str(pointer.get("file_hash", "")),
        )
        if int(pointer.get("commit_sequence", -1)) != reference.commit_sequence:
            raise FSDV2CheckpointError("latest pointer commit sequence mismatch")
        return state, reference

    def load_resume(
        self, resume_path: Path
    ) -> Tuple[Mapping[str, Any], FSDV2CheckpointReference]:
        target = Path(resume_path).expanduser().resolve()
        if target.suffix.lower() == ".json":
            return self.load_pointer(target)
        return self._load_path(target)

    def _prune(self, *, exclude: str) -> None:
        versions = sorted(
            self.state_directory.glob("state-commit-*.pt"),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in versions[self.retention_versions :]:
            if path.name == exclude:
                continue
            try:
                path.unlink()
            except OSError:
                pass


__all__ = [
    "FSDV2CheckpointError",
    "FSDV2CheckpointReference",
    "FSDV2CheckpointStore",
]
