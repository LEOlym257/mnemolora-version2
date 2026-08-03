#!/usr/bin/env python3
"""Build and audit a versioned FD-PSC external-data manifest.

Each split file is JSON with either a top-level ``records`` list or a bare
list.  The command computes file hashes, validates record identity/content
hashes, and refuses every trajectory/transition/frame/content overlap before
writing the manifest.  It never reads or writes model state.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


SPLITS = (
    "calibration",
    "commit_query",
    "plasticity_support",
    "plasticity_query",
    "report_test",
    "anchor",
)


def canonical_json_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(name: str, value: object) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return text.lower()


def load_records(path: Path) -> List[dict]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read split JSON {path}: {exc}") from exc
    if isinstance(value, Mapping):
        if value.get("schema_version") != 1:
            raise ValueError(f"{path}: schema_version must be 1")
        value = value.get("records")
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a records list")
    return value


def stable_ids(name: str, value: object) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        raise ValueError(f"{name} must be a string or sequence of strings")
    if not values or any(not item for item in values):
        raise ValueError(f"{name} must contain stable non-empty IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate IDs")
    return values


def payload_file_content_hash(path: Path) -> str:
    """Return the runtime identity hash after validating payload readability."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"external payload file is not readable: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        try:
            with resolved.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read external payload JSON {resolved}: {exc}") from exc
        actual = canonical_json_hash(value)
    elif suffix in {".pt", ".pth"}:
        try:
            import torch

            try:
                value = torch.load(resolved, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(resolved, map_location="cpu")
        except Exception as exc:
            raise ValueError(f"cannot read external payload {resolved}: {exc}") from exc
        actual = file_sha256(resolved)
    else:
        raise ValueError(f"unsupported external payload format {resolved.suffix!r}: {resolved}")
    if not isinstance(value, Mapping):
        raise ValueError(f"external payload file must contain a mapping: {resolved}")
    return actual


def identities(
    record: Mapping,
    *,
    payload_root: Optional[Path] = None,
) -> Dict[str, Set[str]]:
    required = (
        "record_id",
        "context_identifier",
        "transition_ids",
        "frame_ids",
        "content_hash",
    )
    for key in required:
        if key not in record:
            raise ValueError(f"record missing required field {key!r}")
    record_id = str(record["record_id"])
    context_identifier = str(record["context_identifier"])
    if not record_id or not context_identifier:
        raise ValueError("record_id and context_identifier must be non-empty")
    trajectory_values = record.get("trajectory_ids", record.get("trajectory_id"))
    if trajectory_values is None:
        raise ValueError(f"record {record['record_id']!r} has no trajectory identity")
    trajectories = stable_ids("trajectory_ids", trajectory_values)
    transitions = stable_ids("transition_ids", record["transition_ids"])
    frames = stable_ids("frame_ids", record["frame_ids"])
    content_hash = _sha256_text("content_hash", record["content_hash"])
    has_inline_payload = record.get("payload") is not None
    has_payload_path = record.get("payload_path") is not None
    if has_inline_payload == has_payload_path:
        raise ValueError(
            f"record {record_id!r} must provide exactly one of payload or payload_path"
        )
    if has_inline_payload and not isinstance(record["payload"], Mapping):
        raise ValueError(f"record {record_id!r} payload must be a mapping")
    if has_payload_path and not str(record["payload_path"]):
        raise ValueError(f"record {record_id!r} payload_path must be non-empty")
    actual_content_hashes = []
    if has_inline_payload:
        actual_content_hashes.append(canonical_json_hash(record["payload"]))
    if has_payload_path:
        if payload_root is None:
            raise ValueError(
                f"record {record['record_id']!r} has payload_path but no payload_root was supplied"
            )
        payload_path = Path(str(record["payload_path"])).expanduser()
        if not payload_path.is_absolute():
            payload_path = Path(payload_root) / payload_path
        actual_content_hashes.append(payload_file_content_hash(payload_path))
    for actual_content_hash in actual_content_hashes:
        if not hmac.compare_digest(actual_content_hash, content_hash):
            raise ValueError(
                f"record {record['record_id']!r} payload content hash mismatch: "
                f"expected {content_hash}, got {actual_content_hash}"
            )
    return {
        "trajectory": {str(x) for x in trajectories},
        "transition": {str(x) for x in transitions},
        "frame": {str(x) for x in frames},
        "content": {content_hash},
    }


def audit(
    split_records: Mapping[str, Iterable[Mapping]],
    *,
    payload_root: Optional[Path] = None,
) -> Dict[str, object]:
    seen = {kind: {} for kind in ("trajectory", "transition", "frame", "content")}
    contexts: Dict[str, Dict[str, List[str]]] = {}
    record_count = 0
    for split in SPLITS:
        local_ids: Set[str] = set()
        for record in split_records[split]:
            record_count += 1
            record_id = str(record["record_id"])
            if record_id in local_ids:
                raise ValueError(f"duplicate record_id {record_id!r} in {split}")
            local_ids.add(record_id)
            context = str(record["context_identifier"])
            contexts.setdefault(context, {name: [] for name in SPLITS})[split].append(record_id)
            for kind, values in identities(record, payload_root=payload_root).items():
                for value in values:
                    previous = seen[kind].get(value)
                    if previous is not None:
                        raise ValueError(
                            f"external split leakage: {kind} {value!r} occurs in "
                            f"{previous[0]}/{previous[1]} and {split}/{record_id}"
                        )
                    seen[kind][value] = (split, record_id)
    for by_split in contexts.values():
        for ids in by_split.values():
            ids.sort()
    return {
        "record_count": record_count,
        "context_count": len(contexts),
        "identity_counts": {kind: len(values) for kind, values in seen.items()},
        "contexts": dict(sorted(contexts.items())),
    }


def atomic_json_dump(value: Mapping, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint-hash", required=True)
    parser.add_argument("--preprocess-hash", required=True)
    parser.add_argument("--latent-adapter-schema", required=True)
    parser.add_argument("--manifest-id", default=None)
    parser.add_argument(
        "--episode-contexts",
        type=Path,
        default=None,
        help=(
            "optional JSON object mapping sample/seed episode keys to explicit "
            "context identifiers; this is never inferred from split membership"
        ),
    )
    for split in SPLITS:
        parser.add_argument(f"--{split.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    paths = {split: getattr(args, split).expanduser().resolve() for split in SPLITS}
    for split, path in paths.items():
        if not path.is_file():
            parser.error(f"{split} file does not exist: {path}")
    records = {split: load_records(path) for split, path in paths.items()}
    report = audit(records, payload_root=output.parent)
    episode_contexts = None
    if args.episode_contexts is not None:
        episode_context_path = args.episode_contexts.expanduser().resolve()
        if not episode_context_path.is_file():
            parser.error(f"episode-context mapping does not exist: {episode_context_path}")
        raw_episode_contexts = json.loads(
            episode_context_path.read_text(encoding="utf-8")
        )
        if not isinstance(raw_episode_contexts, Mapping):
            parser.error("episode-context mapping must be a JSON object")
        known_contexts = set(report["contexts"])
        episode_contexts = {}
        for raw_key, raw_context in sorted(raw_episode_contexts.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, str) or not isinstance(raw_context, str):
                parser.error("episode-context keys and values must be strings")
            key = raw_key.strip()
            context = raw_context.strip()
            if not key or not context:
                parser.error("episode-context keys and values must be non-empty")
            if key in episode_contexts:
                parser.error(f"duplicate normalized episode-context key {key!r}")
            if context not in known_contexts:
                parser.error(
                    f"episode-context key {key!r} names unknown context {context!r}"
                )
            episode_contexts[key] = context
    manifest = {
        "schema_version": 1,
        "manifest_id": args.manifest_id or output.stem,
        "base_checkpoint_hash": _sha256_text(
            "base_checkpoint_hash", args.base_checkpoint_hash
        ),
        "preprocess_hash": _sha256_text("preprocess_hash", args.preprocess_hash),
        "latent_adapter_schema": str(args.latent_adapter_schema),
        "splits": {
            split: {
                "path": os.path.relpath(path, output.parent).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for split, path in paths.items()
        },
        "contexts": report.pop("contexts"),
        "leakage_audit": {"status": "pass", **report},
    }
    if episode_contexts is not None:
        manifest["episode_contexts"] = episode_contexts
    manifest["manifest_content_hash"] = canonical_json_hash(manifest)
    atomic_json_dump(manifest, output)
    print(json.dumps({"manifest": str(output), **manifest["leakage_audit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
