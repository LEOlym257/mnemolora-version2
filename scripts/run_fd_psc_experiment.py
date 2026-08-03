#!/usr/bin/env python3
"""Run one isolated FD-PSC baseline/ablation with fixed external splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


class RunnerConfigurationError(ValueError):
    pass


# These values define the run identity and the read-only evaluation boundary.
# Variant-owned overrides are trusted; free-form CLI overrides cannot replace
# them, remove them with ``~key``, or redirect writable memory into another run.
RESERVED_EXACT_KEYS = frozenset(
    {
        "fd_psc",
        "planner.fd_psc",
        "fd_psc.enabled",
        "fd_psc.run_mode",
        "fd_psc.seed",
        "seed",
        "hydra.run.dir",
        "fd_psc.canary.manifest_path",
        "fd_psc.canary.every_episodes",
        "fd_psc.canary.rollout_count",
    }
)
RESERVED_PREFIX_KEYS = (
    "fd_psc.external_eval_data",
    "fd_psc.anchor_data",
    "fd_psc.checkpoint",
)
REQUIRED_ONLINE_SPLITS = (
    "calibration",
    "commit_query",
    "plasticity_support",
    "plasticity_query",
    "anchor",
)
ALL_REPORT_SPLITS = REQUIRED_ONLINE_SPLITS + ("report_test",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _override_key(raw: str) -> str:
    lhs = str(raw).split("=", 1)[0].strip()
    while lhs.startswith(("+", "~")):
        lhs = lhs[1:]
    # Hydra config-group overrides may carry an ``@package`` suffix.  The
    # group still controls the reserved subtree and cannot be used as a bypass.
    return lhs.split("@", 1)[0].strip()


def validate_user_overrides(overrides: Sequence[str]) -> None:
    for override in overrides:
        key = _override_key(override)
        reserved = key in RESERVED_EXACT_KEYS or any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in RESERVED_PREFIX_KEYS
        )
        if reserved:
            raise RunnerConfigurationError(
                f"override {override!r} targets reserved run-isolation key {key!r}"
            )


def variant_run_mode(variant: Mapping[str, Any]) -> str:
    if str(variant.get("fd_config")) == "disabled":
        return "disabled"
    mode = "fd_psc"
    for override in variant.get("overrides", ()):
        text = str(override)
        if _override_key(text) == "fd_psc.run_mode" and "=" in text:
            mode = text.split("=", 1)[1].strip()
    return mode


def _split_available(spec: Any) -> bool:
    if not isinstance(spec, Mapping):
        return False
    if spec.get("path"):
        return True
    records = spec.get("records")
    return isinstance(records, Sequence) and not isinstance(records, (str, bytes)) and bool(records)


def inspect_manifest(
    manifest_path: Path,
    *,
    require_report_test: bool,
) -> Tuple[Dict[str, Any], Dict[str, Path], bool]:
    manifest = manifest_path.expanduser().resolve()
    if not manifest.is_file():
        raise RunnerConfigurationError(f"manifest does not exist: {manifest}")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerConfigurationError(f"manifest is not readable JSON: {exc}") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("splits"), Mapping):
        raise RunnerConfigurationError("manifest must contain a splits object")
    splits = data["splits"]
    missing_online = [name for name in REQUIRED_ONLINE_SPLITS if not _split_available(splits.get(name))]
    if missing_online:
        raise RunnerConfigurationError(
            "enabled variant manifest is missing online protocol split(s): "
            + ", ".join(missing_online)
        )
    report_available = _split_available(splits.get("report_test"))
    if require_report_test and not report_available:
        raise RunnerConfigurationError(
            "full FD-PSC requires a non-empty report_test split; commit_query is not a test set"
        )
    split_paths: Dict[str, Path] = {}
    for split_name in ALL_REPORT_SPLITS:
        spec = splits.get(split_name)
        if not isinstance(spec, Mapping) or not spec.get("path"):
            continue
        path = Path(str(spec["path"])).expanduser()
        path = (path if path.is_absolute() else manifest.parent / path).resolve()
        if not path.is_file():
            raise RunnerConfigurationError(
                f"manifest split {split_name!r} path is unreadable: {path}"
            )
        split_paths[split_name] = path
    # Full protocol config validation requires explicit file paths, not inline
    # records.  Baselines may use inline records and report unavailability.
    if require_report_test:
        missing_paths = [name for name in ALL_REPORT_SPLITS if name not in split_paths]
        if missing_paths:
            raise RunnerConfigurationError(
                "full FD-PSC requires file-backed split path(s): " + ", ".join(missing_paths)
            )
    return dict(data), split_paths, report_available


def _reserved_path_overrides(
    *, manifest: Path, split_paths: Mapping[str, Path], state_dir: Path
) -> list[str]:
    def hydra_path(path: Path) -> str:
        # Forward slashes avoid treating Windows backslashes as Hydra escapes.
        return path.expanduser().resolve().as_posix()

    values = [
        f"fd_psc.external_eval_data.manifest_path={hydra_path(manifest)}",
        f"fd_psc.anchor_data.manifest_path={hydra_path(manifest)}",
        f"fd_psc.checkpoint.state_directory={hydra_path(state_dir)}",
        f"fd_psc.checkpoint.latest_pointer_path={hydra_path(state_dir / 'latest.json')}",
        "fd_psc.checkpoint.resume_path=null",
    ]
    mapping = {
        "calibration": "fd_psc.external_eval_data.calibration_path",
        "commit_query": "fd_psc.external_eval_data.commit_query_path",
        "plasticity_support": "fd_psc.external_eval_data.plasticity_support_path",
        "plasticity_query": "fd_psc.external_eval_data.plasticity_query_path",
        "report_test": "fd_psc.external_eval_data.report_test_path",
        "anchor": "fd_psc.anchor_data.data_path",
    }
    values.extend(
        f"{mapping[name]}={hydra_path(path)}"
        for name, path in sorted(split_paths.items())
    )
    return values


def build_command(
    *,
    root: Path,
    variant: Mapping[str, Any],
    plan_config: str,
    seed: int,
    run_dir: Path,
    state_dir: Path,
    user_overrides: Sequence[str],
    manifest: Optional[Path] = None,
    split_paths: Optional[Mapping[str, Path]] = None,
) -> list[str]:
    validate_user_overrides(user_overrides)
    command = [
        sys.executable,
        str(root / "plan.py"),
        "--config-name",
        str(plan_config),
        f"fd_psc={variant['fd_config']}",
        f"seed={int(seed)}",
    ]
    command.extend(str(value) for value in variant.get("overrides", ()))
    command.extend(str(value) for value in user_overrides)
    # Reserved values are deliberately appended last as a second line of
    # defence, after rejecting attempts to target them above.
    if str(variant["fd_config"]) != "disabled":
        if manifest is None:
            raise RunnerConfigurationError("enabled variants require --manifest")
        command.extend(
            _reserved_path_overrides(
                manifest=manifest.resolve(),
                split_paths=split_paths or {},
                state_dir=state_dir.resolve(),
            )
        )
    command.append(f"hydra.run.dir={run_dir.resolve().as_posix()}")
    return command


def _csv_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    report = summary.get("experiment_report") or {}
    report_test = report.get("report_test") or {}
    aggregate = report_test.get("aggregate") or {}
    planning = report.get("planning_metrics") or {}
    by_name = (report.get("algorithm_metrics") or {}).get("by_name", {})

    def metric_mean(name: str) -> Any:
        metric = by_name.get(name) or {}
        return metric.get("mean", metric.get("last"))

    return {
        "variant": summary.get("variant"),
        "seed": summary.get("seed"),
        "plan_config": summary.get("plan_config"),
        "manifest_sha256": summary.get("manifest_sha256"),
        "run_mode": summary.get("run_mode"),
        "status": summary.get("status"),
        "returncode": summary.get("returncode"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "isolated_state_directory": summary.get("isolated_state_directory"),
        "report_test_status": report_test.get("status"),
        "report_test_record_count": aggregate.get("record_count"),
        "report_test_before_jepa_loss": aggregate.get("before_jepa_loss"),
        "report_test_after_jepa_loss": aggregate.get("after_jepa_loss"),
        "report_test_gain": aggregate.get("report_test_gain"),
        "external_calibration_gain": metric_mean("external_calibration_gain"),
        "commit_query_gain": metric_mean("commit_query_gain"),
        "current_jepa_loss": metric_mean("current_jepa_loss"),
        "planning_metrics_json": json.dumps(planning, sort_keys=True, allow_nan=False),
    }


def write_result_artifacts(
    run_dir: Path,
    output_root: Path,
    summary: Mapping[str, Any],
    *,
    append_aggregate: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    row = _csv_row(summary)
    fields = list(row)
    with (run_dir / "result.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    if append_aggregate:
        output_root.mkdir(parents=True, exist_ok=True)
        aggregate_path = output_root / "results.csv"
        with aggregate_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def _safe_run_id(value: Optional[str]) -> str:
    run_id = value or f"{time.time_ns()}-pid{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise RunnerConfigurationError("run-id may contain only letters, digits, '.', '_' and '-'")
    return run_id


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--plan-config", default="adajepa_plan_cem_maze")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=root / "fd_psc_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    try:
        validate_user_overrides(args.overrides)
        matrix_path = root / "conf" / "fd_psc" / "experiments.yaml"
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["variants"]
        if args.variant not in matrix:
            raise RunnerConfigurationError(
                f"unknown variant {args.variant!r}; choose from {', '.join(sorted(matrix))}"
            )
        variant = matrix[args.variant]
        run_mode = variant_run_mode(variant)
        run_id = _safe_run_id(args.run_id)
        output_root = args.output_root.expanduser().resolve()
        run_dir = output_root / f"{args.variant}-seed{args.seed}-{run_id}"
        if run_dir.exists():
            raise RunnerConfigurationError(
                f"isolated run directory already exists; choose a new --run-id: {run_dir}"
            )
        state_dir = run_dir / "memory"

        manifest = None
        manifest_hash = None
        report_available = False
        split_paths: Dict[str, Path] = {}
        if args.manifest is not None:
            manifest = args.manifest.expanduser().resolve()
            _, split_paths, report_available = inspect_manifest(
                manifest,
                require_report_test=run_mode == "fd_psc",
            )
            manifest_hash = sha256(manifest)
        elif variant["fd_config"] != "disabled":
            raise RunnerConfigurationError("enabled variants require --manifest")

        command = build_command(
            root=root,
            variant=variant,
            plan_config=args.plan_config,
            seed=args.seed,
            run_dir=run_dir,
            state_dir=state_dir,
            user_overrides=args.overrides,
            manifest=manifest,
            split_paths=split_paths,
        )
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunnerConfigurationError(
                f"isolated run directory was claimed concurrently: {run_dir}"
            ) from exc
    except RunnerConfigurationError as exc:
        parser.error(str(exc))

    metadata = {
        "schema_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "plan_config": args.plan_config,
        "run_id": run_id,
        "run_mode": run_mode,
        "command": command,
        "manifest": str(manifest) if manifest else None,
        "manifest_sha256": manifest_hash,
        "report_test_declared": report_available,
        "commit_query_is_test_set": False,
        "run_directory": str(run_dir),
        "isolated_state_directory": str(state_dir),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True, allow_nan=False))
    if args.dry_run:
        summary = {
            **metadata,
            "status": "dry_run",
            "returncode": None,
            "elapsed_seconds": 0.0,
            "experiment_report": {
                "status": "not_run",
                "planning_metrics": {},
                "report_test": {
                    "status": "not_run",
                    "aggregate": None,
                    "reason": "dry-run does not execute planning",
                },
            },
        }
        write_result_artifacts(run_dir, output_root, summary, append_aggregate=False)
        return 0

    started = time.time()
    result = subprocess.run(command, cwd=str(root), check=False)
    elapsed = time.time() - started
    report_path = run_dir / "fd_psc_experiment_report.json"
    experiment_report = None
    artifact_error = None
    if report_path.is_file():
        try:
            experiment_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            artifact_error = f"experiment report is unreadable: {exc}"
    elif result.returncode == 0:
        artifact_error = "planning succeeded but fd_psc_experiment_report.json is missing"

    effective_returncode = int(result.returncode)
    if effective_returncode == 0 and artifact_error is not None:
        effective_returncode = 2
    summary = {
        **metadata,
        "status": "completed" if effective_returncode == 0 else "failed",
        "returncode": effective_returncode,
        "process_returncode": int(result.returncode),
        "elapsed_seconds": elapsed,
        "artifact_error": artifact_error,
        "experiment_report": experiment_report,
    }
    write_result_artifacts(run_dir, output_root, summary)
    return effective_returncode


if __name__ == "__main__":
    raise SystemExit(main())
