"""Read-only report-test evaluation and stable experiment artifacts.

The online protocol is deliberately unaware of report-test values.  This
module is called by the outer planning workspace only: it validates split
availability before planning, then evaluates theta_0 and the final memory
state after planning.  Neither value is returned to candidate selection or
commit logic.
"""

from __future__ import annotations

import csv
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .external_data import ExternalDataError


class ReportTestRequiredError(RuntimeError):
    """Raised when a full FD-PSC run cannot produce its final report-test."""


@dataclass
class ReportTestSession:
    """Opaque pre-planning state used only to produce the final report."""

    status: str
    run_mode: str
    contexts: Tuple[str, ...] = ()
    reason: Optional[str] = None


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def summarize_metric_events(events: Sequence[Any]) -> Dict[str, Any]:
    """Compact the full JSONL event stream for the single-row run report."""

    grouped: Dict[str, list[Any]] = {}
    by_context: Dict[str, Dict[str, list[Any]]] = {}
    for event in events:
        name = str(event.name)
        value = _json_safe(event.value)
        grouped.setdefault(name, []).append(value)
        context = getattr(event, "context_identifier", None)
        if context is not None:
            by_context.setdefault(str(context), {}).setdefault(name, []).append(value)

    def compact(values: Sequence[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"count": len(values), "last": values[-1]}
        if values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            numbers = [float(value) for value in values]
            result.update(
                {
                    "mean": sum(numbers) / len(numbers),
                    "minimum": min(numbers),
                    "maximum": max(numbers),
                }
            )
        return result

    return {
        "event_count": sum(len(values) for values in grouped.values()),
        "by_name": {
            name: compact(values) for name, values in sorted(grouped.items())
        },
        "by_context": {
            context: {
                name: compact(values) for name, values in sorted(metrics.items())
            }
            for context, metrics in sorted(by_context.items())
        },
    }


def _planner_components(planner: Any) -> Tuple[Any, Any, Any]:
    system = getattr(planner, "fd_psc_system", None)
    trainer = getattr(planner, "adajepa_trainer", None)
    wm = getattr(planner, "wm", None)
    return system, trainer, wm


def _contexts(system: Any, eval_seeds: Sequence[int]) -> Tuple[str, ...]:
    values = {
        str(
            system.resolve_context_identifier(
                {"sample_idx": int(index), "seed": int(seed)}
            )
        )
        for index, seed in enumerate(eval_seeds)
    }
    if not values:
        raise ReportTestRequiredError("report-test requires at least one evaluation context")
    return tuple(sorted(values))


def _evaluate_contexts(
    system: Any,
    trainer: Any,
    wm: Any,
    contexts: Sequence[str],
    *,
    accumulated_pilot: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate fixed records in eval mode and verify theta_0 afterwards."""

    if trainer is None or wm is None:
        raise ReportTestRequiredError("enabled FD-PSC planner has no trainer/world model")
    system.assert_base_frozen()
    metrics_state = copy.deepcopy(system.metrics.state_dict())
    metric_events = system.metrics.events()
    reports: Dict[str, Dict[str, Any]] = {}
    with system._preserve_adapter_runtime():
        wm.eval()
        injection = getattr(system, "injection", None)
        if injection is not None:
            injection.enforce_frozen_base_eval()
        for context in contexts:
            if accumulated_pilot:
                # The accumulate baseline's learned adapter lives in Pilot by
                # design.  FDPSCSystem.evaluate_report_test intentionally clears
                # episodic branches, so evaluate the same guarded registry records
                # directly for this one baseline and label the path explicitly.
                if system.external is None:
                    raise ExternalDataError("report-test requires the fixed external registry")
                records = system.external.report_test(str(context))
                loss = trainer.evaluate_external_records(records)
                reports[str(context)] = {
                    "schema_version": 1,
                    "context_identifier": str(context),
                    "record_count": len(records),
                    "jepa_loss": float(loss),
                    "base_checkpoint_hash": system.base_checkpoint_hash,
                    "external_manifest_hash": system.external.manifest_hash,
                    "target_manifest_hash": system.target_manifest.hash,
                    "commit_sequence": int(system._commit_sequence),
                    "run_mode": system.config.run_mode,
                    "evaluation_path": "persistent_accumulated_pilot",
                }
            else:
                reports[str(context)] = dict(
                    system.evaluate_report_test(trainer, str(context))
                )
                reports[str(context)]["evaluation_path"] = "FDPSCSystem.evaluate_report_test"
    system.assert_base_frozen()
    if system.metrics.state_dict() != metrics_state or system.metrics.events() != metric_events:
        raise ReportTestRequiredError("report-test evaluation mutated live metrics state")
    return reports


def _theta0_from_public_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not all("theta0_jepa_loss" in report for report in reports.values()):
        return None
    result: Dict[str, Dict[str, Any]] = {}
    for context, report in reports.items():
        result[str(context)] = {
            "schema_version": 1,
            "context_identifier": str(context),
            "record_count": int(report["record_count"]),
            "jepa_loss": float(report["theta0_jepa_loss"]),
            "base_checkpoint_hash": report["base_checkpoint_hash"],
            "external_manifest_hash": report["external_manifest_hash"],
            "target_manifest_hash": report["target_manifest_hash"],
            "commit_sequence": int(report["commit_sequence"]),
            "run_mode": report["run_mode"],
            "evaluation_path": "FDPSCSystem.evaluate_report_test.theta0",
        }
    return result


def _evaluate_theta0_contexts(
    system: Any,
    trainer: Any,
    wm: Any,
    contexts: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate theta_0 at reporting time with every adapter disabled."""

    if trainer is None or wm is None or system.external is None:
        raise ReportTestRequiredError("theta_0 report-test requires trainer/model/registry")
    system.assert_base_frozen()
    reports: Dict[str, Dict[str, Any]] = {}
    with system._preserve_adapter_runtime():
        wm.eval()
        system.injection.enforce_frozen_base_eval()
        for context in contexts:
            with system._all_adapters_disabled():
                records = system.external.report_test(str(context))
                loss = trainer.evaluate_external_records(records)
            reports[str(context)] = {
                "schema_version": 1,
                "context_identifier": str(context),
                "record_count": len(records),
                "jepa_loss": float(loss),
                "base_checkpoint_hash": system.base_checkpoint_hash,
                "external_manifest_hash": system.external.manifest_hash,
                "target_manifest_hash": system.target_manifest.hash,
                "commit_sequence": int(system._commit_sequence),
                "run_mode": system.config.run_mode,
                "evaluation_path": "theta0_all_adapters_disabled",
            }
    system.assert_base_frozen()
    return reports


def begin_report_test(planner: Any, eval_seeds: Sequence[int]) -> ReportTestSession:
    """Validate report-test availability without evaluating its model loss.

    A disabled FD-PSC baseline has no fixed-split registry, so its report-test
    field is explicitly not applicable; its independent environment rollout
    remains available in the planning metrics.  Missing report-test is fatal
    for the complete ``fd_psc`` protocol and is recorded as unavailable for a
    baseline configuration.
    """

    system, trainer, wm = _planner_components(planner)
    if system is None:
        return ReportTestSession(
            status="not_applicable",
            run_mode="disabled",
            reason=(
                "FD-PSC is disabled for this baseline; use the independent "
                "planning rollout metric."
            ),
        )
    run_mode = str(system.config.run_mode)
    contexts = _contexts(system, eval_seeds)
    try:
        if system.external is None:
            raise ExternalDataError("report-test requires the fixed external registry")
        # Integrity is fail-fast, but losses are not computed until the final
        # offline report.  materialize_external_payload validates detached
        # payload files/checksums and its return value is deliberately dropped.
        for context in contexts:
            for record in system.external.report_test(context):
                system.materialize_external_payload(record)
    except ExternalDataError as exc:
        if run_mode == "fd_psc":
            raise ReportTestRequiredError(
                f"full FD-PSC requires report-test for every evaluation context: {exc}"
            ) from exc
        return ReportTestSession(
            status="unavailable",
            run_mode=run_mode,
            contexts=contexts,
            reason=f"baseline report-test was not supplied: {exc}",
        )
    return ReportTestSession(
        status="pending",
        run_mode=run_mode,
        contexts=contexts,
    )


def _weighted_loss(reports: Mapping[str, Mapping[str, Any]]) -> Tuple[float, int]:
    total = 0.0
    records = 0
    for report in reports.values():
        count = int(report["record_count"])
        if count <= 0:
            raise ReportTestRequiredError("report-test context contains no records")
        total += float(report["jepa_loss"]) * count
        records += count
    if records <= 0:
        raise ReportTestRequiredError("report-test contains no records")
    return total / records, records


def finish_report_test(planner: Any, session: ReportTestSession) -> Dict[str, Any]:
    """Evaluate the final state and build the isolated report-test section."""

    if session.status != "pending":
        return {
            "status": session.status,
            "run_mode": session.run_mode,
            "contexts": [],
            "aggregate": None,
            "reason": session.reason,
            "report_test_influenced_algorithm": False,
            "commit_query_is_test_set": False,
        }

    system, trainer, wm = _planner_components(planner)
    if system is None:
        raise ReportTestRequiredError("FD-PSC disappeared before final report-test")
    try:
        after = _evaluate_contexts(
            system,
            trainer,
            wm,
            session.contexts,
            accumulated_pilot=session.run_mode == "accumulate",
        )
        before = _theta0_from_public_reports(after)
        if before is None:
            before = _evaluate_theta0_contexts(
                system,
                trainer,
                wm,
                session.contexts,
            )
    except ExternalDataError as exc:
        if session.run_mode == "fd_psc":
            raise ReportTestRequiredError(
                f"full FD-PSC final report-test failed: {exc}"
            ) from exc
        return {
            "status": "unavailable",
            "run_mode": session.run_mode,
            "contexts": [],
            "aggregate": None,
            "reason": f"baseline final report-test failed: {exc}",
            "report_test_influenced_algorithm": False,
            "commit_query_is_test_set": False,
        }

    context_rows = []
    for context in session.contexts:
        reference = before[context]
        final = after[context]
        if int(reference["record_count"]) != int(final["record_count"]):
            raise ReportTestRequiredError(
                f"report-test record count changed for context {context!r}"
            )
        context_rows.append(
            {
                "context_identifier": context,
                "record_count": int(final["record_count"]),
                "before_jepa_loss": float(reference["jepa_loss"]),
                "after_jepa_loss": float(final["jepa_loss"]),
                "report_test_gain": float(reference["jepa_loss"]) - float(final["jepa_loss"]),
                "before": reference,
                "after": final,
            }
        )
    before_loss, before_count = _weighted_loss(before)
    after_loss, after_count = _weighted_loss(after)
    if before_count != after_count:
        raise ReportTestRequiredError("report-test aggregate record count changed")
    return {
        "status": "evaluated",
        "run_mode": session.run_mode,
        "contexts": context_rows,
        "aggregate": {
            "record_count": after_count,
            "before_jepa_loss": before_loss,
            "after_jepa_loss": after_loss,
            "report_test_gain": before_loss - after_loss,
        },
        "reason": None,
        "external_manifest_hash": next(iter(after.values()))["external_manifest_hash"],
        "base_checkpoint_hash": next(iter(after.values()))["base_checkpoint_hash"],
        "target_manifest_hash": next(iter(after.values()))["target_manifest_hash"],
        "report_test_influenced_algorithm": False,
        "commit_query_is_test_set": False,
    }


def write_experiment_report(
    output_dir: Path,
    *,
    planning_metrics: Mapping[str, Any],
    report_test: Mapping[str, Any],
    algorithm_metrics: Optional[Mapping[str, Any]] = None,
    metric_artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the same single-run result to stable JSON and one-row CSV."""

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "protocol": "FD-PSC-V2",
        "status": "completed",
        "planning_metrics": _json_safe(planning_metrics),
        "algorithm_metrics": _json_safe(algorithm_metrics or {}),
        "metric_artifacts": _json_safe(metric_artifacts or {}),
        "report_test": _json_safe(report_test),
    }
    (target / "fd_psc_experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    aggregate = report["report_test"].get("aggregate") or {}
    by_name = report["algorithm_metrics"].get("by_name", {})

    def metric_mean(name: str) -> Any:
        metric = by_name.get(name) or {}
        return metric.get("mean", metric.get("last"))

    row = {
        "schema_version": report["schema_version"],
        "protocol": report["protocol"],
        "status": report["status"],
        "report_test_status": report["report_test"].get("status"),
        "report_test_record_count": aggregate.get("record_count"),
        "report_test_before_jepa_loss": aggregate.get("before_jepa_loss"),
        "report_test_after_jepa_loss": aggregate.get("after_jepa_loss"),
        "report_test_gain": aggregate.get("report_test_gain"),
        "external_calibration_gain": metric_mean("external_calibration_gain"),
        "commit_query_gain": metric_mean("commit_query_gain"),
        "current_jepa_loss": metric_mean("current_jepa_loss"),
        "planning_success": next(
            (
                value
                for key, value in report["planning_metrics"].items()
                if str(key).endswith("/success") or str(key).endswith("/success_rate")
            ),
            None,
        ),
        "planning_metrics_json": json.dumps(
            report["planning_metrics"], sort_keys=True, allow_nan=False
        ),
    }
    with (target / "fd_psc_experiment_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return report


__all__ = [
    "ReportTestRequiredError",
    "ReportTestSession",
    "begin_report_test",
    "finish_report_test",
    "summarize_metric_events",
    "write_experiment_report",
]
