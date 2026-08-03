from __future__ import annotations

import csv
import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from fd_psc.experiment_reporting import (
    ReportTestRequiredError,
    begin_report_test,
    build_required_metric_contract,
    finish_report_test,
    summarize_metric_events,
    write_experiment_report,
)
from fd_psc.external_data import ExternalDataError
from fd_psc.metrics import REQUIRED_METRIC_NAMES


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_fd_psc_experiment", ROOT / "scripts" / "run_fd_psc_experiment.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_reserved_overrides_cannot_be_replaced_or_deleted(self):
        bad = (
            "fd_psc.external_eval_data.report_test_path=other.json",
            "+fd_psc.checkpoint.state_directory=shared",
            "~fd_psc.anchor_data",
            "fd_psc@planner.fd_psc=disabled",
            "hydra.run.dir=elsewhere",
            "seed=99",
        )
        for value in bad:
            with self.subTest(value=value), self.assertRaises(runner.RunnerConfigurationError):
                runner.validate_user_overrides([value])
        runner.validate_user_overrides(["fd_psc.gates.drift_tolerance=0.1"])

    def test_reserved_paths_are_last_and_state_is_run_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            split = root / "report.json"
            split.write_text("[]", encoding="utf-8")
            run_dir = root / "run"
            state_dir = run_dir / "memory"
            command = runner.build_command(
                root=ROOT,
                variant={"fd_config": "default", "overrides": ["fd_psc.slice.enabled=false"]},
                plan_config="plan_gd",
                seed=7,
                run_dir=run_dir,
                state_dir=state_dir,
                user_overrides=["planner.adapt.lr=0.001"],
                manifest=manifest,
                split_paths={"report_test": split},
            )
            user_index = command.index("planner.adapt.lr=0.001")
            report_override = next(
                value
                for value in command
                if value.startswith("fd_psc.external_eval_data.report_test_path=")
            )
            report_index = command.index(report_override)
            self.assertTrue(Path(report_override.split("=", 1)[1]).samefile(split))
            self.assertGreater(report_index, user_index)
            self.assertEqual(command[-1], f"hydra.run.dir={run_dir.resolve().as_posix()}")
            self.assertIn(
                f"fd_psc.checkpoint.state_directory={state_dir.resolve().as_posix()}", command
            )
            self.assertIn("fd_psc.checkpoint.resume_path=null", command)

    def _manifest(self, directory: Path, include_report: bool) -> Path:
        names = list(runner.REQUIRED_ONLINE_SPLITS)
        if include_report:
            names.append("report_test")
        splits = {}
        for name in names:
            path = directory / f"{name}.json"
            path.write_text("[]", encoding="utf-8")
            splits[name] = {"path": path.name, "sha256": "0" * 64}
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"splits": splits}), encoding="utf-8")
        return manifest

    def test_full_protocol_requires_report_test_but_baseline_records_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory), include_report=False)
            with self.assertRaises(runner.RunnerConfigurationError):
                runner.inspect_manifest(manifest, require_report_test=True)
            _, paths, available = runner.inspect_manifest(
                manifest, require_report_test=False
            )
            self.assertFalse(available)
            self.assertNotIn("report_test", paths)

    def test_each_run_gets_matching_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            summary = {
                "variant": "x",
                "seed": 1,
                "plan_config": "p",
                "manifest_sha256": "a" * 64,
                "run_mode": "fd_psc",
                "status": "completed",
                "returncode": 0,
                "elapsed_seconds": 1.25,
                "isolated_state_directory": str(run_dir / "memory"),
                "experiment_report": {
                    "planning_metrics": {"final_eval/success": 0.75},
                    "report_test": {
                        "status": "evaluated",
                        "aggregate": {
                            "record_count": 4,
                            "before_jepa_loss": 2.0,
                            "after_jepa_loss": 1.5,
                            "report_test_gain": 0.5,
                        },
                    },
                },
            }
            runner.write_result_artifacts(run_dir, root, summary)
            loaded = json.loads((run_dir / "result.json").read_text("utf-8"))
            self.assertEqual(loaded["experiment_report"]["report_test"]["aggregate"]["report_test_gain"], 0.5)
            with (run_dir / "result.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["report_test_status"], "evaluated")
            self.assertEqual(float(row["report_test_gain"]), 0.5)


class _FakeInjection:
    def __init__(self):
        self.calls = 0

    def enforce_frozen_base_eval(self):
        self.calls += 1


class _FakeExternal:
    manifest_hash = "external"

    def __init__(self, system, missing=False):
        self.system = system
        self.missing = missing

    def report_test(self, context):
        if self.missing:
            raise ExternalDataError("split report_test has no records")
        return (SimpleNamespace(context=context), SimpleNamespace(context=context))


class _FakeTrainer:
    def __init__(self, system):
        self.system = system

    def evaluate_external_records(self, records):
        self.assert_records = len(records)
        return 2.0 if self.system.adapters_disabled else 99.0


class _FakeMetrics:
    def __init__(self):
        self.sequence = 0
        self._events = []

    def state_dict(self):
        return {"sequence": self.sequence, "counters": {}}

    def events(self):
        return tuple(self._events)

    def mutate(self):
        self.sequence += 1
        self._events.append((self.sequence, "report_test_gain"))


class _FakeSystem:
    def __init__(self, mode="fd_psc", missing=False, mutate_metrics=False):
        self.config = SimpleNamespace(run_mode=mode)
        self.injection = _FakeInjection()
        self.external = _FakeExternal(self, missing=missing)
        self.target_manifest = SimpleNamespace(hash="target")
        self.base_checkpoint_hash = "base"
        self._commit_sequence = 0
        self.calls = {}
        self.missing = missing
        self.guard = {"persistent": 3}
        self.adapters_disabled = False
        self.metrics = _FakeMetrics()
        self.mutate_metrics = mutate_metrics

    def resolve_context_identifier(self, metadata):
        return f"ctx-{metadata['sample_idx'] % 2}"

    def assert_base_frozen(self):
        if self.guard != {"persistent": 3}:
            raise AssertionError("report evaluation mutated persistent state")

    def materialize_external_payload(self, record):
        return {"context": record.context}

    @contextlib.contextmanager
    def _preserve_adapter_runtime(self):
        value = self.adapters_disabled
        training = self.wm.training
        try:
            yield
        finally:
            self.adapters_disabled = value
            self.wm.train(training)

    @contextlib.contextmanager
    def _all_adapters_disabled(self):
        value = self.adapters_disabled
        self.adapters_disabled = True
        try:
            yield
        finally:
            self.adapters_disabled = value

    def evaluate_report_test(self, trainer, context):
        if self.mutate_metrics:
            self.metrics.mutate()
        count = self.calls.get(context, 0)
        self.calls[context] = count + 1
        return {
            "schema_version": 1,
            "context_identifier": context,
            "record_count": 2,
            "jepa_loss": 1.25,
            "theta0_jepa_loss": 2.0,
            "report_test_gain": 0.75,
            "base_checkpoint_hash": "base",
            "external_manifest_hash": "external",
            "target_manifest_hash": "target",
            "commit_sequence": self._commit_sequence,
            "run_mode": self.config.run_mode,
        }


class ReportLifecycleTests(unittest.TestCase):
    def _planner(self, system):
        wm = torch.nn.Linear(2, 2)
        system.wm = wm
        return SimpleNamespace(
            fd_psc_system=system,
            adajepa_trainer=_FakeTrainer(system),
            wm=wm,
        )

    def test_report_test_reference_and_final_are_evaluated_only_for_report(self):
        system = _FakeSystem()
        planner = self._planner(system)
        session = begin_report_test(planner, [10, 11, 12])
        metrics_before = system.metrics.state_dict(), system.metrics.events()
        section = finish_report_test(planner, session)
        self.assertEqual(section["status"], "evaluated")
        self.assertEqual(len(section["contexts"]), 2)
        self.assertAlmostEqual(section["aggregate"]["report_test_gain"], 0.75)
        self.assertFalse(section["report_test_influenced_algorithm"])
        self.assertFalse(section["commit_query_is_test_set"])
        self.assertEqual(system.guard, {"persistent": 3})
        self.assertTrue(planner.wm.training)
        self.assertEqual(system.calls, {"ctx-0": 1, "ctx-1": 1})
        self.assertEqual(metrics_before, (system.metrics.state_dict(), system.metrics.events()))

    def test_live_metrics_mutation_is_rejected(self):
        system = _FakeSystem(mutate_metrics=True)
        planner = self._planner(system)
        session = begin_report_test(planner, [10])
        with self.assertRaisesRegex(ReportTestRequiredError, "mutated live metrics"):
            finish_report_test(planner, session)

    def test_missing_report_is_fatal_only_for_full_protocol(self):
        full = self._planner(_FakeSystem(mode="fd_psc", missing=True))
        with self.assertRaises(ReportTestRequiredError):
            begin_report_test(full, [1])

        baseline = self._planner(_FakeSystem(mode="plain_svd", missing=True))
        session = begin_report_test(baseline, [1])
        section = finish_report_test(baseline, session)
        self.assertEqual(section["status"], "unavailable")
        self.assertIn("not supplied", section["reason"])

    def test_disabled_baseline_is_explicitly_not_applicable(self):
        planner = SimpleNamespace(fd_psc_system=None)
        section = finish_report_test(planner, begin_report_test(planner, [1]))
        self.assertEqual(section["status"], "not_applicable")
        self.assertIn("rollout", section["reason"])

    def test_workspace_report_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            report = write_experiment_report(
                Path(directory),
                planning_metrics={"success": torch.tensor(1.0)},
                report_test={"status": "not_applicable", "aggregate": None},
            )
            self.assertEqual(report["planning_metrics"]["success"], 1.0)
            self.assertEqual(
                set(report["metric_contract"]["metrics"]),
                REQUIRED_METRIC_NAMES,
            )
            self.assertEqual(
                report["metric_contract"]["classified_metric_count"],
                len(REQUIRED_METRIC_NAMES),
            )
            self.assertEqual(
                report["metric_contract"]["metrics"]["planning_success"],
                {
                    "status": "available",
                    "value": 1.0,
                    "reason": None,
                    "source": "planning_metrics",
                },
            )
            self.assertEqual(
                report["metric_contract"]["metrics"]["report_test_gain"][
                    "status"
                ],
                "not_applicable",
            )
            self.assertEqual(
                report["metric_contract"]["metrics"]["slow_rank"]["value"],
                None,
            )
            self.assertIn(
                "not emitted",
                report["metric_contract"]["metrics"]["slow_rank"]["reason"],
            )
            self.assertTrue((Path(directory) / "fd_psc_experiment_report.json").is_file())
            self.assertTrue((Path(directory) / "fd_psc_experiment_report.csv").is_file())

            with (Path(directory) / "fd_psc_experiment_report.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["required_metric_coverage_status"], "partial")
            self.assertGreater(int(row["required_metric_unavailable_count"]), 0)
            self.assertEqual(
                set(json.loads(row["required_metric_contract_json"])["metrics"]),
                REQUIRED_METRIC_NAMES,
            )

    def test_metric_summary_keeps_global_and_context_statistics(self):
        events = [
            SimpleNamespace(
                name="commit_query_gain",
                value=0.25,
                context_identifier="a",
                tags={"status": "available"},
            ),
            SimpleNamespace(
                name="commit_query_gain",
                value=0.75,
                context_identifier="b",
                tags={"status": "available"},
            ),
            SimpleNamespace(
                name="episode_terminal",
                value="COMMIT_SLOW",
                context_identifier="b",
                tags={},
            ),
        ]
        summary = summarize_metric_events(events)
        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["by_name"]["commit_query_gain"]["mean"], 0.5)
        self.assertEqual(
            summary["by_name"]["commit_query_gain"]["status_counts"],
            {"available": 2},
        )
        self.assertEqual(
            summary["by_context"]["b"]["episode_terminal"]["last"],
            "COMMIT_SLOW",
        )

    def test_required_metric_contract_preserves_nullable_status_and_sources(self):
        contract = build_required_metric_contract(
            planning_metrics={"final_eval/success_rate": 0.625},
            report_test={
                "status": "evaluated",
                "aggregate": {"report_test_gain": 0.125},
            },
            algorithm_metrics={
                "by_name": {
                    "current_jepa_loss": {
                        "last": 1.5,
                        "mean": 1.25,
                        "last_status": "available",
                        "last_reason": None,
                    },
                    "anchor_loss": {
                        "last": None,
                        "last_status": "not_applicable",
                        "last_reason": "cold start",
                    },
                    "historical_replay_loss": {
                        "last": None,
                        "last_status": "insufficient_observations",
                        "last_reason": "one committed context",
                    },
                }
            },
        )
        self.assertEqual(set(contract["metrics"]), REQUIRED_METRIC_NAMES)
        self.assertEqual(
            contract["metrics"]["current_jepa_loss"]["value"], 1.25
        )
        self.assertEqual(
            contract["metrics"]["anchor_loss"],
            {
                "status": "not_applicable",
                "value": None,
                "reason": "cold start",
                "source": "algorithm_metrics",
            },
        )
        self.assertEqual(
            contract["metrics"]["planning_success"]["source"],
            "planning_metrics",
        )
        self.assertEqual(
            contract["metrics"]["report_test_gain"]["source"],
            "report_test",
        )
        self.assertEqual(contract["coverage_status"], "partial")
        self.assertEqual(
            contract["status_counts"]["insufficient_observations"], 1
        )
        self.assertEqual(contract["observed_or_not_applicable_count"], 4)


if __name__ == "__main__":
    unittest.main()
