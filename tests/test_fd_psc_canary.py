from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fd_psc.canary import (
    CanaryEvaluationError,
    CanaryManifest,
    CanaryManifestError,
    CanaryPhase,
    CanaryRunner,
    CanaryScheduler,
    CanaryStatus,
    CanaryUnavailableError,
)
from fd_psc.checkpoint import state_content_hash
from fd_psc.external_data import canonical_json_hash, sha256_file


BASE_HASH = "a" * 64
PREPROCESS_HASH = "b" * 64


def _manifest_value(*, deterministic_reset: bool = True, count: int = 3) -> dict:
    scenarios = []
    for index in range(count):
        payload = {"task": "pick", "reset_state": index}
        scenarios.append(
            {
                "scenario_id": f"scenario-{index}",
                "context_identifier": "pick",
                "seed": 100 + index,
                "payload": payload,
                "content_hash": canonical_json_hash(payload),
                "metadata": {"difficulty": index},
            }
        )
    return {
        "schema_version": 1,
        "base_checkpoint_hash": BASE_HASH,
        "preprocess_hash": PREPROCESS_HASH,
        "environment_id": "mock-resettable-env/v1",
        "deterministic_reset": deterministic_reset,
        "scenarios": scenarios,
        "metadata": {"owner": "unit-test"},
    }


def _write_manifest(directory: Path, **kwargs) -> tuple[Path, dict]:
    value = _manifest_value(**kwargs)
    path = directory / "canary-manifest.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path, value


def _trigger(*, phase=CanaryPhase.PRE_COMMIT):
    scheduler = CanaryScheduler(
        enabled=True,
        every_episodes=10,
        high_risk_rank_expansion=True,
    )
    return scheduler.decide(
        phase=phase,
        episode_count=1,
        commit_sequence=1,
        before_ranks={"layer": 2},
        candidate_ranks={"layer": 4},
    )


class CanaryManifestTests(unittest.TestCase):
    def test_fixed_manifest_identity_and_payload_checksums(self):
        with tempfile.TemporaryDirectory() as raw:
            path, value = _write_manifest(Path(raw))
            expected_hash = canonical_json_hash(value)
            manifest = CanaryManifest.load(
                path,
                expected_base_checkpoint_hash=BASE_HASH,
                expected_manifest_hash=expected_hash,
            )
            self.assertEqual(manifest.manifest_hash, expected_hash)
            self.assertEqual(manifest.file_hash, sha256_file(path))
            self.assertEqual([item.seed for item in manifest.scenarios], [100, 101, 102])

            value["scenarios"][0]["payload"]["reset_state"] = 999
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(CanaryManifestError):
                CanaryManifest.load(path, expected_base_checkpoint_hash=BASE_HASH)

    def test_external_payload_is_checksum_verified_and_snapshotted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            payload_path = directory / "rollout.json"
            payload_path.write_text(json.dumps({"reset_state": 7}), encoding="utf-8")
            value = _manifest_value(count=1)
            value["scenarios"][0].pop("payload")
            value["scenarios"][0].pop("content_hash")
            value["scenarios"][0]["payload_path"] = payload_path.name
            value["scenarios"][0]["sha256"] = sha256_file(payload_path)
            manifest_path = directory / "canary-manifest.json"
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            manifest = CanaryManifest.load(manifest_path)
            payload_path.write_text(json.dumps({"reset_state": 999}), encoding="utf-8")
            self.assertEqual(manifest.scenarios[0].detached_payload(), {"reset_state": 7})

    def test_base_identity_and_commit_query_references_fail_fast(self):
        with tempfile.TemporaryDirectory() as raw:
            path, value = _write_manifest(Path(raw))
            with self.assertRaises(CanaryManifestError):
                CanaryManifest.load(path, expected_base_checkpoint_hash="c" * 64)
            value["commit_query_path"] = "forbidden.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CanaryManifestError, "commit-query"):
                CanaryManifest.load(path)


class CanarySchedulerTests(unittest.TestCase):
    def test_periodic_and_high_risk_decisions_have_distinct_phases(self):
        scheduler = CanaryScheduler(
            enabled=True,
            every_episodes=10,
            high_risk_rank_expansion=True,
        )
        pre = scheduler.decide(
            phase=CanaryPhase.PRE_COMMIT,
            episode_count=9,
            commit_sequence=3,
            before_ranks={"a": 2, "b": 4},
            candidate_ranks={"a": 3, "b": 4},
        )
        self.assertTrue(pre.should_run)
        self.assertEqual(pre.reasons, ("rank_expansion",))
        self.assertTrue(pre.rank_expanded)

        not_periodic = scheduler.decide(
            phase=CanaryPhase.POST_COMMIT,
            episode_count=9,
            commit_sequence=3,
        )
        periodic = scheduler.decide(
            phase=CanaryPhase.POST_COMMIT,
            episode_count=10,
            commit_sequence=4,
        )
        self.assertFalse(not_periodic.should_run)
        self.assertTrue(periodic.should_run)
        self.assertEqual(periodic.reasons, ("periodic",))

    def test_explicit_high_risk_and_exception_merge_trigger_precommit(self):
        scheduler = CanaryScheduler(enabled=True, every_episodes=10)
        decision = scheduler.decide(
            phase=CanaryPhase.PRE_COMMIT,
            episode_count=1,
            commit_sequence=1,
            high_risk_commit=True,
            exception_merge=True,
        )
        self.assertEqual(decision.reasons, ("high_risk_commit", "exception_merge"))
        disabled = CanaryScheduler(enabled=False, every_episodes=10).decide(
            phase=CanaryPhase.PRE_COMMIT,
            episode_count=1,
            commit_sequence=1,
            high_risk_commit=True,
        )
        self.assertFalse(disabled.should_run)


class CanaryRunnerTests(unittest.TestCase):
    def test_paired_budget_fixed_seeds_clones_and_serializable_result(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw))
            manifest = CanaryManifest.load(path)
            before = {"weights": torch.tensor([1.0]), "router": {"uses": 2}}
            candidate = {"weights": torch.tensor([2.0]), "router": {"uses": 2}}
            before_hash = state_content_hash(before)
            candidate_hash = state_content_hash(candidate)
            calls = []

            def evaluate(request):
                # Mutation is confined to the disposable clone passed in the request.
                request.state["weights"].add_(100.0)
                request.state["router"]["uses"] += 1
                request.payload["reset_state"] = -1
                calls.append(
                    (
                        request.pair_id,
                        request.scenario_id,
                        request.seed,
                        request.state_label,
                        request.requires_deterministic_reset,
                    )
                )
                return {
                    "success": True,
                    "metrics": {"return": 1.0 if request.state_label == "before" else 2.0},
                }

            runner = CanaryRunner(
                manifest,
                rollout_count=2,
                evaluator=mock.Mock(side_effect=evaluate),
            )
            result = runner.run(_trigger(), before_state=before, candidate_state=candidate)
            self.assertEqual(result.status, CanaryStatus.PASS)
            self.assertEqual(result.completed_pairs, 2)
            self.assertEqual(len(calls), 4)
            for offset in (0, 2):
                left, right = calls[offset], calls[offset + 1]
                self.assertEqual(left[:3], right[:3])
                self.assertEqual((left[3], right[3]), ("before", "candidate"))
                self.assertTrue(left[4] and right[4])
            self.assertEqual(state_content_hash(before), before_hash)
            self.assertEqual(state_content_hash(candidate), candidate_hash)
            encoded = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)
            self.assertIn('"requested_rollouts_per_state": 2', encoded)
            self.assertNotIn("commit_query", encoded)

    def test_candidate_regression_fails_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw), count=2)
            manifest = CanaryManifest.load(path)

            def evaluate(request):
                return {"success": request.state_label == "before", "metrics": {}}

            result = CanaryRunner(manifest, rollout_count=2, evaluator=evaluate).run(
                _trigger(), before_state={"v": 1}, candidate_state={"v": 2}
            )
            self.assertEqual(result.status, CanaryStatus.FAIL)
            self.assertEqual(result.before_successes, 2)
            self.assertEqual(result.candidate_successes, 0)

    def test_unavailable_policy_reports_unrun_without_success_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw), deterministic_reset=False)
            manifest = CanaryManifest.load(path)
            evaluator = mock.Mock(side_effect=AssertionError("must not run"))
            result = CanaryRunner(
                manifest,
                rollout_count=2,
                evaluator=evaluator,
                unavailable_policy="report_unrun",
            ).run(_trigger(), before_state={"v": 1}, candidate_state={"v": 2})
            self.assertEqual(result.status, CanaryStatus.UNRUN)
            self.assertIsNone(result.before_success_rate)
            self.assertIsNone(result.candidate_success_rate)
            evaluator.assert_not_called()
            json.dumps(result.to_dict(), allow_nan=False)

    def test_callback_unavailable_and_error_policy(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw))
            manifest = CanaryManifest.load(path)
            report = CanaryRunner(
                manifest,
                rollout_count=1,
                evaluator=lambda request: (_ for _ in ()).throw(
                    CanaryUnavailableError("worker cannot reset")
                ),
            ).run(_trigger(), before_state={"v": 1}, candidate_state={"v": 2})
            self.assertEqual(report.status, CanaryStatus.UNRUN)
            self.assertIn("worker cannot reset", report.reason)

            strict = CanaryRunner(
                manifest,
                rollout_count=1,
                evaluator=None,
                unavailable_policy="error",
            )
            with self.assertRaises(CanaryUnavailableError):
                strict.run(_trigger(), before_state={"v": 1}, candidate_state={"v": 2})

    def test_not_triggered_never_invokes_rollout(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw))
            manifest = CanaryManifest.load(path)
            evaluator = mock.Mock()
            trigger = CanaryScheduler(enabled=True, every_episodes=10).decide(
                phase=CanaryPhase.POST_COMMIT,
                episode_count=9,
                commit_sequence=1,
            )
            result = CanaryRunner(manifest, rollout_count=1, evaluator=evaluator).run(
                trigger, before_state={"v": 1}, candidate_state={"v": 2}
            )
            self.assertEqual(result.status, CanaryStatus.NOT_APPLICABLE)
            evaluator.assert_not_called()

    def test_clone_contract_detects_alias_that_mutates_live_state(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _ = _write_manifest(Path(raw))
            manifest = CanaryManifest.load(path)
            before = {"v": 1}

            def evaluator(request):
                request.state["v"] += 1
                return True

            runner = CanaryRunner(
                manifest,
                rollout_count=1,
                evaluator=evaluator,
                clone_state=lambda state: state,
            )
            with self.assertRaisesRegex(CanaryEvaluationError, "sharing mutable objects"):
                runner.run(_trigger(), before_state=before, candidate_state={"v": 2})
            self.assertEqual(before, {"v": 1})


if __name__ == "__main__":
    unittest.main()
