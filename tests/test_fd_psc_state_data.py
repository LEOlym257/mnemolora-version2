from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fd_psc.checkpoint import (
    CheckpointError,
    CheckpointValidationError,
    SidecarCheckpointManager,
    state_content_hash,
)
from fd_psc.commit_gates import (
    CommitGateError,
    CommitGateEvaluator,
    CommitGateInputs,
    GateStatus,
)
from fd_psc.config import GatesConfig
from fd_psc.diagnostics import Diagnostics, assert_finite_tree, bitwise_state_equal
from fd_psc.exception_router import ExceptionRouter, ExceptionRouterError
from fd_psc.external_data import (
    CommitQueryAccessError,
    DataLeakageError,
    ExternalDataRegistry,
    ManifestSchemaError,
    canonical_json_hash,
)
from fd_psc.metrics import MetricsError, StructuredMetrics
from fd_psc.repair import RepairEngine, ScreeningResult
from fd_psc.replay_memory import ClusterBalancedReplay, GRASPSampler, ReplayError, ReplayWindow
from fd_psc.state_machine import (
    FDPSCState,
    FDPSCStateMachine,
    FinalProposal,
    ProposalType,
    StateMachineError,
)
from fd_psc.transaction import StateTransaction
from scripts.generate_fd_psc_manifest import audit as audit_manifest_records


HASH_A = "a" * 64
HASH_B = "b" * 64


def _record(split: str, index: int, context: str = "ctx") -> dict:
    payload = {"split": split, "index": index}
    return {
        "record_id": f"{split}-{index}",
        "context_identifier": context,
        "trajectory_id": f"trajectory-{split}-{index}",
        "transition_ids": [f"transition-{split}-{index}-0"],
        "frame_ids": [f"frame-{split}-{index}-0", f"frame-{split}-{index}-1"],
        "content_hash": canonical_json_hash(payload),
        "payload": payload,
    }


def _manifest(path: Path, *, duplicate: bool = False) -> Path:
    splits = {}
    contexts = {"ctx": {}}
    names = (
        "calibration",
        "commit_query",
        "plasticity_support",
        "plasticity_query",
        "report_test",
        "anchor",
    )
    for index, name in enumerate(names):
        record = _record(name, index)
        if duplicate and name == "commit_query":
            record["trajectory_id"] = "trajectory-calibration-0"
        splits[name] = {"records": [record]}
        contexts["ctx"][name] = [record["record_id"]]
    value = {
        "schema_version": 1,
        "base_checkpoint_hash": HASH_A,
        "preprocess_hash": HASH_B,
        "latent_adapter_schema": "mock-latent/v1",
        "splits": splits,
        "contexts": contexts,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _window(index: int, embedding=(1.0, 0.0), *, episode="ep", committed=True) -> ReplayWindow:
    return ReplayWindow(
        window_id=f"window-{index}",
        trajectory_id=f"trajectory-support-{index}",
        transition_ids=(f"support-transition-{index}",),
        frame_ids=(f"support-frame-{index}-0", f"support-frame-{index}-1"),
        timesteps=(index * 2, index * 2 + 1),
        content_hash=hashlib.sha256(f"window-{index}".encode()).hexdigest(),
        context_identifier="ctx",
        context_embedding=embedding,
        visual_latent=torch.tensor([index], dtype=torch.float16),
        proprio=torch.tensor([index], dtype=torch.float32),
        actions=torch.tensor([index], dtype=torch.float32),
        residual=torch.tensor([float(index)]),
        source_episode=episode,
        committed=committed,
        difficulty_score=float(index),
    )


class ExternalDataTests(unittest.TestCase):
    def test_episode_context_mapping_is_validated_at_registry_startup(self):
        invalid_cases = (
            ([], "manifest.episode_contexts must be an object"),
            ({"": "ctx"}, "keys must be non-empty strings"),
            ({"sample:0": ""}, "values must be non-empty strings"),
            ({"sample:0": 3}, "values must be non-empty strings"),
            ({"sample:0": "missing"}, "names unknown context"),
            ({" sample:0": "ctx", "sample:0": "ctx"}, "duplicate normalized key"),
        )
        for episode_contexts, message in invalid_cases:
            with self.subTest(value=episode_contexts), tempfile.TemporaryDirectory() as raw:
                manifest_path = _manifest(Path(raw) / "manifest.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["episode_contexts"] = episode_contexts
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ManifestSchemaError, message):
                    ExternalDataRegistry(manifest_path)

        with tempfile.TemporaryDirectory() as raw:
            manifest_path = _manifest(Path(raw) / "manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["episode_contexts"] = {"sample:7": "ctx"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            registry = ExternalDataRegistry(manifest_path)
            self.assertEqual(registry.episode_contexts, {"sample:7": "ctx"})

    def test_declared_manifest_content_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest_path = _manifest(Path(raw) / "manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["manifest_content_hash"] = "e" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ManifestSchemaError, "manifest content hash mismatch"):
                ExternalDataRegistry(manifest_path)

    def test_split_leakage_fails_fast(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _manifest(Path(raw) / "manifest.json", duplicate=True)
            with self.assertRaises(DataLeakageError):
                ExternalDataRegistry(path)

    def test_payload_path_fake_hash_cannot_hide_cross_split_file_reuse(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload_path = root / "shared.json"
            payload = {"z": [[[[1.0]]]], "actions": [[0.0]]}
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            actual_hash = canonical_json_hash(payload)

            manifest_path = _manifest(root / "manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            calibration = manifest["splits"]["calibration"]["records"][0]
            commit_query = manifest["splits"]["commit_query"]["records"][0]
            for record in (calibration, commit_query):
                record.pop("payload")
                record["payload_path"] = payload_path.name
            calibration["content_hash"] = actual_hash
            commit_query["content_hash"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ManifestSchemaError, "payload hash mismatch"):
                ExternalDataRegistry(manifest_path)

            records = {
                split: spec["records"] for split, spec in manifest["splits"].items()
            }
            with self.assertRaisesRegex(ValueError, "payload content hash mismatch"):
                audit_manifest_records(records, payload_root=root)

    def test_support_audit_and_single_use_commit_query(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ExternalDataRegistry(_manifest(Path(raw) / "manifest.json"))
            registry.begin_episode("ep-1", "ctx")
            support = {
                "record_id": "support-1",
                "context_identifier": "ctx",
                "trajectory_id": "support-trajectory",
                "transition_ids": ["support-transition"],
                "frame_ids": ["support-frame-0", "support-frame-1"],
                "content_hash": "c" * 64,
            }
            registry.audit_and_register_support(support)
            registry.seal_support_for_online_update()
            with self.assertRaises(Exception):
                registry.audit_and_register_support({**support, "record_id": "late"})
            self.assertEqual(len(registry.calibration()), 1)
            with self.assertRaises(CommitQueryAccessError):
                registry._context_records("commit_query", "ctx")
            token = registry.issue_commit_query_token("ep-1", "proposal-1")
            query = registry.consume_commit_query(token, proposal_id="proposal-1")
            self.assertEqual(len(query), 1)
            self.assertEqual(registry.commit_query_access_count("ep-1"), 1)
            with self.assertRaises(CommitQueryAccessError):
                registry.consume_commit_query(token, proposal_id="proposal-1")
            with self.assertRaises(CommitQueryAccessError):
                registry.issue_commit_query_token("ep-1", "proposal-2")
            registry.end_episode()

    def test_support_external_overlap_fails_before_update(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ExternalDataRegistry(_manifest(Path(raw) / "manifest.json"))
            registry.begin_episode("ep-1", "ctx")
            calibration = registry.calibration()[0].identity
            support = {
                "record_id": "bad-support",
                "context_identifier": "ctx",
                "trajectory_id": calibration.trajectory_ids[0],
                "transition_ids": ["support-only-transition"],
                "frame_ids": ["support-only-frame"],
                "content_hash": "d" * 64,
            }
            with self.assertRaises(DataLeakageError):
                registry.audit_and_register_support(support)


class StateMachineTests(unittest.TestCase):
    def test_single_proposal_and_query_lifecycle(self):
        machine = FDPSCStateMachine()
        machine.begin_episode("ep", "ctx")
        machine.note_support_window()
        machine.note_online_update()
        machine.activate_centered()
        machine.enter_sleep()
        proposal = FinalProposal("p1", ProposalType.GLOBAL_SLOW, {"rank": 8})
        machine.set_final_proposal(proposal)
        machine.begin_final_gate("p1", "token")
        machine.commit_slow()
        machine.finish_episode()
        self.assertEqual(machine.state, FDPSCState.IDLE)
        self.assertEqual(machine.persistent_commit_count, 1)

    def test_no_proposal_never_enters_query_and_abort_rolls_back(self):
        machine = FDPSCStateMachine()
        machine.begin_episode("ep", "ctx")
        machine.enter_sleep()
        machine.reject_no_proposal("zero_task_vector")
        with self.assertRaises(StateMachineError):
            machine.begin_final_gate("none", "token")
        machine.finish_episode()
        machine.begin_episode("ep2", "ctx")
        machine.abort("planner_exception")
        self.assertEqual(machine.state, FDPSCState.IDLE)
        self.assertEqual(machine.rollback_count, 1)


class ReplayTests(unittest.TestCase):
    def test_contiguous_window_validation(self):
        value = _window(0)
        value.timesteps = (0, 2)
        with self.assertRaises(ReplayError):
            value.__post_init__()

    def test_balanced_reservoir_state_and_commit_boundary(self):
        memory = ClusterBalancedReplay(
            4,
            maximum_context_clusters=4,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=2,
            seed=7,
        )
        for index in range(8):
            embedding = (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
            memory.add_committed_window(_window(index, embedding), commit_kind="slow")
        self.assertEqual(len(memory), 4)
        self.assertEqual(len(memory.cluster_ids), 2)
        state = memory.state_dict()
        restored = ClusterBalancedReplay(
            4,
            maximum_context_clusters=4,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=2,
            seed=7,
        )
        restored.load_state_dict(state)
        left = [item.window_id for item in memory.sample_balanced(6)]
        right = [item.window_id for item in restored.sample_balanced(6)]
        self.assertEqual(left, right)
        with self.assertRaises(ReplayError):
            memory.add_committed_window(_window(100), commit_kind="exception")

    def test_grasp_schedule_exact_and_ordered(self):
        counts = GRASPSampler.phase_counts(7)
        self.assertEqual(counts, {"easy": 2, "balanced": 3, "hard": 2})
        self.assertEqual(sum(counts.values()), 7)
        phases = [GRASPSampler.phase_for_step(step, 7) for step in range(7)]
        self.assertEqual(phases, sorted(phases, key=("easy", "balanced", "hard").index))

    def test_grasp_uses_frozen_online_cluster_distance_and_restores_it(self):
        memory = ClusterBalancedReplay(
            8,
            maximum_context_clusters=1,
            new_cluster_similarity_threshold=-1.0,
            minimum_windows_per_cluster=1,
            seed=19,
        )
        embeddings = ((1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        for index, embedding in enumerate(embeddings):
            window = _window(index, embedding)
            window.difficulty_score = 0.0
            memory.add_committed_window(window, commit_kind="slow")

        snapshot = memory.windows()
        self.assertEqual({item.frozen_context_cluster_id for item in snapshot}, {"cluster-00000000"})
        self.assertTrue(all(item.frozen_context_cluster_prototype is not None for item in snapshot))
        distances = {item.window_id: item.frozen_context_cluster_distance for item in snapshot}
        self.assertGreater(distances["window-0"], distances["window-1"])

        sampler = GRASPSampler(seed=23)
        easy = sampler.sample(snapshot, step_index=0, total_steps=10, batch_size=1)
        hard = sampler.sample(snapshot, step_index=7, total_steps=10, batch_size=1)
        self.assertEqual(easy.windows[0].window_id, "window-1")
        self.assertEqual(hard.windows[0].window_id, "window-0")

        state = memory.state_dict()
        restored = ClusterBalancedReplay(
            8,
            maximum_context_clusters=1,
            new_cluster_similarity_threshold=-1.0,
            minimum_windows_per_cluster=1,
            seed=19,
        )
        restored.load_state_dict(state)
        restored_snapshot = restored.windows()
        self.assertEqual(
            [(item.window_id, item.frozen_context_cluster_id) for item in snapshot],
            [(item.window_id, item.frozen_context_cluster_id) for item in restored_snapshot],
        )
        for left, right in zip(snapshot, restored_snapshot):
            self.assertAlmostEqual(
                left.frozen_context_cluster_distance,
                right.frozen_context_cluster_distance,
            )
            self.assertTrue(
                torch.equal(
                    left.frozen_context_cluster_prototype,
                    right.frozen_context_cluster_prototype,
                )
            )

    def test_grasp_score_combines_cluster_distance_and_difficulty_signals(self):
        near = _window(10, (1.0, 0.0))
        near.difficulty_score = 0.75
        near = near.with_frozen_context_cluster("cluster", (1.0, 0.0))
        far = _window(11, (0.0, 1.0))
        far.difficulty_score = 0.0
        far = far.with_frozen_context_cluster("cluster", (1.0, 0.0))
        scores = {item.window_id: item for item in GRASPSampler.score_windows((near, far))}
        self.assertAlmostEqual(scores[near.window_id].context_cluster_distance, 0.0)
        self.assertAlmostEqual(scores[near.window_id].residual_contact_dynamics, 0.75)
        self.assertAlmostEqual(scores[near.window_id].total_score, 0.75)
        self.assertAlmostEqual(scores[far.window_id].context_cluster_distance, 1.0)
        self.assertAlmostEqual(scores[far.window_id].total_score, 1.0)

        signaled = _window(12, (1.0, 0.0))
        signaled.difficulty_score = 0.25
        signaled.contact_available = True
        signaled.contact = torch.tensor([False, True])
        signaled.dynamics_change_available = True
        signaled.dynamics_change = torch.tensor([False, True])
        signaled = signaled.with_frozen_context_cluster("cluster", (1.0, 0.0))
        signaled_score = GRASPSampler.score_windows((signaled,))[0]
        self.assertAlmostEqual(signaled_score.residual_contact_dynamics, 2.25)
        self.assertAlmostEqual(signaled_score.total_score, 2.25)

    def test_grasp_balanced_phase_is_cluster_balanced_and_rng_checkpointable(self):
        memory = ClusterBalancedReplay(
            12,
            maximum_context_clusters=2,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=1,
            seed=29,
        )
        for index in range(3):
            memory.add_committed_window(_window(index, (1.0, 0.0)), commit_kind="slow")
            memory.add_committed_window(_window(index + 10, (0.0, 1.0)), commit_kind="slow")
        snapshot = memory.windows()
        sampler = GRASPSampler(seed=31)
        first = sampler.sample(snapshot, step_index=3, total_steps=10, batch_size=6)
        cluster_ids = [item.frozen_context_cluster_id for item in first.windows]
        self.assertEqual(cluster_ids.count("cluster-00000000"), 3)
        self.assertEqual(cluster_ids.count("cluster-00000001"), 3)
        state = sampler.state_dict()

        restored = GRASPSampler(seed=31)
        restored.load_state_dict(state)
        original_next = sampler.sample(snapshot, step_index=4, total_steps=10, batch_size=6)
        restored_next = restored.sample(snapshot, step_index=4, total_steps=10, batch_size=6)
        self.assertEqual(
            tuple(item.window_id for item in original_next.windows),
            tuple(item.window_id for item in restored_next.windows),
        )
        self.assertEqual(original_next.duplicate_rate, restored_next.duplicate_rate)

    def test_grasp_ties_use_ascending_window_id_in_easy_and_hard_pools(self):
        windows = []
        for index in reversed(range(4)):
            window = _window(index, (1.0, 0.0))
            window.difficulty_score = 1.0
            windows.append(window)
        sampler = GRASPSampler(seed=37)
        easy = sampler.sample(windows, step_index=0, total_steps=10, batch_size=2)
        hard = sampler.sample(windows, step_index=7, total_steps=10, batch_size=2)
        self.assertEqual(tuple(item.window_id for item in easy.windows), ("window-0", "window-1"))
        self.assertEqual(tuple(item.window_id for item in hard.windows), ("window-0", "window-1"))


class RouterTests(unittest.TestCase):
    def test_raw_sufficient_statistics_preserve_heterogeneous_descriptor_magnitude(self):
        router = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=23,
        )
        adapter_id, _ = router.commit_new(
            adapter_state={"version": 1},
            context_descriptors=[torch.tensor([10.0, 0.0])],
            residual_descriptors=[torch.tensor([2.0, 0.0])],
        )
        router.commit_replace(
            adapter_id,
            adapter_state={"version": 2},
            context_descriptors=[torch.tensor([0.0, 1.0])],
            residual_descriptors=[torch.tensor([0.0, 1.0])],
        )
        record = router.get(adapter_id)
        self.assertTrue(
            torch.equal(record.context_descriptor_sum, torch.tensor([10.0, 1.0]))
        )
        self.assertTrue(
            torch.allclose(
                record.context_prototype,
                torch.tensor([10.0, 1.0]) / torch.sqrt(torch.tensor(101.0)),
            )
        )
        self.assertTrue(
            torch.equal(record.residual_descriptor_sum, torch.tensor([2.0, 1.0]))
        )
        self.assertTrue(
            torch.allclose(
                record.residual_prototype,
                torch.tensor([2.0, 1.0]) / torch.sqrt(torch.tensor(5.0)),
            )
        )

        restored = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=23,
        )
        restored.load_state_dict(router.state_dict())
        self.assertTrue(
            bitwise_state_equal(
                restored.get(adapter_id).context_descriptor_sum,
                record.context_descriptor_sum,
            )
        )
        self.assertTrue(
            bitwise_state_equal(
                restored.get(adapter_id).residual_descriptor_sum,
                record.residual_descriptor_sum,
            )
        )

    def test_schema1_prototype_only_checkpoint_has_explicit_compatibility_statistics(self):
        router = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=29,
        )
        adapter_id, _ = router.commit_new(
            adapter_state={"version": 1},
            context_descriptors=[torch.tensor([4.0, 0.0])],
            residual_descriptors=[torch.tensor([0.0, 3.0])],
        )
        legacy = copy.deepcopy(router.state_dict())
        legacy["schema_version"] = 1
        for raw in legacy["records"].values():
            del raw["context_descriptor_sum"]
            del raw["residual_descriptor_sum"]

        restored = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=29,
        )
        restored.load_state_dict(legacy)
        compatible = restored.get(adapter_id)
        self.assertTrue(
            torch.equal(compatible.context_descriptor_sum, torch.tensor([1.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(compatible.residual_descriptor_sum, torch.tensor([0.0, 1.0]))
        )
        restored.commit_replace(
            adapter_id,
            adapter_state={"version": 2},
            context_descriptors=[torch.tensor([0.0, 1.0])],
            residual_descriptors=[torch.tensor([1.0, 0.0])],
        )
        updated = restored.get(adapter_id)
        expected = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        self.assertTrue(torch.allclose(updated.context_prototype, expected))
        self.assertTrue(torch.allclose(updated.residual_prototype, expected))
        migrated = restored.state_dict()
        self.assertEqual(migrated["schema_version"], 2)
        verified = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=29,
        )
        verified.load_state_dict(migrated)
        self.assertTrue(
            bitwise_state_equal(
                verified.get(adapter_id).context_descriptor_sum,
                updated.context_descriptor_sum,
            )
        )

    def test_residual_running_mean_uses_its_own_committed_count(self):
        router = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=19,
        )
        adapter_id, _ = router.commit_new(
            adapter_state={"version": 1},
            context_descriptors=[torch.tensor([1.0, 0.0])],
            residual_descriptors=[torch.tensor([1.0, 0.0])],
        )
        router.commit_replace(
            adapter_id,
            adapter_state={"version": 2},
            context_descriptors=[
                torch.tensor([1.0, 0.0]),
                torch.tensor([1.0, 0.0]),
                torch.tensor([1.0, 0.0]),
            ],
            residual_descriptors=[torch.tensor([0.0, 0.0])],
        )
        record = router.get(adapter_id)
        self.assertEqual(record.descriptor_count, 4)
        self.assertEqual(record.residual_descriptor_count, 2)
        self.assertTrue(record.residual_available)
        self.assertTrue(
            torch.equal(record.residual_prototype, torch.tensor([1.0, 0.0]))
        )

        restored = ExceptionRouter(
            maximum_adapters=1,
            minimum_route_similarity=0.0,
            local_replay_windows=0,
            seed=19,
        )
        restored.load_state_dict(router.state_dict())
        self.assertEqual(
            restored.get(adapter_id).residual_descriptor_count,
            2,
        )

    def test_threshold_fixed_route_replace_and_local_bound(self):
        router = ExceptionRouter(
            maximum_adapters=2,
            minimum_route_similarity=0.8,
            local_replay_windows=2,
            seed=3,
        )
        adapter_id, evicted = router.commit_new(
            adapter_state={"w": torch.tensor([1.0])},
            context_descriptors=[(1.0, 0.0)],
            local_windows=[_window(0), _window(1), _window(2)],
            validation_gain=1.0,
        )
        self.assertIsNone(evicted)
        self.assertEqual(router.get(adapter_id).usage_count, 1)
        self.assertEqual(router.state_dict()["usage_clock"], 1)
        self.assertTrue(router.route((1.0, 0.0)).matched)
        self.assertFalse(router.route((0.0, 1.0)).matched)
        self.assertFalse(router.route((0.0, 0.0)).matched)
        fixed = router.begin_episode("ep", (1.0, 0.0))
        self.assertEqual(router.route_for_episode("ep"), fixed)
        router.end_episode("ep")
        self.assertEqual(router.get(adapter_id).usage_count, 2)
        self.assertEqual(router.state_dict()["usage_clock"], 2)
        before = router.state_dict()
        with self.assertRaises(ExceptionRouterError):
            router.commit_replace(
                adapter_id,
                adapter_state={},
                context_descriptors=[(float("nan"), 0.0)],
            )
        self.assertTrue(bitwise_state_equal(before, router.state_dict()))
        router.commit_replace(
            adapter_id,
            adapter_state={"w": torch.tensor([2.0])},
            context_descriptors=[(1.0, 0.1)],
            local_windows=[_window(3), _window(4)],
            validation_gain=0.5,
        )
        record = router.get(adapter_id)
        self.assertEqual(record.descriptor_count, 2)
        self.assertEqual(record.usage_count, 2)
        self.assertEqual(record.last_used_clock, 2)
        self.assertLessEqual(len(record.local_replay), 2)


class GateTests(unittest.TestCase):
    def _inputs(self, **updates):
        values = dict(
            proposal_type="global_slow",
            before_commit_loss=1.0,
            fast_commit_loss=0.5,
            candidate_commit_loss=0.6,
            historical_replay_exists=False,
            before_anchor_loss=1.0,
            candidate_anchor_loss=1.0,
            plasticity_before_gain=0.1,
            plasticity_candidate_gain=0.09,
            functional_error_by_layer={"l": 0.01},
            drift_before_by_layer={"l": 0.1},
            drift_candidate_by_layer={"l": 0.12},
        )
        values.update(updates)
        return CommitGateInputs(**values)

    def test_cold_start_and_single_invocation(self):
        evaluator = CommitGateEvaluator(GatesConfig(), functional_error_threshold=0.02)
        report = evaluator.evaluate_once(
            episode_id="ep",
            proposal_id="p",
            query_token_id="q",
            inputs=self._inputs(),
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.by_name("history").status, GateStatus.NOT_APPLICABLE)
        with self.assertRaises(CommitGateError):
            evaluator.evaluate_once(
                episode_id="ep",
                proposal_id="p2",
                query_token_id="q2",
                inputs=self._inputs(),
            )

    def test_existing_history_missing_metric_fails(self):
        evaluator = CommitGateEvaluator(GatesConfig(), functional_error_threshold=0.02)
        report = evaluator.evaluate_once(
            episode_id="ep",
            proposal_id="p",
            query_token_id="q",
            inputs=self._inputs(historical_replay_exists=True),
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.by_name("history").status, GateStatus.FAIL)


class _Box:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": copy.deepcopy(self.value)}

    def load_state_dict(self, state):
        self.value = copy.deepcopy(state["value"])


class TransactionCheckpointTests(unittest.TestCase):
    def test_transaction_restores_all_participants_and_rng(self):
        box = _Box({"x": torch.tensor([1.0])})
        generator = torch.Generator().manual_seed(123)
        with StateTransaction({"box": box}, generators={"g": generator}):
            box.value["x"].add_(4)
            random.random()
            torch.rand(1)
            torch.rand(1, generator=generator)
        self.assertTrue(torch.equal(box.value["x"], torch.tensor([1.0])))
        before = box.state_dict()
        with StateTransaction({"box": box}) as transaction:
            box.value["x"].add_(2)
            transaction.commit()
        self.assertFalse(bitwise_state_equal(before, box.state_dict()))

    def test_checkpoint_atomic_latest_and_recovery(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
                retention_versions=4,
            )
            first = manager.save_committed(
                {"tensor": torch.tensor([1.0])},
                commit_id="c1",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            second = manager.save_committed(
                {"tensor": torch.tensor([2.0])},
                commit_id="c2",
                commit_sequence=2,
                config_identity=HASH_A,
            )
            state, reference = manager.load_latest()
            self.assertEqual(reference.commit_id, "c2")
            self.assertTrue(torch.equal(state["tensor"], torch.tensor([2.0])))
            (root / "latest.json").write_text("not-json", encoding="utf-8")
            recovered, recovered_ref = manager.load_latest(recover_if_needed=True)
            self.assertEqual(recovered_ref.commit_id, "c2")
            self.assertTrue(torch.equal(recovered["tensor"], torch.tensor([2.0])))

    def test_checkpoint_rejects_tampered_journal_identity_and_sequence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
                retention_versions=4,
            )
            manager.save_committed(
                {"value": 1},
                commit_id="c1",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            manager.save_committed(
                {"value": 2},
                commit_id="c2",
                commit_sequence=2,
                config_identity=HASH_A,
            )
            journal_path = root / "state" / "journal-c2.json"
            original = json.loads(journal_path.read_text(encoding="utf-8"))

            wrong_sequence = dict(original)
            wrong_sequence["commit_sequence"] = 999
            journal_path.write_text(
                json.dumps(wrong_sequence, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises(CheckpointValidationError):
                manager.load_latest()
            recovered, recovered_reference = manager.recover_latest()
            self.assertEqual(recovered_reference.commit_id, "c1")
            self.assertEqual(recovered["value"], 1)

            wrong_identity = dict(original)
            wrong_identity["commit_id"] = "c1"
            journal_path.write_text(
                json.dumps(wrong_identity, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises(CheckpointValidationError):
                manager.load_latest()
            recovered, recovered_reference = manager.recover_latest()
            self.assertEqual(recovered_reference.commit_id, "c1")
            self.assertEqual(recovered["value"], 1)

    def test_retention_ignores_corrupt_high_sequence_journal(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
                retention_versions=2,
            )
            first = manager.save_committed(
                {"value": 1},
                commit_id="c1",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            second = manager.save_committed(
                {"value": 2},
                commit_id="c2",
                commit_sequence=2,
                config_identity=HASH_A,
            )
            corrupt = dict(manager.read_journal("c1"))
            corrupt["commit_id"] = "evil"
            corrupt["commit_sequence"] = 999
            (root / "state" / "journal-evil.json").write_text(
                json.dumps(corrupt, sort_keys=True),
                encoding="utf-8",
            )
            third = manager.save_committed(
                {"value": 3},
                commit_id="c3",
                commit_sequence=3,
                config_identity=HASH_A,
            )

            self.assertFalse((root / "state" / first.version_file).exists())
            self.assertTrue((root / "state" / second.version_file).is_file())
            self.assertTrue((root / "state" / third.version_file).is_file())
            third_journal_path = root / "state" / "journal-c3.json"
            third_journal = json.loads(
                third_journal_path.read_text(encoding="utf-8")
            )
            third_journal["commit_sequence"] = 1000
            third_journal_path.write_text(
                json.dumps(third_journal, sort_keys=True),
                encoding="utf-8",
            )
            recovered, recovered_reference = manager.load_latest(
                recover_if_needed=True
            )
            self.assertEqual(recovered_reference.commit_id, "c2")
            self.assertEqual(recovered["value"], 2)

    def test_recovery_orders_mixed_commit_and_episode_snapshot_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
                retention_versions=4,
            )
            manager.save_committed(
                {"value": "commit-1"},
                commit_id="commit-00000001",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            manager.save_committed(
                {"value": "episode-2"},
                commit_id="snapshot-episode-00000002",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            (root / "latest.json").write_text("not-json", encoding="utf-8")
            recovered, reference = manager.load_latest(recover_if_needed=True)
            self.assertEqual(reference.commit_id, "snapshot-episode-00000002")
            self.assertEqual(recovered["value"], "episode-2")

            manager.save_committed(
                {"value": "commit-2"},
                commit_id="commit-00000002",
                commit_sequence=2,
                config_identity=HASH_A,
            )
            (root / "latest.json").write_text("not-json", encoding="utf-8")
            recovered, reference = manager.load_latest(recover_if_needed=True)
            self.assertEqual(reference.commit_id, "commit-00000002")
            self.assertEqual(recovered["value"], "commit-2")

    def test_checkpoint_pointer_failure_preserves_previous_latest(self):
        import fd_psc.checkpoint as checkpoint_module

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
            )
            manager.save_committed(
                {"value": 1}, commit_id="c1", commit_sequence=1, config_identity=HASH_A
            )
            real_atomic_json = checkpoint_module._atomic_json

            def fail_latest(path, value):
                if Path(path) == manager.latest_pointer_path:
                    raise OSError("injected pointer failure")
                return real_atomic_json(path, value)

            with mock.patch.object(checkpoint_module, "_atomic_json", side_effect=fail_latest):
                with self.assertRaises(CheckpointError):
                    manager.save_committed(
                        {"value": 2}, commit_id="c2", commit_sequence=2, config_identity=HASH_A
                    )
            state, reference = manager.load_latest()
            self.assertEqual(reference.commit_id, "c1")
            self.assertEqual(state["value"], 1)
            retry_id = manager.next_available_id("c2")
            self.assertEqual(retry_id, "c2-attempt-00000002")
            manager.save_committed(
                {"value": 2},
                commit_id=retry_id,
                commit_sequence=2,
                config_identity=HASH_A,
            )
            retried_state, retried_reference = manager.load_latest()
            self.assertEqual(retried_reference.commit_id, retry_id)
            self.assertEqual(retried_state["value"], 2)

    def test_checkpoint_rollback_journal_invalidates_superseded_versions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manager = SidecarCheckpointManager(
                state_directory=root / "state",
                latest_pointer_path=root / "latest.json",
                base_checkpoint_hash=HASH_A,
                manifest_hash=HASH_B,
                retention_versions=4,
            )
            first = manager.save_committed(
                {"value": 1},
                commit_id="commit-00000001",
                commit_sequence=1,
                config_identity=HASH_A,
            )
            second = manager.save_committed(
                {"value": 2},
                commit_id="commit-00000002",
                commit_sequence=2,
                config_identity=HASH_A,
            )
            rollback = manager.save_committed(
                {"value": 0, "commit_high_water": 3},
                commit_id="rollback-00000003",
                commit_sequence=3,
                config_identity=HASH_A,
                journal_metadata={"event_type": "periodic_canary_rollback"},
            )
            manager.mark_rolled_back(
                (
                    "commit-00000001",
                    "commit-00000002",
                    "commit-00000003-attempt-00000002",
                ),
                rollback_id=rollback.commit_id,
                rollback_sequence=rollback.commit_sequence,
                reason="mock periodic regression",
            )

            state, reference = manager.load_latest()
            self.assertEqual(reference.commit_id, "rollback-00000003")
            self.assertEqual(state["value"], 0)
            self.assertEqual(
                manager.read_journal("commit-00000001")["status"], "rolled_back"
            )
            self.assertEqual(
                manager.read_journal("commit-00000002")["status"], "rolled_back"
            )
            # The failed current candidate did not have a version, but it still
            # receives a terminal journal so its ID cannot be reused.
            attempted = manager.read_journal(
                "commit-00000003-attempt-00000002"
            )
            self.assertEqual(attempted["status"], "rolled_back")
            self.assertEqual(attempted["rollback_id"], rollback.commit_id)
            with self.assertRaises(CheckpointValidationError):
                manager.load_version(manager.state_directory / first.version_file)
            with self.assertRaises(CheckpointValidationError):
                manager.load_version(manager.state_directory / second.version_file)
            recovered, recovered_reference = manager.recover_latest()
            self.assertEqual(recovered_reference.commit_id, rollback.commit_id)
            self.assertEqual(recovered["value"], 0)
            manager.save_committed(
                {"value": 4},
                commit_id="commit-00000004",
                commit_sequence=4,
                config_identity=HASH_A,
            )
            self.assertFalse((manager.state_directory / first.version_file).exists())
            self.assertFalse((manager.state_directory / second.version_file).exists())
            self.assertEqual(
                manager.read_journal("commit-00000001")["status"], "rolled_back"
            )
            with self.assertRaises(CheckpointValidationError):
                manager.load_version(manager.state_directory / first.version_file)


class RepairMetricsDiagnosticsTests(unittest.TestCase):
    def test_repair_is_cumulative_and_screening_stops_first_success(self):
        engine = RepairEngine(
            maximum_steps=4,
            candidate_steps=[2, 4],
            windows_per_batch=1,
            current_weight=1.0,
            replay_weight=1.0,
            proximal_enabled=False,
            proximal_weight=0.0,
            pcgrad_enabled=True,
            seed=0,
        )

        def train_step(state, batch, step, pcgrad):
            state["value"] += 1
            return {"loss": 5 - state["value"]}

        def screen(state, step):
            return ScreeningResult(state["value"] >= 4, {"value": state["value"]}, "checked")

        result = engine.run(
            {"value": 0},
            current_windows=[_window(20)],
            replay_windows=[_window(21)],
            train_step=train_step,
            screen_candidate=screen,
        )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.selected_step, 4)
        self.assertEqual([item.cumulative_step for item in result.checkpoints], [2, 4])

    def test_structured_metrics_and_diagnostics(self):
        metrics = StructuredMetrics()
        metrics.record("commit_query_gate_invocation_count", 1, episode_id="ep")
        metrics.increment("rollback_count")
        self.assertEqual(metrics.counter("rollback_count"), 1)
        diagnostics = Diagnostics()
        diagnostics.fallback("svd_retry", "rank_deficient", logical_layer="l")
        self.assertEqual(len(diagnostics.events()), 1)
        assert_finite_tree({"x": torch.tensor([1.0])})
        with self.assertRaises(Exception):
            assert_finite_tree({"x": torch.tensor([float("nan")])})

    def test_nullable_metrics_preserve_missingness_in_jsonl_and_csv(self):
        metrics = StructuredMetrics()
        available = metrics.record_nullable(
            "forgetting",
            0.25,
            status="available",
            episode_id="ep",
        )
        unavailable = metrics.record_nullable(
            "backward_transfer",
            None,
            status="not_applicable",
            reason="historical_replay_is_empty",
            episode_id="ep",
        )
        self.assertEqual(available.tags["status"], "available")
        self.assertIsNone(unavailable.value)
        self.assertEqual(unavailable.tags["status"], "not_applicable")
        self.assertEqual(
            unavailable.tags["reason"],
            "historical_replay_is_empty",
        )

        with self.assertRaises(MetricsError):
            metrics.record_nullable("forgetting", None, status="available")
        with self.assertRaises(MetricsError):
            metrics.record_nullable("forgetting", 0.0, status="unavailable")
        with self.assertRaises(MetricsError):
            metrics.record_nullable("forgetting", None, status="invented")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "metrics.jsonl"
            csv_path = root / "metrics.csv"
            metrics.write_jsonl(jsonl)
            metrics.write_csv(csv_path)
            rows = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[1]["name"], "backward_transfer")
            self.assertIsNone(rows[1]["value"])
            self.assertEqual(rows[1]["tags"]["status"], "not_applicable")
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("not_applicable", csv_text)
            self.assertNotIn("nan", csv_text.lower())


if __name__ == "__main__":
    unittest.main()
