import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from fd_psc.lora_layers import DualLoRAConv2d, DualLoRALinear
from fd_psc.replay_memory import (
    ClusterBalancedReplay,
    RawReplayMemory,
    RawReplayWindow,
    ReplayError,
    deep_cpu_clone,
)
from fd_psc.v2.geometry import FreshReplayGeometryBuilder


def _raw_window(
    index: int,
    *,
    embedding=(1.0, 0.0),
    visual=None,
    optional_cache=None,
    optional_cache_version=-1,
) -> RawReplayWindow:
    if visual is None:
        visual = torch.arange(12, dtype=torch.uint8).reshape(1, 3, 2, 2) + index
    obs = {
        "visual": visual,
        "proprio": torch.tensor(
            [[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]],
            dtype=torch.float32,
        )
        + float(index),
    }
    actions = torch.tensor([[[0.25], [0.5]]], dtype=torch.float32) + float(index)
    trajectory_id = f"trajectory-{index}"
    return RawReplayWindow(
        window_id=f"raw-window-{index}",
        trajectory_id=trajectory_id,
        transition_ids=tuple(
            f"{trajectory_id}:transition={step}" for step in range(2)
        ),
        frame_ids=tuple(f"{trajectory_id}:frame={step}" for step in range(3)),
        timesteps=(0, 1, 2),
        content_hash=RawReplayWindow.compute_content_hash(obs, actions),
        context_identifier=f"context-{index % 2}",
        obs=obs,
        actions=actions,
        context_embedding=embedding,
        source_episode=f"episode-{index}",
        preprocess_hash="a" * 64,
        base_checkpoint_hash="b" * 64,
        optional_latent_cache=optional_cache,
        optional_latent_cache_model_version=optional_cache_version,
        committed=True,
    )


def _feature_window(
    index: int,
    features: torch.Tensor,
    *,
    optional_cache=None,
    optional_cache_version=-1,
) -> RawReplayWindow:
    if features.ndim != 3 or features.shape[:2] != (1, 3):
        raise ValueError("feature fixture must be [1,3,d]")
    obs = {"features": features}
    actions = torch.zeros(1, 2, 1)
    trajectory_id = f"feature-trajectory-{index}"
    return RawReplayWindow(
        window_id=f"feature-window-{index}",
        trajectory_id=trajectory_id,
        transition_ids=(f"{trajectory_id}:t0", f"{trajectory_id}:t1"),
        frame_ids=(f"{trajectory_id}:f0", f"{trajectory_id}:f1", f"{trajectory_id}:f2"),
        timesteps=(0, 1, 2),
        content_hash=RawReplayWindow.compute_content_hash(obs, actions),
        context_identifier="geometry-context",
        obs=obs,
        actions=actions,
        context_embedding=(1.0, 0.0),
        optional_latent_cache=optional_cache,
        optional_latent_cache_model_version=optional_cache_version,
        committed=True,
    )


class _TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        first = nn.Linear(3, 3, bias=False)
        second = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            first.weight.copy_(torch.eye(3))
            second.weight.copy_(
                torch.tensor([[1.0, 0.5, 0.0], [0.0, -0.5, 1.0]])
            )
        self.first = DualLoRALinear(first, rank=1, alpha=1.0, logical_id="first")
        self.second = DualLoRALinear(second, rank=1, alpha=1.0, logical_id="second")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


class RawReplaySchemaTests(unittest.TestCase):
    def test_deep_cpu_clone_preserves_dtype_bits_and_ownership(self):
        source = {
            "uint8": torch.tensor([[0, 255]], dtype=torch.uint8),
            "float": [torch.tensor([1.25], dtype=torch.float64)],
        }
        cloned = deep_cpu_clone(source)
        self.assertEqual(cloned["uint8"].device.type, "cpu")
        self.assertEqual(cloned["uint8"].dtype, torch.uint8)
        self.assertEqual(cloned["float"][0].dtype, torch.float64)
        self.assertTrue(torch.equal(cloned["uint8"], source["uint8"]))
        source["uint8"].zero_()
        self.assertEqual(int(cloned["uint8"][0, 1]), 255)

    def test_raw_obs_actions_roundtrip_rng_and_checkpoint_are_bitwise(self):
        self.assertEqual(ClusterBalancedReplay.SCHEMA_VERSION, 1)
        self.assertEqual(RawReplayMemory.SCHEMA_VERSION, 2)
        memory = RawReplayMemory(
            4,
            maximum_context_clusters=2,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=1,
            seed=17,
        )
        original = _raw_window(0)
        expected_visual = original.obs["visual"].clone()
        expected_actions = original.actions.clone()
        memory.add_committed_window(original, commit_kind="slow")
        original.obs["visual"].zero_()
        original.actions.zero_()
        for index in range(1, 4):
            embedding = (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
            memory.add_committed_window(
                _raw_window(index, embedding=embedding),
                commit_kind="slow",
            )

        stored = {window.window_id: window for window in memory.windows()}[
            "raw-window-0"
        ]
        self.assertTrue(torch.equal(stored.obs["visual"], expected_visual))
        self.assertTrue(torch.equal(stored.actions, expected_actions))
        self.assertEqual(stored.obs["visual"].dtype, torch.uint8)
        self.assertFalse(hasattr(stored, "visual_latent"))

        state = memory.state_dict()
        restored = RawReplayMemory(
            4,
            maximum_context_clusters=2,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=1,
            seed=17,
        )
        restored.load_state_dict(copy.deepcopy(state))
        expected_sequence = [
            window.window_id for window in memory.sample_balanced(8)
        ]
        restored_sequence = [
            window.window_id for window in restored.sample_balanced(8)
        ]
        self.assertEqual(restored_sequence, expected_sequence)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "raw-replay.pt"
            torch.save(state, checkpoint)
            loaded_state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            resumed = RawReplayMemory(
                4,
                maximum_context_clusters=2,
                new_cluster_similarity_threshold=0.8,
                minimum_windows_per_cluster=1,
                seed=17,
            )
            resumed.load_state_dict(loaded_state)
            before = {window.window_id: window for window in memory.windows()}
            after = {window.window_id: window for window in resumed.windows()}
            self.assertEqual(set(after), set(before))
            for window_id in before:
                self.assertTrue(
                    torch.equal(after[window_id].obs["visual"], before[window_id].obs["visual"])
                )
                self.assertTrue(
                    torch.equal(after[window_id].actions, before[window_id].actions)
                )

    def test_load_reaudits_raw_truth_and_rejects_tampering_or_v1(self):
        memory = RawReplayMemory(
            2,
            maximum_context_clusters=1,
            new_cluster_similarity_threshold=0.0,
            minimum_windows_per_cluster=1,
            seed=23,
        )
        memory.add_committed_window(_raw_window(0), commit_kind="slow")
        state = memory.state_dict()
        cluster = next(iter(state["clusters"].values()))
        cluster["windows"][0].obs["visual"][0, 0, 0, 0] += 1
        restored = RawReplayMemory(
            2,
            maximum_context_clusters=1,
            new_cluster_similarity_threshold=0.0,
            minimum_windows_per_cluster=1,
            seed=23,
        )
        with self.assertRaisesRegex(ReplayError, "content_hash"):
            restored.load_state_dict(state)

        dtype_tamper = copy.deepcopy(memory.state_dict())
        dtype_cluster = next(iter(dtype_tamper["clusters"].values()))
        dtype_cluster["windows"][0].actions = dtype_cluster["windows"][
            0
        ].actions.to(dtype=torch.float64)
        with self.assertRaisesRegex(ReplayError, "content_hash"):
            restored.load_state_dict(dtype_tamper)

        legacy = copy.deepcopy(memory.state_dict())
        legacy["schema_version"] = 1
        with self.assertRaisesRegex(ReplayError, "raw replay schema"):
            restored.load_state_dict(legacy)

    def test_optional_cache_requires_exact_model_version(self):
        cache = torch.tensor([123.0])
        window = _raw_window(
            0,
            optional_cache=cache,
            optional_cache_version=7,
        )
        self.assertIsNone(window.latent_cache_for_model_version(6))
        self.assertIsNone(window.latent_cache_for_model_version(8))
        self.assertTrue(
            torch.equal(window.latent_cache_for_model_version(7), cache)
        )


class FreshReplayGeometryTests(unittest.TestCase):
    def _builder(self, model: _TwoLayerModel, *, maximum_rank=3):
        entries = (
            SimpleNamespace(
                module_path="first",
                logical_layer_id="first",
                logical_group=None,
                injected=True,
            ),
            SimpleNamespace(
                module_path="second",
                logical_layer_id="second",
                logical_group=None,
                injected=True,
            ),
        )
        return FreshReplayGeometryBuilder(
            {"first": model.first, "second": model.second},
            entries,
            maximum_rank,
            1.0,
            1.0e-10,
            1.0e-8,
        )

    @staticmethod
    def _forward(model: _TwoLayerModel, seen_caches):
        def forward_window(window: RawReplayWindow):
            seen_caches.append(window.optional_latent_cache)
            value = window.obs["features"].reshape(-1, 3).to(dtype=torch.float32)
            return model(value)

        return forward_window

    def test_geometry_ignores_cache_and_reembeds_after_upstream_slow_change(self):
        model = _TwoLayerModel()
        features = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]]
        )
        window = _feature_window(
            0,
            features,
            optional_cache=torch.full((5,), 999.0),
            optional_cache_version=3,
        )
        builder = self._builder(model)
        seen_caches = []
        before = builder.build(
            (window,),
            self._forward(model, seen_caches),
            persistent_model_version=4,
        )
        self.assertEqual(seen_caches, [None])
        self.assertIsNone(window.latent_cache_for_model_version(4))
        self.assertEqual(before.window_count, 1)
        self.assertEqual(before.persistent_model_version, 4)
        self.assertGreater(before.by_layer["first"].output_energy, 0.0)
        self.assertGreater(before.by_layer["second"].output_energy, 0.0)

        model.first.replace_slow_adapter(
            torch.tensor([[0.0], [2.0], [0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        after = builder.build(
            (window,),
            self._forward(model, []),
            persistent_model_version=5,
        )
        torch.testing.assert_close(
            before.by_layer["first"].eigenvalues,
            after.by_layer["first"].eigenvalues,
        )
        self.assertFalse(
            torch.allclose(
                before.by_layer["second"].eigenvalues,
                after.by_layer["second"].eigenvalues,
            )
        )

    def test_empty_replay_returns_typed_cold_start_without_forward(self):
        model = _TwoLayerModel()
        callback_count = 0

        def forward_window(_window):
            nonlocal callback_count
            callback_count += 1
            raise AssertionError("empty replay must not execute a model forward")

        result = self._builder(model).build(
            (),
            forward_window,
            persistent_model_version=0,
        )
        self.assertEqual(callback_count, 0)
        self.assertEqual(result.window_count, 0)
        self.assertEqual(set(result.by_layer), {"first", "second"})
        for geometry in result.by_layer.values():
            self.assertEqual(geometry.rank, 0)
            self.assertEqual(geometry.sample_count, 0)
            self.assertEqual(geometry.tail_upper_bound, 0.0)
            self.assertEqual(geometry.q.dtype, torch.float32)

    def test_nonempty_rank_zero_keeps_first_discarded_eigenvalue(self):
        model = _TwoLayerModel()
        window = _feature_window(
            0,
            torch.tensor(
                [[[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.0, 1.0]]]
            ),
        )
        result = self._builder(model, maximum_rank=0).build(
            (window,),
            self._forward(model, []),
        )
        for geometry in result.by_layer.values():
            self.assertEqual(geometry.rank, 0)
            self.assertGreater(geometry.sample_count, 0)
            self.assertGreater(geometry.tail_upper_bound, 0.0)

    def test_grouped_conv_geometry_is_separate_per_group(self):
        base = nn.Conv2d(4, 4, kernel_size=1, groups=2, bias=False)
        with torch.no_grad():
            base.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]]],
                        [[[0.0]], [[1.0]]],
                        [[[1.0]], [[0.0]]],
                        [[[0.0]], [[1.0]]],
                    ]
                )
            )
        conv = DualLoRAConv2d(base, rank=1, alpha=1.0, logical_id="conv")
        builder = FreshReplayGeometryBuilder(
            {"conv": conv},
            maximum_rank=2,
            spectral_energy_threshold=1.0,
            minimum_energy=1.0e-10,
        )
        visual = torch.tensor(
            [
                [
                    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[2.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
                    [[[0.5, 0.0], [0.0, 0.5]], [[0.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
                    [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [0.0, 0.0]]],
                ]
            ]
        )
        window = _raw_window(0, visual=visual)

        def forward_window(raw: RawReplayWindow):
            value = raw.obs["visual"].reshape(-1, 4, 2, 2).float()
            return conv(value)

        result = builder.build((window,), forward_window)
        self.assertEqual(set(result.by_layer), {"conv::group=0", "conv::group=1"})
        for geometry in result.by_layer.values():
            self.assertEqual(geometry.input_dim, 2)
            self.assertEqual(geometry.sample_count, 12)
            self.assertGreater(geometry.output_energy, 0.0)

        changed_visual = visual.clone()
        changed_visual[:, :, 2:] *= 4.0
        changed = builder.build(
            (_raw_window(1, visual=changed_visual),),
            forward_window,
        )
        torch.testing.assert_close(
            result.by_layer["conv::group=0"].eigenvalues,
            changed.by_layer["conv::group=0"].eigenvalues,
        )
        self.assertFalse(
            torch.allclose(
                result.by_layer["conv::group=1"].eigenvalues,
                changed.by_layer["conv::group=1"].eigenvalues,
            )
        )

    def test_model_state_context_restores_fast_state_and_hooks_on_failure(self):
        model = _TwoLayerModel()
        window = _feature_window(0, torch.ones(1, 3, 3))
        builder = self._builder(model)
        fast_before = copy.deepcopy(model.first.adapter_state_dict())
        hook_counts = (
            len(model.first._forward_hooks),
            len(model.second._forward_hooks),
        )

        @contextlib.contextmanager
        def persistent_state():
            state = copy.deepcopy(model.first.adapter_state_dict())
            try:
                with torch.no_grad():
                    model.first.pilot_B.zero_()
                yield
            finally:
                model.first.load_adapter_state_dict(state)

        def fail(_window):
            model(_window.obs["features"].reshape(-1, 3))
            raise RuntimeError("synthetic raw forward failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            builder.build(
                (window,),
                fail,
                model_state_context=persistent_state,
            )
        self.assertEqual(
            (len(model.first._forward_hooks), len(model.second._forward_hooks)),
            hook_counts,
        )
        restored = model.first.adapter_state_dict()
        for key in ("slow_A", "slow_B", "pilot_A", "pilot_B"):
            self.assertTrue(torch.equal(restored[key], fast_before[key]), key)


if __name__ == "__main__":
    unittest.main()
