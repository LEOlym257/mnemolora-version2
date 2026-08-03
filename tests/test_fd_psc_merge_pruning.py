import unittest
from types import SimpleNamespace

import torch

from fd_psc.config import FDPSCConfig, FDPSCConfigError
from fd_psc.trainer import FDPSCSystem, _MergeSimilaritySignals


class MergeCoefficientPruningTest(unittest.TestCase):
    def _system(self) -> FDPSCSystem:
        system = object.__new__(FDPSCSystem)
        system.config = FDPSCConfig(enabled=True)
        return system

    def test_conflict_restricts_shared_grid(self):
        system = self._system()
        result = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=-0.5)
        )
        self.assertEqual(result.shared_coefficients, (0.0, 0.1, 0.25))
        self.assertEqual(result.reason, "conflict_pruned")
        self.assertEqual(result.signal_decisions, {"gradient": "conflict"})

    def test_match_restricts_shared_grid(self):
        system = self._system()
        result = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=0.5)
        )
        self.assertEqual(result.shared_coefficients, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(result.reason, "match_pruned")

    def test_missing_configured_signal_preserves_full_grid(self):
        system = self._system()
        system.config.merge.context_conflict_threshold = -0.2
        system.config.merge.context_match_threshold = 0.2
        result = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=0.8, context=None)
        )
        self.assertEqual(
            result.shared_coefficients,
            tuple(system.config.merge.shared_coefficients),
        )
        self.assertEqual(result.reason, "missing_signal:context")

    def test_contradictory_signals_preserve_full_grid(self):
        system = self._system()
        system.config.merge.context_conflict_threshold = -0.2
        system.config.merge.context_match_threshold = 0.2
        result = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=0.8, context=-0.8)
        )
        self.assertEqual(
            result.shared_coefficients,
            tuple(system.config.merge.shared_coefficients),
        )
        self.assertEqual(result.reason, "contradictory_signals")

    def test_default_missing_history_preserves_full_grid(self):
        system = self._system()
        # The shipped defaults configure only gradient thresholds.  A first
        # episode has no history gradient, so it must not be pruned.
        result = system._pruned_coefficient_grid(_MergeSimilaritySignals())
        self.assertEqual(
            result.shared_coefficients,
            tuple(system.config.merge.shared_coefficients),
        )
        self.assertEqual(result.reason, "missing_signal:gradient")

    def test_null_thresholds_and_disabled_signals_do_not_prune(self):
        system = self._system()
        system.config.merge.gradient_conflict_threshold = None
        system.config.merge.gradient_match_threshold = None
        system.config.merge.context_conflict_threshold = None
        system.config.merge.context_match_threshold = None
        system.config.merge.residual_match_threshold = None
        result = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=-1.0, context=1.0, residual=1.0)
        )
        self.assertEqual(
            result.shared_coefficients,
            tuple(system.config.merge.shared_coefficients),
        )
        self.assertEqual(result.reason, "no_decisive_signal")

        system.config.merge.gradient_conflict_threshold = -0.1
        system.config.merge.gradient_match_threshold = 0.1
        system.config.merge.use_gradient_similarity = False
        disabled = system._pruned_coefficient_grid(
            _MergeSimilaritySignals(gradient=-1.0)
        )
        self.assertEqual(disabled.shared_coefficients, result.shared_coefficients)

    def test_residual_match_uses_only_audited_theta0_pattern_descriptors(self):
        def window(window_id, residual, *, audited=True):
            return SimpleNamespace(
                window_id=window_id,
                residual=torch.tensor(residual, dtype=torch.float32),
                metadata=(
                    {
                        "theta0_residual": True,
                        "residual_descriptor_schema": "theta0_jepa_pattern_v1",
                    }
                    if audited
                    else {}
                ),
            )

        current = (window("current-1", [2.0, 0.0]), window("current-2", [1.0, 0.0]))
        history = (window("history", [4.0, 0.0]),)
        similarity = FDPSCSystem._nearest_residual_similarity(current, history)
        self.assertAlmostEqual(similarity, 1.0, places=6)
        self.assertIsNone(
            FDPSCSystem._nearest_residual_similarity(
                current,
                (window("legacy", [4.0, 0.0], audited=False),),
            )
        )

    def test_nearest_frozen_context_similarity_is_deterministic(self):
        history = (
            SimpleNamespace(window_id="b", context_embedding=torch.tensor([0.0, 1.0])),
            SimpleNamespace(window_id="a", context_embedding=torch.tensor([0.8, 0.6])),
        )
        similarity = FDPSCSystem._nearest_context_similarity(
            torch.tensor([1.0, 0.0]),
            history,
        )
        self.assertAlmostEqual(similarity, 0.8, places=6)

    def test_similarity_thresholds_are_bounded_and_ordered(self):
        for field in (
            "context_conflict_threshold",
            "context_match_threshold",
            "gradient_conflict_threshold",
            "gradient_match_threshold",
            "residual_match_threshold",
        ):
            with self.subTest(field=field):
                cfg = FDPSCConfig(enabled=True)
                setattr(cfg.merge, field, 1.01)
                with self.assertRaisesRegex(FDPSCConfigError, f"merge\\.{field}"):
                    cfg.validate(require_files=False)

        cfg = FDPSCConfig(enabled=True)
        cfg.merge.gradient_conflict_threshold = 0.2
        cfg.merge.gradient_match_threshold = 0.1
        with self.assertRaisesRegex(FDPSCConfigError, "gradient conflict threshold"):
            cfg.validate(require_files=False)


if __name__ == "__main__":
    unittest.main()
