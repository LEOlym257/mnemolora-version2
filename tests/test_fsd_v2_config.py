import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from fd_psc.config import (
    AdaptiveBudgetConfig,
    DeepSleepConfig,
    FDPSCConfig,
    FDPSCConfigError,
    RTRCConfig,
    RawReplayConfig,
    minimal_fsd_v2_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FSDV2ConfigTests(unittest.TestCase):
    def _shipped_config(self) -> tuple[dict, FDPSCConfig]:
        raw = yaml.safe_load(
            (ROOT / "conf" / "fd_psc" / "fsd_v2.yaml").read_text("utf-8")
        )
        raw["seed"] = 7
        return raw, FDPSCConfig.from_mapping(raw)

    def test_shipped_yaml_is_typed_and_has_no_external_algorithm_data(self):
        raw, cfg = self._shipped_config()

        self.assertNotIn("external_eval_data", raw)
        self.assertNotIn("anchor_data", raw)
        self.assertEqual(cfg.run_mode, "fsd_v2")
        self.assertIsInstance(cfg.rtrc, RTRCConfig)
        self.assertIsInstance(cfg.raw_replay, RawReplayConfig)
        self.assertIsInstance(cfg.deep_sleep, DeepSleepConfig)
        self.assertIsInstance(cfg.adaptive_budget, AdaptiveBudgetConfig)
        self.assertTrue(cfg.deep_sleep.enabled)
        self.assertEqual(cfg.deep_sleep.core_storage.mode, "dense_delta")
        self.assertEqual(cfg.slow_lora.persistent_rank, 32)
        self.assertEqual(cfg.rtrc.geometry_windows, 128)
        cfg.validate(ROOT, require_files=True)

    def test_shipped_yaml_explicitly_disables_every_legacy_main_path(self):
        raw, _cfg = self._shipped_config()

        for section in (
            "gradient_geometry",
            "slice",
            "sdc",
            "spectral_surgery",
            "activation_subspace",
            "repair",
            "exception",
            "canary",
        ):
            with self.subTest(section=section):
                self.assertIs(raw[section]["enabled"], False)
        self.assertIs(raw["merge"]["soft_ness_enabled"], False)
        self.assertIs(raw["gates"]["allow_unsafe_ablation"], False)
        for name in (
            "current_gain_enabled",
            "history_enabled",
            "anchor_enabled",
            "plasticity_enabled",
            "functional_error_enabled",
            "spectral_drift_enabled",
        ):
            self.assertIs(raw["gates"][name], False)

    def test_v2_validation_ignores_legacy_external_data_fields(self):
        cfg = minimal_fsd_v2_config()
        missing = str(ROOT / "definitely-missing-fsd-v2-external-data.json")
        cfg.external_eval_data.manifest_path = missing
        cfg.external_eval_data.calibration_path = missing
        cfg.external_eval_data.commit_query_path = missing
        cfg.external_eval_data.plasticity_support_path = missing
        cfg.external_eval_data.plasticity_query_path = missing
        cfg.external_eval_data.report_test_path = missing
        cfg.external_eval_data.representation = "deliberately-not-frozen-latent"
        cfg.anchor_data.manifest_path = missing
        cfg.anchor_data.data_path = missing
        cfg.anchor_data.windows = -100
        cfg.gradient_geometry.anchor_batches = -100
        cfg.spectral_surgery.anchor_weight = float("nan")

        cfg.validate(ROOT, require_files=True)

    def test_v2_rejects_enabled_legacy_algorithm_controls(self):
        mutations = (
            lambda cfg: setattr(cfg.gradient_geometry, "enabled", True),
            lambda cfg: setattr(cfg.slice, "enabled", True),
            lambda cfg: setattr(cfg.sdc, "enabled", True),
            lambda cfg: setattr(cfg.spectral_surgery, "enabled", True),
            lambda cfg: setattr(cfg.activation_subspace, "enabled", True),
            lambda cfg: setattr(cfg.merge, "soft_ness_enabled", True),
            lambda cfg: setattr(cfg.repair, "enabled", True),
            lambda cfg: setattr(cfg.exception, "enabled", True),
            lambda cfg: setattr(cfg.gates, "current_gain_enabled", True),
            lambda cfg: setattr(cfg.gates, "history_enabled", True),
            lambda cfg: setattr(cfg.gates, "anchor_enabled", True),
            lambda cfg: setattr(cfg.gates, "plasticity_enabled", True),
            lambda cfg: setattr(cfg.gates, "functional_error_enabled", True),
            lambda cfg: setattr(cfg.gates, "spectral_drift_enabled", True),
            lambda cfg: setattr(cfg.canary, "enabled", True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(control=index):
                cfg = minimal_fsd_v2_config()
                mutate(cfg)
                with self.assertRaisesRegex(
                    FDPSCConfigError,
                    "first-round runtime requires",
                ):
                    cfg.validate(ROOT, require_files=True)

    def test_v1_file_requirements_remain_strict(self):
        with self.assertRaisesRegex(FDPSCConfigError, "required FD-PSC path"):
            FDPSCConfig(enabled=True).validate(ROOT, require_files=True)

    def test_minimal_helper_is_valid_without_external_files(self):
        cfg = minimal_fsd_v2_config(
            slow_lora={"persistent_rank": 4},
            rtrc={"geometry_windows": 3, "geometry_maximum_rank": 2},
            raw_replay={"historical_windows": 8, "minimum_windows_per_cluster": 2},
        )

        self.assertEqual(cfg.run_mode, "fsd_v2")
        self.assertEqual(cfg.slow_lora.persistent_rank, 4)
        self.assertEqual(cfg.rtrc.geometry_windows, 3)
        self.assertFalse(cfg.checkpoint.enabled)
        cfg.validate(ROOT, require_files=True)

    def test_v2_persistence_identity_excludes_reporting_paths_and_legacy_controls(self):
        cfg = minimal_fsd_v2_config()
        expected = cfg.persistence_identity_hash()
        self.assertEqual(expected, cfg.v2_persistence_identity_hash())

        changed = copy.deepcopy(cfg)
        changed.external_eval_data.manifest_path = "algorithm-external.json"
        changed.external_eval_data.calibration_path = "calibration.json"
        changed.external_eval_data.commit_query_path = "query.json"
        changed.external_eval_data.plasticity_support_path = "support.json"
        changed.external_eval_data.plasticity_query_path = "plasticity.json"
        changed.external_eval_data.report_test_path = "report-only.json"
        changed.anchor_data.manifest_path = "anchor-manifest.json"
        changed.anchor_data.data_path = "anchor.json"
        changed.gradient_geometry.enabled = True
        changed.gates.anchor_enabled = True
        changed.repair.enabled = True
        changed.checkpoint.state_directory = "another-state-dir"
        changed.checkpoint.latest_pointer_path = "another-latest.json"
        changed.checkpoint.retention_versions = 99
        changed.logging.save_candidate_reports = True

        self.assertEqual(expected, changed.persistence_identity_hash())
        identity = changed.v2_persistence_identity()
        self.assertNotIn("external_eval_data", identity)
        self.assertNotIn("anchor_data", identity)
        self.assertNotIn("checkpoint", identity)
        self.assertNotIn("gates", identity)

    def test_v2_persistence_identity_changes_with_each_v2_algorithm_family(self):
        baseline = minimal_fsd_v2_config()
        expected = baseline.persistence_identity_hash()
        changes = (
            ("seed", lambda cfg: setattr(cfg, "seed", cfg.seed + 1)),
            (
                "target",
                lambda cfg: setattr(cfg.target_modules, "action_encoder_linear", True),
            ),
            (
                "episodic",
                lambda cfg: setattr(cfg.episodic_lora, "rank", cfg.episodic_lora.rank + 1),
            ),
            (
                "slow",
                lambda cfg: setattr(cfg.slow_lora, "persistent_rank", 7),
            ),
            (
                "rtrc",
                lambda cfg: setattr(cfg.rtrc, "geometry_windows", cfg.rtrc.geometry_windows + 1),
            ),
            (
                "raw_replay",
                lambda cfg: setattr(
                    cfg.raw_replay,
                    "historical_windows",
                    cfg.raw_replay.historical_windows + 1,
                ),
            ),
            (
                "deep_sleep",
                lambda cfg: setattr(
                    cfg.deep_sleep,
                    "trigger_consecutive_commits",
                    cfg.deep_sleep.trigger_consecutive_commits + 1,
                ),
            ),
            (
                "adaptive_budget",
                lambda cfg: setattr(
                    cfg.adaptive_budget,
                    "controller_learning_rate",
                    cfg.adaptive_budget.controller_learning_rate + 0.1,
                ),
            ),
        )
        for name, mutate in changes:
            with self.subTest(name=name):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.assertNotEqual(expected, changed.persistence_identity_hash())

    def test_v1_persistence_identity_ignores_new_v2_only_defaults(self):
        cfg = FDPSCConfig(enabled=True)
        legacy = cfg.to_dict()
        legacy.pop("rtrc")
        legacy.pop("raw_replay")
        legacy.pop("deep_sleep")
        legacy.pop("adaptive_budget")
        legacy["slow_lora"].pop("persistent_rank")
        legacy["checkpoint"]["resume_path"] = None
        payload = json.dumps(
            legacy,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        self.assertEqual(cfg.persistence_identity_hash(), expected)

    def test_resolve_v2_paths_never_returns_external_paths(self):
        cfg = minimal_fsd_v2_config(
            checkpoint={
                "enabled": True,
                "state_directory": "memory",
                "latest_pointer_path": "memory/latest.json",
            }
        )
        cfg.external_eval_data.manifest_path = "must-not-be-resolved.json"
        cfg.anchor_data.data_path = "must-not-be-resolved-anchor.json"

        paths = cfg.resolve_v2_paths(ROOT)

        self.assertEqual(set(paths), {"state_directory", "latest_pointer", "resume"})
        self.assertEqual(paths["state_directory"], (ROOT / "memory").resolve())
        self.assertEqual(paths["latest_pointer"], (ROOT / "memory/latest.json").resolve())
        self.assertIsNone(paths["resume"])

    def test_resume_path_is_structurally_and_physically_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            memory.mkdir()
            latest = memory / "latest.json"
            latest.write_text("{}", encoding="utf-8")
            cfg = minimal_fsd_v2_config(
                checkpoint={
                    "enabled": True,
                    "state_directory": "memory",
                    "latest_pointer_path": "memory/latest.json",
                    "resume_path": "memory/latest.json",
                }
            )
            cfg.validate(root, require_files=True)

            cfg.checkpoint.resume_path = "memory/missing.pt"
            with self.assertRaisesRegex(FDPSCConfigError, "not readable"):
                cfg.validate(root, require_files=True)

            cfg.checkpoint.resume_path = "outside.pt"
            with self.assertRaisesRegex(FDPSCConfigError, "directly inside"):
                cfg.validate(root, require_files=False)

            cfg.checkpoint.resume_path = ""
            with self.assertRaisesRegex(FDPSCConfigError, "null or a non-empty"):
                cfg.validate(root, require_files=False)

    def test_v2_numeric_controls_fail_fast(self):
        cases = (
            ("persistent_rank", lambda cfg: setattr(cfg.slow_lora, "persistent_rank", 0)),
            (
                "budget_order",
                lambda cfg: setattr(cfg.rtrc, "budget_fraction_minimum", 0.5),
            ),
            ("geometry_windows", lambda cfg: setattr(cfg.rtrc, "geometry_windows", 0)),
            (
                "geometry_rank",
                lambda cfg: setattr(cfg.rtrc, "geometry_maximum_rank", 0),
            ),
            (
                "geometry_threshold",
                lambda cfg: setattr(cfg.rtrc, "geometry_energy_threshold", math.nan),
            ),
            ("shared_dual", lambda cfg: setattr(cfg.rtrc, "use_shared_dual", False)),
            ("tail", lambda cfg: setattr(cfg.rtrc, "tail_mode", "discard")),
            (
                "bisection",
                lambda cfg: setattr(cfg.rtrc, "bisection_iterations", 0),
            ),
            ("epsilon", lambda cfg: setattr(cfg.rtrc, "epsilon", 0.0)),
            (
                "raw_capacity",
                lambda cfg: setattr(cfg.raw_replay, "historical_windows", 0),
            ),
            (
                "raw_truth",
                lambda cfg: setattr(cfg.raw_replay, "store_model_input_obs", False),
            ),
            (
                "latent_cache",
                lambda cfg: setattr(
                    cfg.raw_replay,
                    "store_frozen_visual_latent_cache",
                    True,
                ),
            ),
            (
                "raw_dtype",
                lambda cfg: setattr(cfg.raw_replay, "image_storage_dtype", "float16"),
            ),
            (
                "checkpoint_journal",
                lambda cfg: setattr(cfg.checkpoint, "keep_commit_journal", False),
            ),
            (
                "checkpoint_atomic",
                lambda cfg: setattr(cfg.checkpoint, "atomic_write", False),
            ),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                cfg = minimal_fsd_v2_config()
                mutate(cfg)
                with self.assertRaises(FDPSCConfigError):
                    cfg.validate(require_files=False)


if __name__ == "__main__":
    unittest.main()
