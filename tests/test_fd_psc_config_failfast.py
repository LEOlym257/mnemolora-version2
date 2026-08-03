import copy
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from fd_psc.config import FDPSCConfig, FDPSCConfigError


class DeclaredControlFailFastTest(unittest.TestCase):
    def test_shipped_default_values_remain_valid(self):
        FDPSCConfig(enabled=True).validate(require_files=False)

    def test_unimplemented_nondefault_controls_are_rejected(self):
        cases = (
            ("target_modules.exclude_frozen_backbone", False),
            ("episodic_lora.pilot_enabled", False),
            ("episodic_lora.a_initialization", "zeros"),
            ("episodic_lora.b_initialization", "normal"),
            ("gradient_geometry.projection_scope", "global"),
            ("gradient_geometry.hook_normalization", "batch_mean"),
            ("slice.trigger_only", False),
            ("merge.selection_policy", "first_feasible"),
            ("replay.compression", "zstd"),
            ("replay.sampling", "uniform"),
            ("external_eval_data.source", "online"),
            ("external_eval_data.context_source", "payload"),
            ("external_eval_data.split_unit", "window"),
            ("external_eval_data.require_context_match", False),
            ("external_eval_data.missing_context_policy", "fallback"),
            ("anchor_data.source", "inline"),
            ("anchor_data.verify_checksums", False),
            ("anchor_data.missing_policy", "skip"),
            ("exception.routing", "learned"),
            ("exception.routed_episode_update", "merge_exception"),
            ("exception.eviction_policy", "lowest_gain"),
            ("exception.merge_similar_adapters", True),
            ("checkpoint.save_every_episodes", 2),
            ("checkpoint.keep_commit_journal", False),
            ("checkpoint.atomic_write", False),
            ("logging.per_layer_metrics", False),
            ("logging.save_candidate_reports", False),
            ("logging.save_gradient_statistics", False),
        )
        for dotted_name, unsupported in cases:
            with self.subTest(field=dotted_name, value=unsupported):
                cfg = FDPSCConfig(enabled=True)
                owner = cfg
                parts = dotted_name.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                setattr(owner, parts[-1], unsupported)
                with self.assertRaisesRegex(
                    FDPSCConfigError,
                    rf"^{re.escape(dotted_name)}=.* is not implemented;",
                ):
                    cfg.validate(require_files=False)

    def test_implemented_control_enums_reject_unknown_values(self):
        cases = (
            ("gradient_geometry.projection_method", "mystery_projection"),
            ("gradient_geometry.global_cosine_weighting", "mystery_weighting"),
            ("replay.repair_sampling", "mystery_sampling"),
            ("repair.optimizer", "mystery_optimizer"),
            ("slice.initialization", "mystery_slice"),
            ("slice.fallback_initialization", "mystery_fallback"),
            ("slice.magnitude_mode", "mystery_magnitude"),
            ("activation_subspace.soft_ness_tau_mode", "mystery_tau"),
            ("replay.visual_latent_dtype", "float64"),
            ("replay.auxiliary_dtype", "int8"),
            ("exception.no_match_behavior", "nearest_anyway"),
            ("canary.unavailable_policy", "pretend_pass"),
        )
        for dotted_name, unsupported in cases:
            with self.subTest(field=dotted_name):
                cfg = FDPSCConfig(enabled=True)
                owner = cfg
                parts = dotted_name.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                setattr(owner, parts[-1], unsupported)
                with self.assertRaisesRegex(FDPSCConfigError, re.escape(dotted_name)):
                    cfg.validate(require_files=False)

    def test_spectral_weights_are_finite_nonnegative_and_not_all_zero(self):
        cfg = FDPSCConfig(enabled=True)
        cfg.spectral_surgery.current_weight = -1.0
        with self.assertRaisesRegex(FDPSCConfigError, "spectral_surgery weights"):
            cfg.validate(require_files=False)

        cfg = FDPSCConfig(enabled=True)
        cfg.spectral_surgery.current_weight = 0.0
        cfg.spectral_surgery.history_weight = 0.0
        cfg.spectral_surgery.anchor_weight = 0.0
        with self.assertRaisesRegex(FDPSCConfigError, "at least one positive weight"):
            cfg.validate(require_files=False)

    def test_sdc_anchor_regression_trigger_is_finite_nonnegative(self):
        cfg = FDPSCConfig(enabled=True)
        cfg.sdc.anchor_regression_trigger = 0.25
        cfg.validate(require_files=False)

        for invalid in (-0.01, float("inf"), float("-inf"), float("nan"), None, "bad"):
            with self.subTest(value=invalid):
                cfg = FDPSCConfig(enabled=True)
                cfg.sdc.anchor_regression_trigger = invalid
                with self.assertRaisesRegex(
                    FDPSCConfigError,
                    r"sdc\.anchor_regression_trigger must be finite and non-negative",
                ):
                    cfg.validate(require_files=False)

    def test_anchor_capacity_covers_fixed_gradient_batches(self):
        cfg = FDPSCConfig(enabled=True)
        cfg.anchor_data.windows = (
            cfg.gradient_geometry.anchor_batches
            * cfg.gradient_geometry.windows_per_batch
            - 1
        )
        with self.assertRaisesRegex(
            FDPSCConfigError,
            r"anchor_data\.windows must cover gradient_geometry\.anchor_batches",
        ):
            cfg.validate(require_files=False)

    def test_allowed_slow_ranks_cannot_exceed_maximum(self):
        cfg = FDPSCConfig(enabled=True)
        cfg.slow_lora.allowed_ranks = [32]
        cfg.slow_lora.initial_rank = 32
        cfg.slow_lora.maximum_rank = 16
        with self.assertRaisesRegex(
            FDPSCConfigError,
            "allowed_ranks must not exceed slow_lora.maximum_rank",
        ):
            cfg.validate(require_files=False)

        cfg.slow_lora.allowed_ranks = [8, 16]
        cfg.slow_lora.initial_rank = 32
        cfg.validate(require_files=False)

        cfg.slow_lora.allowed_ranks = [16]
        cfg.slow_lora.initial_rank = 8
        with self.assertRaisesRegex(
            FDPSCConfigError,
            "deterministically clipped slow_lora.allowed_ranks",
        ):
            cfg.validate(require_files=False)

    def test_canary_retention_covers_every_commit_plus_known_good(self):
        disabled_checkpoint = FDPSCConfig(enabled=True)
        disabled_checkpoint.canary.enabled = True
        disabled_checkpoint.checkpoint.enabled = False
        with self.assertRaisesRegex(
            FDPSCConfigError,
            "requires sidecar checkpointing",
        ):
            disabled_checkpoint.validate(require_files=False)

        cfg = FDPSCConfig(enabled=True)
        cfg.canary.enabled = True
        cfg.canary.every_episodes = 3
        cfg.checkpoint.enabled = True
        cfg.checkpoint.save_every_episodes = 1
        cfg.checkpoint.retention_versions = 3
        with self.assertRaisesRegex(
            FDPSCConfigError,
            "cover a canary period plus one known-good version",
        ):
            cfg.validate(require_files=False)

        cfg.checkpoint.retention_versions = 4
        cfg.validate(require_files=False)

    def test_anchor_files_remain_required_when_geometry_uses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixed.json"
            fixture.write_text("{}", encoding="utf-8")
            cfg = FDPSCConfig(enabled=True)
            cfg.gates.allow_unsafe_ablation = True
            cfg.gates.anchor_enabled = False
            cfg.external_eval_data.manifest_path = str(fixture)
            cfg.external_eval_data.calibration_path = str(fixture)
            cfg.external_eval_data.commit_query_path = str(fixture)
            cfg.external_eval_data.plasticity_support_path = str(fixture)
            cfg.external_eval_data.plasticity_query_path = str(fixture)
            with self.assertRaisesRegex(
                FDPSCConfigError,
                "required FD-PSC path is missing: anchor_manifest",
            ):
                cfg.validate(require_files=True)

    def test_every_declared_experiment_variant_still_validates(self):
        root = Path(__file__).resolve().parents[1]
        default = yaml.safe_load((root / "conf" / "fd_psc" / "default.yaml").read_text("utf-8"))
        disabled = yaml.safe_load((root / "conf" / "fd_psc" / "disabled.yaml").read_text("utf-8"))
        variants = yaml.safe_load(
            (root / "conf" / "fd_psc" / "experiments.yaml").read_text("utf-8")
        )["variants"]
        default["seed"] = 0
        disabled["seed"] = 0

        for variant_name, variant in variants.items():
            with self.subTest(variant=variant_name):
                raw = copy.deepcopy(default if variant["fd_config"] == "default" else disabled)
                for override in variant.get("overrides", ()):
                    key, separator, encoded = str(override).partition("=")
                    if not separator or not key.startswith("fd_psc."):
                        continue
                    owner = raw
                    parts = key.removeprefix("fd_psc.").split(".")
                    for part in parts[:-1]:
                        owner = owner[part]
                    owner[parts[-1]] = yaml.safe_load(encoded)
                FDPSCConfig.from_mapping(raw).validate(require_files=False)


if __name__ == "__main__":
    unittest.main()
