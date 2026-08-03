"""Typed configuration and validation for FD-PSC.

The public plan files use Hydra/OmegaConf, but the memory subsystem accepts a
plain mapping as well.  Keeping the validation here makes the CPU test fixture
independent from Hydra and prevents disabled FD-PSC runs from importing or
mutating model modules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class FDPSCConfigError(ValueError):
    """Raised when an FD-PSC configuration violates a hard invariant."""


@dataclass
class TargetModulesConfig:
    predictor_linear: bool = True
    post_backbone_projection_linear: bool = True
    action_encoder_linear: bool = False
    proprio_encoder_linear: bool = False
    exclude_frozen_backbone: bool = True
    require_active_forward_path: bool = True
    fail_on_empty_predictor_targets: bool = True
    fail_on_empty_projection_targets: bool = False
    require_projection_targets_if_head_exists: bool = True


@dataclass
class EpisodicLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    pilot_enabled: bool = True
    a_initialization: str = "kaiming_uniform"
    b_initialization: str = "zeros"


@dataclass
class ConvLoRAConfig:
    enabled: bool = True
    target_scope: str = "post_backbone_projection_head"
    parameterization: str = "flattened_kernel"
    groups_mode: str = "groupwise"


@dataclass
class SlowLoRAConfig:
    initial_rank: int = 8
    allowed_ranks: List[int] = field(default_factory=lambda: [8, 16, 24, 32])
    maximum_rank: int = 32
    spectral_energy_threshold: float = 0.99
    functional_error_threshold: float = 0.02


@dataclass
class GradientGeometryConfig:
    enabled: bool = True
    ema_beta: float = 0.8
    conflict_threshold: float = -0.1
    minimum_transitions: int = 3
    consecutive_conflicts: int = 2
    current_batches: int = 2
    history_batches: int = 2
    anchor_batches: int = 1
    windows_per_batch: int = 8
    projection_method: str = "dual_constraint"
    projection_scope: str = "per_logical_layer"
    global_cosine_weighting: str = "gradient_norm"
    hook_normalization: str = "exact_loss_gradient"
    epsilon: float = 1.0e-8
    c_pcgrad_coefficient: float = 1.0
    history_slack: float = 0.0
    anchor_slack: float = 0.0


@dataclass
class SliceConfig:
    enabled: bool = True
    trigger_only: bool = True
    initialization: str = "slice_exact"
    fallback_initialization: str = "slice_symmetric"
    rank: int = 8
    randomized_svd_oversampling: int = 2
    power_iterations: int = 1
    magnitude_mode: str = "first_step_match"
    maximum_scale: float = 10.0


@dataclass
class SDCConfig:
    enabled: bool = True
    event_triggered: bool = True
    check_every_replans: int = 4
    base_energy_threshold: float = 0.9
    drift_threshold: float = 0.25
    drift_consecutive_checks: int = 2
    drift_increase_tolerance: float = 0.01
    minimum_gamma: float = 0.1
    anchor_regression_trigger: float = 0.0


@dataclass
class SpectralSurgeryConfig:
    enabled: bool = True
    output_writing_layers_only: bool = True
    steps: int = 2
    learning_rate: float = 0.1
    minimum_scale: float = 0.75
    maximum_scale: float = 1.25
    preserve_spectral_l2_norm: bool = True
    current_weight: float = 1.0
    history_weight: float = 1.0
    anchor_weight: float = 1.0


@dataclass
class ActivationSubspaceConfig:
    enabled: bool = True
    maximum_rank: int = 64
    spectral_energy_threshold: float = 0.99
    forgetting_factor: float = 0.99
    soft_ness_tau_mode: str = "median"
    soft_ness_tau_fixed: Optional[float] = None
    soft_ness_tau_quantile: float = 0.5
    minimum_energy: float = 1.0e-8


@dataclass
class MergeConfig:
    soft_ness_enabled: bool = True
    shared_coefficients: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    safe_coefficients: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    use_context_similarity: bool = True
    use_gradient_similarity: bool = True
    use_residual_similarity: bool = True
    context_conflict_threshold: Optional[float] = None
    context_match_threshold: Optional[float] = None
    gradient_conflict_threshold: Optional[float] = -0.1
    gradient_match_threshold: Optional[float] = 0.1
    residual_match_threshold: Optional[float] = None
    selection_policy: str = "calibration_lexicographic"


@dataclass
class ReplayConfig:
    historical_windows: int = 512
    visual_latent_dtype: str = "float16"
    auxiliary_dtype: str = "float32"
    compression: str = "none"
    sampling: str = "balanced_uniform"
    repair_sampling: str = "grasp"
    maximum_context_clusters: int = 32
    new_cluster_similarity_threshold: float = 0.8
    minimum_windows_per_cluster: int = 4


@dataclass
class ExternalEvalDataConfig:
    source: str = "fixed_manifest"
    manifest_path: Optional[str] = None
    calibration_path: Optional[str] = None
    commit_query_path: Optional[str] = None
    plasticity_support_path: Optional[str] = None
    plasticity_query_path: Optional[str] = None
    report_test_path: Optional[str] = None
    representation: str = "frozen_backbone_latent"
    context_key: str = "context_identifier"
    context_source: str = "episode_metadata"
    split_unit: str = "trajectory"
    require_context_match: bool = True
    verify_checksums: bool = True
    missing_context_policy: str = "error"
    commit_query_policy: str = "single_final_proposal"


@dataclass
class AnchorDataConfig:
    source: str = "fixed_manifest"
    manifest_path: Optional[str] = None
    data_path: Optional[str] = None
    windows: int = 128
    verify_checksums: bool = True
    missing_policy: str = "error"


@dataclass
class RepairConfig:
    enabled: bool = True
    maximum_steps: int = 20
    candidate_steps: List[int] = field(default_factory=lambda: [5, 10, 20])
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-4
    windows_per_batch: int = 8
    current_weight: float = 1.0
    replay_weight: float = 1.0
    proximal_enabled: bool = True
    proximal_weight: float = 1.0
    pcgrad_enabled: bool = True
    checkpoint_schedule: str = "cumulative"
    proximal_layer_tags: List[str] = field(
        default_factory=lambda: [
            "encoder_projection",
            "attention_output",
            "mlp_output",
            "final_projection",
        ]
    )


@dataclass
class ExceptionConfig:
    enabled: bool = True
    maximum_adapters: int = 8
    routing: str = "nearest_prototype"
    minimum_route_similarity: float = 0.5
    no_match_behavior: str = "slow_only"
    minimum_calibration_fast_gain: float = 0.0
    minimum_commit_fast_gain: float = 0.0
    maximum_rank: int = 32
    local_replay_windows: int = 64
    routed_episode_update: str = "replace_exception"
    eviction_policy: str = "least_recently_used_then_lowest_gain"
    merge_similar_adapters: bool = False


@dataclass
class GatesConfig:
    allow_unsafe_ablation: bool = False
    current_gain_enabled: bool = True
    history_enabled: bool = True
    anchor_enabled: bool = True
    plasticity_enabled: bool = True
    functional_error_enabled: bool = True
    spectral_drift_enabled: bool = True
    fast_gain_retention: float = 0.8
    history_loss_tolerance: float = 0.0
    anchor_loss_tolerance: float = 0.0
    worst_context_loss_tolerance: float = 0.0
    plasticity_retention: float = 0.9
    drift_tolerance: float = 0.05
    absolute_numerical_tolerance: float = 1.0e-6
    relative_numerical_tolerance: float = 1.0e-5


@dataclass
class CanaryConfig:
    enabled: bool = False
    every_episodes: int = 10
    rollout_count: int = 4
    manifest_path: Optional[str] = None
    high_risk_rank_expansion: bool = True
    unavailable_policy: str = "report_unrun"


@dataclass
class CheckpointConfig:
    enabled: bool = True
    state_directory: str = "fd_psc_state"
    latest_pointer_path: str = "fd_psc_state_latest.json"
    resume_path: Optional[str] = None
    save_every_episodes: int = 1
    retention_versions: int = 20
    keep_commit_journal: bool = True
    atomic_write: bool = True


@dataclass
class LoggingConfig:
    per_layer_metrics: bool = True
    save_candidate_reports: bool = True
    save_gradient_statistics: bool = True


_NESTED_TYPES = {
    "target_modules": TargetModulesConfig,
    "episodic_lora": EpisodicLoRAConfig,
    "conv_lora": ConvLoRAConfig,
    "slow_lora": SlowLoRAConfig,
    "gradient_geometry": GradientGeometryConfig,
    "slice": SliceConfig,
    "sdc": SDCConfig,
    "spectral_surgery": SpectralSurgeryConfig,
    "activation_subspace": ActivationSubspaceConfig,
    "merge": MergeConfig,
    "replay": ReplayConfig,
    "external_eval_data": ExternalEvalDataConfig,
    "anchor_data": AnchorDataConfig,
    "repair": RepairConfig,
    "exception": ExceptionConfig,
    "gates": GatesConfig,
    "canary": CanaryConfig,
    "checkpoint": CheckpointConfig,
    "logging": LoggingConfig,
}


@dataclass
class FDPSCConfig:
    enabled: bool = False
    seed: int = 0
    # Explicitly-labelled comparison modes share the same target manifest and
    # ranks.  Only ``fd_psc`` may be reported as the full method.
    run_mode: str = "fd_psc"
    target_modules: TargetModulesConfig = field(default_factory=TargetModulesConfig)
    episodic_lora: EpisodicLoRAConfig = field(default_factory=EpisodicLoRAConfig)
    conv_lora: ConvLoRAConfig = field(default_factory=ConvLoRAConfig)
    slow_lora: SlowLoRAConfig = field(default_factory=SlowLoRAConfig)
    gradient_geometry: GradientGeometryConfig = field(default_factory=GradientGeometryConfig)
    slice: SliceConfig = field(default_factory=SliceConfig)
    sdc: SDCConfig = field(default_factory=SDCConfig)
    spectral_surgery: SpectralSurgeryConfig = field(default_factory=SpectralSurgeryConfig)
    activation_subspace: ActivationSubspaceConfig = field(default_factory=ActivationSubspaceConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    external_eval_data: ExternalEvalDataConfig = field(default_factory=ExternalEvalDataConfig)
    anchor_data: AnchorDataConfig = field(default_factory=AnchorDataConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    exception_: ExceptionConfig = field(default_factory=ExceptionConfig, metadata={"external_name": "exception"})
    gates: GatesConfig = field(default_factory=GatesConfig)
    canary: CanaryConfig = field(default_factory=CanaryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def exception(self) -> ExceptionConfig:
        return self.exception_

    @classmethod
    def from_mapping(cls, value: Optional[Any]) -> "FDPSCConfig":
        if value is None:
            return cls(enabled=False)
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_container"):
            value = value.to_container(resolve=True)
        elif not isinstance(value, Mapping) and hasattr(value, "items"):
            value = dict(value.items())
        if not isinstance(value, Mapping):
            raise FDPSCConfigError(f"fd_psc must be a mapping, got {type(value).__name__}")
        raw = dict(value)
        if "fd_psc" in raw and isinstance(raw["fd_psc"], Mapping):
            raw = dict(raw["fd_psc"])
        known = {f.name for f in dataclasses.fields(cls)} | {"exception"}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise FDPSCConfigError(f"unknown fd_psc fields: {unknown}")
        kwargs: Dict[str, Any] = {}
        for key, item in raw.items():
            attr = "exception_" if key == "exception" else key
            nested_type = _NESTED_TYPES.get(key)
            if nested_type is not None:
                if isinstance(item, nested_type):
                    kwargs[attr] = item
                elif isinstance(item, Mapping) or hasattr(item, "items"):
                    item_dict = dict(item.items())
                    field_names = {f.name for f in dataclasses.fields(nested_type)}
                    extra = sorted(set(item_dict) - field_names)
                    if extra:
                        raise FDPSCConfigError(f"unknown fd_psc.{key} fields: {extra}")
                    kwargs[attr] = nested_type(**item_dict)
                else:
                    raise FDPSCConfigError(f"fd_psc.{key} must be a mapping")
            else:
                kwargs[attr] = item
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["exception"] = result.pop("exception_")
        return result

    def identity_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def persistence_identity_hash(self) -> str:
        """Algorithm identity used by sidecar resume validation.

        ``resume_path`` selects *which already-committed sidecar to read*; it
        is not an algorithm choice and necessarily changes from null on the
        first run to a path on a resumed run.  Excluding only this selector
        keeps every numerical/data/gate setting strict while permitting the
        documented resume workflow.
        """

        value = self.to_dict()
        value["checkpoint"]["resume_path"] = None
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_paths(self, runtime_output_dir: Path) -> Dict[str, Optional[Path]]:
        root = Path(runtime_output_dir).expanduser().resolve()

        def resolved(raw: Optional[str]) -> Optional[Path]:
            if raw is None:
                return None
            path = Path(raw).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        return {
            "external_manifest": resolved(self.external_eval_data.manifest_path),
            "calibration": resolved(self.external_eval_data.calibration_path),
            "commit_query": resolved(self.external_eval_data.commit_query_path),
            "plasticity_support": resolved(self.external_eval_data.plasticity_support_path),
            "plasticity_query": resolved(self.external_eval_data.plasticity_query_path),
            "report_test": resolved(self.external_eval_data.report_test_path),
            "anchor_manifest": resolved(self.anchor_data.manifest_path),
            "anchor_data": resolved(self.anchor_data.data_path),
            "canary_manifest": resolved(self.canary.manifest_path),
            "state_directory": resolved(self.checkpoint.state_directory),
            "latest_pointer": resolved(self.checkpoint.latest_pointer_path),
            "resume": resolved(self.checkpoint.resume_path),
        }

    def validate(self, runtime_output_dir: Optional[Path] = None, require_files: bool = True) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise FDPSCConfigError("fd_psc.seed must be a non-negative integer")
        if not self.enabled:
            return

        if self.run_mode not in {"fd_psc", "episodic_reset", "accumulate", "plain_svd"}:
            raise FDPSCConfigError(
                "fd_psc.run_mode must be fd_psc, episodic_reset, accumulate, or plain_svd"
            )
        if self.gradient_geometry.projection_method not in {
            "dual_constraint",
            "c_pcgrad",
            "per_step_c_pcgrad",
        }:
            raise FDPSCConfigError(
                "gradient_geometry.projection_method must be dual_constraint, "
                "c_pcgrad, or per_step_c_pcgrad"
            )
        if self.gradient_geometry.global_cosine_weighting not in {
            "gradient_norm",
            "parameter_count",
            "uniform",
        }:
            raise FDPSCConfigError(
                "gradient_geometry.global_cosine_weighting must be "
                "gradient_norm, parameter_count, or uniform"
            )
        if self.replay.repair_sampling not in {"grasp", "balanced_uniform"}:
            raise FDPSCConfigError(
                "replay.repair_sampling must be grasp or balanced_uniform"
            )
        if self.repair.optimizer not in {"adamw", "adam", "sgd"}:
            raise FDPSCConfigError("repair.optimizer must be adamw, adam, or sgd")
        if self.slice.initialization not in {"slice_exact", "slice_symmetric"}:
            raise FDPSCConfigError(
                "slice.initialization must be slice_exact or slice_symmetric"
            )
        if self.slice.fallback_initialization != "slice_symmetric":
            raise FDPSCConfigError(
                "slice.fallback_initialization must be slice_symmetric"
            )
        if self.slice.magnitude_mode not in {"first_step_match", "none"}:
            raise FDPSCConfigError(
                "slice.magnitude_mode must be first_step_match or none"
            )
        if self.activation_subspace.soft_ness_tau_mode not in {
            "median",
            "quantile",
            "fixed",
        }:
            raise FDPSCConfigError(
                "activation_subspace.soft_ness_tau_mode must be median, quantile, or fixed"
            )
        if self.activation_subspace.soft_ness_tau_mode == "fixed":
            tau = self.activation_subspace.soft_ness_tau_fixed
            if tau is None or not math.isfinite(float(tau)) or float(tau) <= 0:
                raise FDPSCConfigError(
                    "activation_subspace.soft_ness_tau_fixed must be finite and positive "
                    "when soft_ness_tau_mode=fixed"
                )
        replay_dtypes = {"float16", "float32", "bfloat16"}
        if self.replay.visual_latent_dtype not in replay_dtypes:
            raise FDPSCConfigError(
                "replay.visual_latent_dtype must be float16, float32, or bfloat16"
            )
        if self.replay.auxiliary_dtype not in replay_dtypes:
            raise FDPSCConfigError(
                "replay.auxiliary_dtype must be float16, float32, or bfloat16"
            )
        if self.exception.no_match_behavior != "slow_only":
            raise FDPSCConfigError("exception.no_match_behavior must be slow_only")
        if self.canary.unavailable_policy not in {"report_unrun", "error"}:
            raise FDPSCConfigError(
                "canary.unavailable_policy must be report_unrun or error"
            )
        if not isinstance(self.external_eval_data.context_key, str) or not self.external_eval_data.context_key:
            raise FDPSCConfigError("external_eval_data.context_key must be a non-empty string")

        def positive_int(name: str, value: int) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise FDPSCConfigError(f"{name} must be a positive integer")

        def nonnegative_int(name: str, value: int) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FDPSCConfigError(f"{name} must be a non-negative integer")

        def finite_positive(name: str, value: float) -> None:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise FDPSCConfigError(f"{name} must be finite and positive")

        def interval(name: str, value: float, lo: float, hi: float, *, left_open: bool = False) -> None:
            x = float(value)
            ok = math.isfinite(x) and (x > lo if left_open else x >= lo) and x <= hi
            if not ok:
                left = "(" if left_open else "["
                raise FDPSCConfigError(f"{name} must be in {left}{lo}, {hi}]")

        def implemented_value(name: str, value: Any, supported: Any) -> None:
            """Reject declarative controls that the runtime does not yet consume.

            Keeping these fields in the public schema documents the intended
            protocol, but silently accepting another value would label a run as
            an ablation that never actually happened.  The supported values are
            the V2 defaults and therefore keep the shipped configuration valid.
            """

            if value != supported:
                raise FDPSCConfigError(
                    f"{name}={value!r} is not implemented; "
                    f"the only supported value is {supported!r}"
                )

        for name, value, supported in (
            (
                "target_modules.exclude_frozen_backbone",
                self.target_modules.exclude_frozen_backbone,
                True,
            ),
            ("episodic_lora.pilot_enabled", self.episodic_lora.pilot_enabled, True),
            (
                "episodic_lora.a_initialization",
                self.episodic_lora.a_initialization,
                "kaiming_uniform",
            ),
            ("episodic_lora.b_initialization", self.episodic_lora.b_initialization, "zeros"),
            (
                "gradient_geometry.projection_scope",
                self.gradient_geometry.projection_scope,
                "per_logical_layer",
            ),
            (
                "gradient_geometry.hook_normalization",
                self.gradient_geometry.hook_normalization,
                "exact_loss_gradient",
            ),
            ("slice.trigger_only", self.slice.trigger_only, True),
            ("merge.selection_policy", self.merge.selection_policy, "calibration_lexicographic"),
            ("replay.compression", self.replay.compression, "none"),
            ("replay.sampling", self.replay.sampling, "balanced_uniform"),
            ("external_eval_data.source", self.external_eval_data.source, "fixed_manifest"),
            (
                "external_eval_data.context_source",
                self.external_eval_data.context_source,
                "episode_metadata",
            ),
            ("external_eval_data.split_unit", self.external_eval_data.split_unit, "trajectory"),
            (
                "external_eval_data.require_context_match",
                self.external_eval_data.require_context_match,
                True,
            ),
            (
                "external_eval_data.missing_context_policy",
                self.external_eval_data.missing_context_policy,
                "error",
            ),
            ("anchor_data.source", self.anchor_data.source, "fixed_manifest"),
            ("anchor_data.verify_checksums", self.anchor_data.verify_checksums, True),
            ("anchor_data.missing_policy", self.anchor_data.missing_policy, "error"),
            ("exception.routing", self.exception.routing, "nearest_prototype"),
            (
                "exception.routed_episode_update",
                self.exception.routed_episode_update,
                "replace_exception",
            ),
            (
                "exception.eviction_policy",
                self.exception.eviction_policy,
                "least_recently_used_then_lowest_gain",
            ),
            ("exception.merge_similar_adapters", self.exception.merge_similar_adapters, False),
            ("checkpoint.save_every_episodes", self.checkpoint.save_every_episodes, 1),
            ("checkpoint.keep_commit_journal", self.checkpoint.keep_commit_journal, True),
            ("checkpoint.atomic_write", self.checkpoint.atomic_write, True),
            ("logging.per_layer_metrics", self.logging.per_layer_metrics, True),
            ("logging.save_candidate_reports", self.logging.save_candidate_reports, True),
            ("logging.save_gradient_statistics", self.logging.save_gradient_statistics, True),
        ):
            implemented_value(name, value, supported)

        positive_int("episodic_lora.rank", self.episodic_lora.rank)
        positive_int("slice.rank", self.slice.rank)
        positive_int("slow_lora.maximum_rank", self.slow_lora.maximum_rank)
        positive_int("slow_lora.initial_rank", self.slow_lora.initial_rank)
        finite_positive("episodic_lora.alpha", self.episodic_lora.alpha)
        interval("episodic_lora.dropout", self.episodic_lora.dropout, 0.0, 1.0)
        allowed = self.slow_lora.allowed_ranks
        if not allowed or allowed != sorted(set(allowed)) or any(r <= 0 for r in allowed):
            raise FDPSCConfigError("slow_lora.allowed_ranks must be non-empty, positive, unique, and strictly increasing")
        if any(int(rank) > int(self.slow_lora.maximum_rank) for rank in allowed):
            raise FDPSCConfigError(
                "slow_lora.allowed_ranks must not exceed slow_lora.maximum_rank"
            )
        clipped_allowed = {int(rank) for rank in allowed}
        clipped_initial = min(
            int(self.slow_lora.initial_rank), int(self.slow_lora.maximum_rank)
        )
        if clipped_initial not in clipped_allowed:
            raise FDPSCConfigError(
                "min(slow_lora.initial_rank, slow_lora.maximum_rank) must be in "
                "the deterministically clipped slow_lora.allowed_ranks"
            )

        for name, value in (
            ("slow_lora.spectral_energy_threshold", self.slow_lora.spectral_energy_threshold),
            ("sdc.base_energy_threshold", self.sdc.base_energy_threshold),
            ("activation_subspace.spectral_energy_threshold", self.activation_subspace.spectral_energy_threshold),
        ):
            interval(name, value, 0.0, 1.0, left_open=True)
        for name, value in (
            ("gradient_geometry.ema_beta", self.gradient_geometry.ema_beta),
            ("activation_subspace.forgetting_factor", self.activation_subspace.forgetting_factor),
            ("activation_subspace.soft_ness_tau_quantile", self.activation_subspace.soft_ness_tau_quantile),
            ("replay.new_cluster_similarity_threshold", self.replay.new_cluster_similarity_threshold),
            ("exception.minimum_route_similarity", self.exception.minimum_route_similarity),
            ("gates.fast_gain_retention", self.gates.fast_gain_retention),
            ("gates.plasticity_retention", self.gates.plasticity_retention),
            ("sdc.minimum_gamma", self.sdc.minimum_gamma),
        ):
            interval(name, value, 0.0, 1.0)

        for name, value in (
            ("gradient_geometry.epsilon", self.gradient_geometry.epsilon),
            ("slice.maximum_scale", self.slice.maximum_scale),
            ("spectral_surgery.learning_rate", self.spectral_surgery.learning_rate),
            ("activation_subspace.minimum_energy", self.activation_subspace.minimum_energy),
            ("repair.learning_rate", self.repair.learning_rate),
            ("gates.absolute_numerical_tolerance", self.gates.absolute_numerical_tolerance),
            ("gates.relative_numerical_tolerance", self.gates.relative_numerical_tolerance),
        ):
            finite_positive(name, value)

        try:
            anchor_regression_trigger = float(self.sdc.anchor_regression_trigger)
        except (TypeError, ValueError) as exc:
            raise FDPSCConfigError(
                "sdc.anchor_regression_trigger must be finite and non-negative"
            ) from exc
        if (
            not math.isfinite(anchor_regression_trigger)
            or anchor_regression_trigger < 0.0
        ):
            raise FDPSCConfigError(
                "sdc.anchor_regression_trigger must be finite and non-negative"
            )

        spectral_weights = (
            self.spectral_surgery.current_weight,
            self.spectral_surgery.history_weight,
            self.spectral_surgery.anchor_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in spectral_weights):
            raise FDPSCConfigError("spectral_surgery weights must be finite and non-negative")
        if self.spectral_surgery.enabled and sum(float(value) for value in spectral_weights) <= 0:
            raise FDPSCConfigError("enabled spectral_surgery requires at least one positive weight")

        for name, value in (
            ("gradient_geometry.minimum_transitions", self.gradient_geometry.minimum_transitions),
            ("gradient_geometry.consecutive_conflicts", self.gradient_geometry.consecutive_conflicts),
            ("gradient_geometry.current_batches", self.gradient_geometry.current_batches),
            ("gradient_geometry.history_batches", self.gradient_geometry.history_batches),
            ("gradient_geometry.anchor_batches", self.gradient_geometry.anchor_batches),
            ("gradient_geometry.windows_per_batch", self.gradient_geometry.windows_per_batch),
            ("sdc.check_every_replans", self.sdc.check_every_replans),
            ("sdc.drift_consecutive_checks", self.sdc.drift_consecutive_checks),
            ("activation_subspace.maximum_rank", self.activation_subspace.maximum_rank),
            ("repair.windows_per_batch", self.repair.windows_per_batch),
            ("canary.every_episodes", self.canary.every_episodes),
            ("canary.rollout_count", self.canary.rollout_count),
            ("checkpoint.save_every_episodes", self.checkpoint.save_every_episodes),
            ("checkpoint.retention_versions", self.checkpoint.retention_versions),
        ):
            positive_int(name, value)
        for name, value in (
            ("slice.randomized_svd_oversampling", self.slice.randomized_svd_oversampling),
            ("slice.power_iterations", self.slice.power_iterations),
            ("spectral_surgery.steps", self.spectral_surgery.steps),
            ("replay.historical_windows", self.replay.historical_windows),
            ("replay.maximum_context_clusters", self.replay.maximum_context_clusters),
            ("replay.minimum_windows_per_cluster", self.replay.minimum_windows_per_cluster),
            ("anchor_data.windows", self.anchor_data.windows),
            ("repair.maximum_steps", self.repair.maximum_steps),
            ("exception.maximum_adapters", self.exception.maximum_adapters),
            ("exception.local_replay_windows", self.exception.local_replay_windows),
        ):
            nonnegative_int(name, value)

        if self.slice.enabled and not self.episodic_lora.pilot_enabled:
            raise FDPSCConfigError("slice.enabled requires episodic_lora.pilot_enabled")
        if (self.gradient_geometry.enabled or self.slice.enabled or self.sdc.enabled) and self.episodic_lora.dropout != 0:
            raise FDPSCConfigError("gradient geometry, SLICE, and SDC require episodic_lora.dropout=0")
        if self.slice.enabled and not self.gradient_geometry.enabled:
            raise FDPSCConfigError("slice.enabled requires gradient_geometry.enabled")
        if self.sdc.enabled and not self.gradient_geometry.enabled:
            raise FDPSCConfigError("sdc.enabled requires effective-weight gradient collection")
        if self.conv_lora.enabled:
            if self.conv_lora.target_scope != "post_backbone_projection_head":
                raise FDPSCConfigError("ConvLoRA target_scope must be post_backbone_projection_head")
            if self.conv_lora.parameterization != "flattened_kernel" or self.conv_lora.groups_mode != "groupwise":
                raise FDPSCConfigError("ConvLoRA requires flattened_kernel/groupwise semantics")
        if self.external_eval_data.commit_query_policy != "single_final_proposal":
            raise FDPSCConfigError("commit_query_policy must be single_final_proposal")
        if self.external_eval_data.representation != "frozen_backbone_latent":
            raise FDPSCConfigError("the compliant path requires frozen_backbone_latent external data")

        if not self.merge.shared_coefficients or not self.merge.safe_coefficients:
            raise FDPSCConfigError("merge coefficient grids must be non-empty")
        for name, values in (
            ("merge.shared_coefficients", self.merge.shared_coefficients),
            ("merge.safe_coefficients", self.merge.safe_coefficients),
        ):
            if any(not math.isfinite(float(x)) for x in values):
                raise FDPSCConfigError(f"{name} must contain only finite values")
        similarity_thresholds = (
            ("merge.context_conflict_threshold", self.merge.context_conflict_threshold),
            ("merge.context_match_threshold", self.merge.context_match_threshold),
            ("merge.gradient_conflict_threshold", self.merge.gradient_conflict_threshold),
            ("merge.gradient_match_threshold", self.merge.gradient_match_threshold),
            ("merge.residual_match_threshold", self.merge.residual_match_threshold),
        )
        for name, value in similarity_thresholds:
            if value is not None:
                interval(name, value, -1.0, 1.0)
        for signal_name, conflict, match in (
            (
                "context",
                self.merge.context_conflict_threshold,
                self.merge.context_match_threshold,
            ),
            (
                "gradient",
                self.merge.gradient_conflict_threshold,
                self.merge.gradient_match_threshold,
            ),
        ):
            if (
                conflict is not None
                and match is not None
                and float(conflict) >= float(match)
            ):
                raise FDPSCConfigError(
                    f"merge {signal_name} conflict threshold must be less than its match threshold"
                )
        c = float(self.gradient_geometry.c_pcgrad_coefficient)
        if not self.gates.allow_unsafe_ablation and not (0.0 <= c <= 1.0):
            raise FDPSCConfigError("c_pcgrad_coefficient outside [0,1] requires allow_unsafe_ablation")
        for name, value in (
            ("gradient_geometry.history_slack", self.gradient_geometry.history_slack),
            ("gradient_geometry.anchor_slack", self.gradient_geometry.anchor_slack),
            ("gates.history_loss_tolerance", self.gates.history_loss_tolerance),
            ("gates.anchor_loss_tolerance", self.gates.anchor_loss_tolerance),
            ("gates.worst_context_loss_tolerance", self.gates.worst_context_loss_tolerance),
            ("gates.drift_tolerance", self.gates.drift_tolerance),
            ("slow_lora.functional_error_threshold", self.slow_lora.functional_error_threshold),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise FDPSCConfigError(f"{name} must be finite and non-negative")
        if not 0 < self.spectral_surgery.minimum_scale <= self.spectral_surgery.maximum_scale:
            raise FDPSCConfigError("spectral surgery scales must satisfy 0 < minimum <= maximum")

        candidate_steps = self.repair.candidate_steps
        if self.repair.enabled:
            if not candidate_steps or candidate_steps != sorted(set(candidate_steps)) or any(x <= 0 for x in candidate_steps):
                raise FDPSCConfigError("repair.candidate_steps must be positive, unique, and strictly increasing")
            if candidate_steps[-1] > self.repair.maximum_steps:
                raise FDPSCConfigError("repair.candidate_steps cannot exceed maximum_steps")
            if self.repair.checkpoint_schedule != "cumulative":
                raise FDPSCConfigError("the compliant repair schedule is cumulative")
            if self.repair.proximal_enabled and not self.repair.proximal_layer_tags:
                raise FDPSCConfigError("proximal repair requires at least one layer tag")

        gate_values = [
            self.gates.current_gain_enabled,
            self.gates.history_enabled,
            self.gates.anchor_enabled,
            self.gates.plasticity_enabled,
            self.gates.functional_error_enabled,
            self.gates.spectral_drift_enabled,
        ]
        if not all(gate_values) and not self.gates.allow_unsafe_ablation:
            raise FDPSCConfigError("disabling Gates 1-6 requires gates.allow_unsafe_ablation=true")

        minimum_retention = self.canary.every_episodes + 1
        if self.canary.enabled and not self.checkpoint.enabled:
            raise FDPSCConfigError(
                "enabled periodic canary requires sidecar checkpointing for journal rollback"
            )
        if (
            self.canary.enabled
            and self.checkpoint.retention_versions < minimum_retention
        ):
            raise FDPSCConfigError(
                "checkpoint.retention_versions must cover a canary period plus one known-good version"
            )

        anchor_required = (
            self.gates.anchor_enabled
            or self.gradient_geometry.enabled
            or (
                self.spectral_surgery.enabled
                and self.spectral_surgery.anchor_weight > 0
            )
        )
        if anchor_required and self.anchor_data.windows <= 0:
            raise FDPSCConfigError(
                "anchor_data.windows must be positive when an enabled component uses anchor data"
            )
        anchor_gradient_required = self.gradient_geometry.enabled or (
            self.spectral_surgery.enabled
            and self.spectral_surgery.anchor_weight > 0
        )
        if anchor_gradient_required:
            required_anchor_gradient_windows = (
                self.gradient_geometry.anchor_batches
                * self.gradient_geometry.windows_per_batch
            )
            if self.anchor_data.windows < required_anchor_gradient_windows:
                raise FDPSCConfigError(
                    "anchor_data.windows must cover gradient_geometry.anchor_batches "
                    "* gradient_geometry.windows_per_batch"
                )

        if require_files:
            runtime_root = Path.cwd() if runtime_output_dir is None else Path(runtime_output_dir)
            paths = self.resolve_paths(runtime_root)
            required = {
                "external_manifest": paths["external_manifest"],
                "calibration": paths["calibration"],
                "commit_query": paths["commit_query"],
                "plasticity_support": paths["plasticity_support"] if self.gates.plasticity_enabled else None,
                "plasticity_query": paths["plasticity_query"] if self.gates.plasticity_enabled else None,
                "anchor_manifest": paths["anchor_manifest"] if anchor_required else None,
                "anchor_data": paths["anchor_data"] if anchor_required else None,
                "canary_manifest": paths["canary_manifest"] if self.canary.enabled else None,
            }
            for name, path in required.items():
                if path is None:
                    if name in ("plasticity_support", "plasticity_query") and not self.gates.plasticity_enabled:
                        continue
                    if name in ("anchor_manifest", "anchor_data") and not anchor_required:
                        continue
                    if name == "canary_manifest" and not self.canary.enabled:
                        continue
                    raise FDPSCConfigError(f"required FD-PSC path is missing: {name}")
                if not path.is_file():
                    raise FDPSCConfigError(f"required FD-PSC file is not readable: {path}")


def minimal_test_config(**overrides: Any) -> FDPSCConfig:
    """Return a dependency-light enabled config for algorithm/unit tests.

    This helper is intentionally not used by production configuration.  It
    disables components whose hard contract requires fixed external datasets;
    callers can selectively re-enable them with fixture manifests.
    """

    raw: Dict[str, Any] = {
        "enabled": True,
        "gradient_geometry": {"enabled": False},
        "slice": {"enabled": False},
        "sdc": {"enabled": False},
        "spectral_surgery": {"enabled": False},
        "activation_subspace": {"enabled": False},
        "merge": {"soft_ness_enabled": False},
        "repair": {"enabled": False},
        "exception": {"enabled": False, "maximum_adapters": 0, "local_replay_windows": 0},
        "gates": {
            "allow_unsafe_ablation": True,
            "current_gain_enabled": False,
            "history_enabled": False,
            "anchor_enabled": False,
            "plasticity_enabled": False,
            "functional_error_enabled": False,
            "spectral_drift_enabled": False,
        },
        "checkpoint": {"enabled": False},
    }
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(raw.get(key), Mapping):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    cfg = FDPSCConfig.from_mapping(raw)
    cfg.validate(require_files=False)
    return cfg


__all__ = [
    "FDPSCConfig",
    "FDPSCConfigError",
    "minimal_test_config",
]
