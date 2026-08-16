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
    # V1 continues to select from ``allowed_ranks``.  FSD V2 instead requires
    # one deterministic compression rank and validates that this field is set.
    persistent_rank: Optional[int] = None
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
class RTRCConfig:
    enabled: bool = True
    budget_fraction_initial: float = 0.20
    budget_fraction_minimum: float = 0.02
    budget_fraction_maximum: float = 1.00
    geometry_windows: int = 128
    geometry_maximum_rank: int = 64
    geometry_energy_threshold: float = 0.999
    minimum_energy: float = 1.0e-8
    use_shared_dual: bool = True
    layer_weight_mode: str = "relative_output_energy"
    tail_mode: str = "conservative_isotropic"
    bisection_iterations: int = 80
    bisection_relative_tolerance: float = 1.0e-7
    epsilon: float = 1.0e-8


@dataclass
class RawReplayConfig:
    historical_windows: int = 512
    store_model_input_obs: bool = True
    store_frozen_visual_latent_cache: bool = False
    image_storage_dtype: str = "uint8_or_source"
    auxiliary_dtype: str = "float32"
    maximum_context_clusters: int = 32
    minimum_windows_per_cluster: int = 4


@dataclass
class AdaptiveBudgetConfig:
    enabled: bool = True
    controller_learning_rate: float = 0.25
    plasticity_loss_target: float = 0.10
    history_regression_target: float = 0.01
    history_weight: float = 2.0
    error_clip: float = 1.0
    minimum_wake_gain: float = 1.0e-8


@dataclass
class CoreStorageConfig:
    """Storage contract for the dense long-term core increment."""

    mode: str = "dense_delta"


@dataclass
class DeepSleepConfig:
    """Capacity-overflow consolidation and residual-rank recycling."""

    # Phase 2 is a live part of the shipped FSD V2 algorithm.  The trigger is
    # still capacity-driven, so enabling it does not make distillation run on
    # every commit.
    enabled: bool = True
    trigger_relative_rank_error: float = 0.05
    trigger_consecutive_commits: int = 3
    minimum_replay_windows: int = 64
    strategy: str = "residual_distill"
    maximum_steps: int = 100
    learning_rate: float = 1.0e-5
    output_residual_weight: float = 1.0
    hidden_residual_weight: float = 0.25
    current_task_weight: float = 0.25
    residual_rank: int = 8
    batch_size: int = 8
    hidden_layer_maximum: int = 4
    validation_fraction: float = 0.20
    minimum_validation_windows: int = 8
    functional_error_threshold: float = 0.02
    functional_error_absolute_tolerance: float = 1.0e-8
    epsilon: float = 1.0e-8
    core_storage: CoreStorageConfig = field(default_factory=CoreStorageConfig)


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
    "rtrc": RTRCConfig,
    "raw_replay": RawReplayConfig,
    "adaptive_budget": AdaptiveBudgetConfig,
    "deep_sleep": DeepSleepConfig,
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
    rtrc: RTRCConfig = field(default_factory=RTRCConfig)
    raw_replay: RawReplayConfig = field(default_factory=RawReplayConfig)
    adaptive_budget: AdaptiveBudgetConfig = field(
        default_factory=AdaptiveBudgetConfig
    )
    deep_sleep: DeepSleepConfig = field(default_factory=DeepSleepConfig)
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
                    if nested_type is DeepSleepConfig:
                        core_storage = item_dict.get("core_storage")
                        if isinstance(core_storage, Mapping) or hasattr(
                            core_storage, "items"
                        ):
                            core_mapping = dict(core_storage.items())
                            core_fields = {
                                f.name for f in dataclasses.fields(CoreStorageConfig)
                            }
                            core_extra = sorted(set(core_mapping) - core_fields)
                            if core_extra:
                                raise FDPSCConfigError(
                                    "unknown fd_psc.deep_sleep.core_storage fields: "
                                    f"{core_extra}"
                                )
                            item_dict["core_storage"] = CoreStorageConfig(
                                **core_mapping
                            )
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

    def v2_persistence_identity(self) -> Dict[str, Any]:
        """Return only configuration that changes the FSD V2 algorithm.

        External evaluation, legacy FD-PSC controls, reporting, and checkpoint
        locations are deliberately absent.  In particular, attaching an
        offline report manifest must never make an otherwise identical V2
        sidecar incompatible with resume.
        """

        if self.run_mode != "fsd_v2":
            raise FDPSCConfigError(
                "v2_persistence_identity is only defined for run_mode=fsd_v2"
            )
        return {
            "identity_schema_version": 2,
            "algorithm_version": "fsd_v2",
            "seed": self.seed,
            "target_modules": asdict(self.target_modules),
            "episodic_lora": asdict(self.episodic_lora),
            "conv_lora": asdict(self.conv_lora),
            "slow_lora": {
                "persistent_rank": self.slow_lora.persistent_rank,
            },
            "rtrc": asdict(self.rtrc),
            "raw_replay": asdict(self.raw_replay),
            "adaptive_budget": asdict(self.adaptive_budget),
            "deep_sleep": asdict(self.deep_sleep),
        }

    def v2_persistence_identity_hash(self) -> str:
        payload = json.dumps(
            self.v2_persistence_identity(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def persistence_identity_hash(self) -> str:
        """Algorithm identity used by sidecar resume validation.

        ``resume_path`` selects *which already-committed sidecar to read*; it
        is not an algorithm choice and necessarily changes from null on the
        first run to a path on a resumed run.  Excluding only this selector
        keeps every numerical/data/gate setting strict while permitting the
        documented resume workflow.
        """

        if self.run_mode == "fsd_v2":
            return self.v2_persistence_identity_hash()

        value = self.to_dict()
        # These fields did not exist in the V1 schema and have no V1 runtime
        # semantics.  Removing them preserves existing V1 sidecar identities.
        value.pop("rtrc", None)
        value.pop("raw_replay", None)
        value.pop("adaptive_budget", None)
        value.pop("deep_sleep", None)
        value["slow_lora"].pop("persistent_rank", None)
        value["checkpoint"]["resume_path"] = None
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_v2_paths(self, runtime_output_dir: Path) -> Dict[str, Optional[Path]]:
        """Resolve only FSD V2 sidecar paths.

        Keeping this separate from :meth:`resolve_paths` makes it impossible
        for V2 construction/validation to accidentally consume an external,
        anchor, plasticity, or commit-query path.
        """

        root = Path(runtime_output_dir).expanduser().resolve()

        def resolved(raw: Optional[str]) -> Optional[Path]:
            if raw is None:
                return None
            path = Path(raw).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        return {
            "state_directory": resolved(self.checkpoint.state_directory),
            "latest_pointer": resolved(self.checkpoint.latest_pointer_path),
            "resume": resolved(self.checkpoint.resume_path),
        }

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

    def _validate_fsd_v2(
        self,
        *,
        runtime_output_dir: Optional[Path],
        require_files: bool,
    ) -> None:
        """Validate only controls consumed by the first FSD V2 path."""

        def require_bool(name: str, value: Any) -> None:
            if not isinstance(value, bool):
                raise FDPSCConfigError(f"{name} must be boolean")

        def positive_int(name: str, value: Any) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise FDPSCConfigError(f"{name} must be a positive integer")

        def finite(name: str, value: Any) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise FDPSCConfigError(f"{name} must be finite") from exc
            if not math.isfinite(result):
                raise FDPSCConfigError(f"{name} must be finite")
            return result

        def finite_positive(name: str, value: Any) -> float:
            result = finite(name, value)
            if result <= 0.0:
                raise FDPSCConfigError(f"{name} must be finite and positive")
            return result

        # First-round FSD V2 has one explicit algorithm path.  Legacy
        # correction/search/gating switches are not merely unused defaults:
        # accepting them as enabled would falsely advertise behavior that the
        # V2 runtime does not execute.
        disabled_legacy_controls = (
            ("gradient_geometry.enabled", self.gradient_geometry.enabled),
            ("slice.enabled", self.slice.enabled),
            ("sdc.enabled", self.sdc.enabled),
            ("spectral_surgery.enabled", self.spectral_surgery.enabled),
            ("activation_subspace.enabled", self.activation_subspace.enabled),
            ("merge.soft_ness_enabled", self.merge.soft_ness_enabled),
            ("repair.enabled", self.repair.enabled),
            ("exception.enabled", self.exception.enabled),
            ("gates.allow_unsafe_ablation", self.gates.allow_unsafe_ablation),
            ("gates.current_gain_enabled", self.gates.current_gain_enabled),
            ("gates.history_enabled", self.gates.history_enabled),
            ("gates.anchor_enabled", self.gates.anchor_enabled),
            ("gates.plasticity_enabled", self.gates.plasticity_enabled),
            ("gates.functional_error_enabled", self.gates.functional_error_enabled),
            ("gates.spectral_drift_enabled", self.gates.spectral_drift_enabled),
            ("canary.enabled", self.canary.enabled),
        )
        for name, value in disabled_legacy_controls:
            require_bool(name, value)
            if value:
                raise FDPSCConfigError(
                    f"fsd_v2 first-round runtime requires {name}=false"
                )

        require_bool("fd_psc.enabled", self.enabled)
        for name, value in (
            ("target_modules.predictor_linear", self.target_modules.predictor_linear),
            (
                "target_modules.post_backbone_projection_linear",
                self.target_modules.post_backbone_projection_linear,
            ),
            (
                "target_modules.action_encoder_linear",
                self.target_modules.action_encoder_linear,
            ),
            (
                "target_modules.proprio_encoder_linear",
                self.target_modules.proprio_encoder_linear,
            ),
            (
                "target_modules.exclude_frozen_backbone",
                self.target_modules.exclude_frozen_backbone,
            ),
            (
                "target_modules.require_active_forward_path",
                self.target_modules.require_active_forward_path,
            ),
            (
                "target_modules.fail_on_empty_predictor_targets",
                self.target_modules.fail_on_empty_predictor_targets,
            ),
            (
                "target_modules.fail_on_empty_projection_targets",
                self.target_modules.fail_on_empty_projection_targets,
            ),
            (
                "target_modules.require_projection_targets_if_head_exists",
                self.target_modules.require_projection_targets_if_head_exists,
            ),
        ):
            require_bool(name, value)
        if not self.target_modules.predictor_linear:
            raise FDPSCConfigError(
                "fsd_v2 currently requires target_modules.predictor_linear=true"
            )
        if not self.target_modules.exclude_frozen_backbone:
            raise FDPSCConfigError(
                "fsd_v2 first-round target discovery requires "
                "target_modules.exclude_frozen_backbone=true"
            )
        if not self.target_modules.require_active_forward_path:
            raise FDPSCConfigError(
                "fsd_v2 requires target_modules.require_active_forward_path=true"
            )
        if not self.target_modules.fail_on_empty_predictor_targets:
            raise FDPSCConfigError(
                "fsd_v2 requires target_modules.fail_on_empty_predictor_targets=true"
            )

        positive_int("episodic_lora.rank", self.episodic_lora.rank)
        finite_positive("episodic_lora.alpha", self.episodic_lora.alpha)
        dropout = finite("episodic_lora.dropout", self.episodic_lora.dropout)
        if dropout != 0.0:
            raise FDPSCConfigError("fsd_v2 currently requires episodic_lora.dropout=0")
        require_bool("episodic_lora.pilot_enabled", self.episodic_lora.pilot_enabled)
        if not self.episodic_lora.pilot_enabled:
            raise FDPSCConfigError("fsd_v2 requires episodic_lora.pilot_enabled=true")
        if self.episodic_lora.a_initialization != "kaiming_uniform":
            raise FDPSCConfigError(
                "fsd_v2 requires episodic_lora.a_initialization=kaiming_uniform"
            )
        if self.episodic_lora.b_initialization != "zeros":
            raise FDPSCConfigError(
                "fsd_v2 requires episodic_lora.b_initialization=zeros"
            )

        require_bool("conv_lora.enabled", self.conv_lora.enabled)
        if self.conv_lora.enabled and (
            self.conv_lora.target_scope != "post_backbone_projection_head"
            or self.conv_lora.parameterization != "flattened_kernel"
            or self.conv_lora.groups_mode != "groupwise"
        ):
            raise FDPSCConfigError(
                "fsd_v2 ConvLoRA currently requires "
                "post_backbone_projection_head/flattened_kernel/groupwise semantics"
            )

        persistent_rank = self.slow_lora.persistent_rank
        positive_int("slow_lora.persistent_rank", persistent_rank)
        positive_int("slow_lora.maximum_rank", self.slow_lora.maximum_rank)
        assert persistent_rank is not None
        if persistent_rank > self.slow_lora.maximum_rank:
            raise FDPSCConfigError(
                "slow_lora.persistent_rank must not exceed slow_lora.maximum_rank"
            )

        require_bool("rtrc.enabled", self.rtrc.enabled)
        if not self.rtrc.enabled:
            raise FDPSCConfigError("fsd_v2 requires rtrc.enabled=true")
        beta_minimum = finite_positive(
            "rtrc.budget_fraction_minimum",
            self.rtrc.budget_fraction_minimum,
        )
        beta_initial = finite_positive(
            "rtrc.budget_fraction_initial",
            self.rtrc.budget_fraction_initial,
        )
        beta_maximum = finite_positive(
            "rtrc.budget_fraction_maximum",
            self.rtrc.budget_fraction_maximum,
        )
        if not (beta_minimum <= beta_initial <= beta_maximum <= 1.0):
            raise FDPSCConfigError(
                "rtrc budget fractions must satisfy "
                "0 < minimum <= initial <= maximum <= 1"
            )
        positive_int("rtrc.geometry_windows", self.rtrc.geometry_windows)
        positive_int("rtrc.geometry_maximum_rank", self.rtrc.geometry_maximum_rank)
        geometry_threshold = finite(
            "rtrc.geometry_energy_threshold",
            self.rtrc.geometry_energy_threshold,
        )
        if not 0.0 < geometry_threshold <= 1.0:
            raise FDPSCConfigError(
                "rtrc.geometry_energy_threshold must be in (0, 1]"
            )
        finite_positive("rtrc.minimum_energy", self.rtrc.minimum_energy)
        require_bool("rtrc.use_shared_dual", self.rtrc.use_shared_dual)
        if not self.rtrc.use_shared_dual:
            raise FDPSCConfigError("fsd_v2 requires rtrc.use_shared_dual=true")
        if self.rtrc.layer_weight_mode != "relative_output_energy":
            raise FDPSCConfigError(
                "fsd_v2 requires rtrc.layer_weight_mode=relative_output_energy"
            )
        if self.rtrc.tail_mode != "conservative_isotropic":
            raise FDPSCConfigError(
                "fsd_v2 requires rtrc.tail_mode=conservative_isotropic"
            )
        positive_int("rtrc.bisection_iterations", self.rtrc.bisection_iterations)
        finite_positive(
            "rtrc.bisection_relative_tolerance",
            self.rtrc.bisection_relative_tolerance,
        )
        finite_positive("rtrc.epsilon", self.rtrc.epsilon)

        positive_int(
            "raw_replay.historical_windows",
            self.raw_replay.historical_windows,
        )
        require_bool(
            "raw_replay.store_model_input_obs",
            self.raw_replay.store_model_input_obs,
        )
        if not self.raw_replay.store_model_input_obs:
            raise FDPSCConfigError(
                "fsd_v2 requires raw_replay.store_model_input_obs=true"
            )
        require_bool(
            "raw_replay.store_frozen_visual_latent_cache",
            self.raw_replay.store_frozen_visual_latent_cache,
        )
        if self.raw_replay.store_frozen_visual_latent_cache:
            raise FDPSCConfigError(
                "raw_replay.store_frozen_visual_latent_cache=true is not implemented"
            )
        if self.raw_replay.image_storage_dtype != "uint8_or_source":
            raise FDPSCConfigError(
                "raw_replay.image_storage_dtype must be uint8_or_source"
            )
        if self.raw_replay.auxiliary_dtype != "float32":
            raise FDPSCConfigError("raw_replay.auxiliary_dtype must be float32")
        positive_int(
            "raw_replay.maximum_context_clusters",
            self.raw_replay.maximum_context_clusters,
        )
        positive_int(
            "raw_replay.minimum_windows_per_cluster",
            self.raw_replay.minimum_windows_per_cluster,
        )
        if (
            self.raw_replay.minimum_windows_per_cluster
            > self.raw_replay.historical_windows
        ):
            raise FDPSCConfigError(
                "raw_replay.minimum_windows_per_cluster must not exceed "
                "raw_replay.historical_windows"
            )

        require_bool("adaptive_budget.enabled", self.adaptive_budget.enabled)
        if not self.adaptive_budget.enabled:
            raise FDPSCConfigError("fsd_v2 requires adaptive_budget.enabled=true")
        finite_positive(
            "adaptive_budget.controller_learning_rate",
            self.adaptive_budget.controller_learning_rate,
        )
        for name, value in (
            ("plasticity_loss_target", self.adaptive_budget.plasticity_loss_target),
            (
                "history_regression_target",
                self.adaptive_budget.history_regression_target,
            ),
            ("history_weight", self.adaptive_budget.history_weight),
            ("minimum_wake_gain", self.adaptive_budget.minimum_wake_gain),
        ):
            numeric = finite(f"adaptive_budget.{name}", value)
            if numeric < 0.0:
                raise FDPSCConfigError(
                    f"adaptive_budget.{name} must be non-negative"
                )
        finite_positive("adaptive_budget.error_clip", self.adaptive_budget.error_clip)

        require_bool("deep_sleep.enabled", self.deep_sleep.enabled)
        trigger_error = finite(
            "deep_sleep.trigger_relative_rank_error",
            self.deep_sleep.trigger_relative_rank_error,
        )
        if trigger_error < 0.0:
            raise FDPSCConfigError(
                "deep_sleep.trigger_relative_rank_error must be non-negative"
            )
        positive_int(
            "deep_sleep.trigger_consecutive_commits",
            self.deep_sleep.trigger_consecutive_commits,
        )
        positive_int(
            "deep_sleep.minimum_replay_windows",
            self.deep_sleep.minimum_replay_windows,
        )
        if self.deep_sleep.strategy != "residual_distill":
            raise FDPSCConfigError(
                "deep_sleep.strategy must be residual_distill"
            )
        positive_int("deep_sleep.maximum_steps", self.deep_sleep.maximum_steps)
        finite_positive("deep_sleep.learning_rate", self.deep_sleep.learning_rate)
        weights = {}
        for name, value in (
            ("output_residual_weight", self.deep_sleep.output_residual_weight),
            ("hidden_residual_weight", self.deep_sleep.hidden_residual_weight),
            ("current_task_weight", self.deep_sleep.current_task_weight),
        ):
            numeric = finite(f"deep_sleep.{name}", value)
            if numeric < 0.0:
                raise FDPSCConfigError(
                    f"deep_sleep.{name} must be non-negative"
                )
            weights[name] = numeric
        if weights["output_residual_weight"] <= 0.0:
            raise FDPSCConfigError(
                "deep_sleep.output_residual_weight must be positive"
            )
        positive_int("deep_sleep.residual_rank", self.deep_sleep.residual_rank)
        positive_int("deep_sleep.batch_size", self.deep_sleep.batch_size)
        positive_int(
            "deep_sleep.hidden_layer_maximum",
            self.deep_sleep.hidden_layer_maximum,
        )
        validation_fraction = finite(
            "deep_sleep.validation_fraction",
            self.deep_sleep.validation_fraction,
        )
        if not 0.0 < validation_fraction < 1.0:
            raise FDPSCConfigError(
                "deep_sleep.validation_fraction must be in (0, 1)"
            )
        positive_int(
            "deep_sleep.minimum_validation_windows",
            self.deep_sleep.minimum_validation_windows,
        )
        functional_threshold = finite(
            "deep_sleep.functional_error_threshold",
            self.deep_sleep.functional_error_threshold,
        )
        if functional_threshold < 0.0:
            raise FDPSCConfigError(
                "deep_sleep.functional_error_threshold must be non-negative"
            )
        absolute_tolerance = finite(
            "deep_sleep.functional_error_absolute_tolerance",
            self.deep_sleep.functional_error_absolute_tolerance,
        )
        if absolute_tolerance < 0.0:
            raise FDPSCConfigError(
                "deep_sleep.functional_error_absolute_tolerance must be non-negative"
            )
        finite_positive("deep_sleep.epsilon", self.deep_sleep.epsilon)
        if not isinstance(self.deep_sleep.core_storage, CoreStorageConfig):
            raise FDPSCConfigError(
                "deep_sleep.core_storage must be a typed mapping"
            )
        if self.deep_sleep.core_storage.mode != "dense_delta":
            raise FDPSCConfigError(
                "deep_sleep.core_storage.mode must be dense_delta"
            )

        require_bool("checkpoint.enabled", self.checkpoint.enabled)
        require_bool(
            "checkpoint.keep_commit_journal",
            self.checkpoint.keep_commit_journal,
        )
        require_bool("checkpoint.atomic_write", self.checkpoint.atomic_write)
        if not self.checkpoint.keep_commit_journal:
            raise FDPSCConfigError(
                "fsd_v2 requires checkpoint.keep_commit_journal=true"
            )
        if not self.checkpoint.atomic_write:
            raise FDPSCConfigError("fsd_v2 requires checkpoint.atomic_write=true")
        positive_int(
            "checkpoint.save_every_episodes",
            self.checkpoint.save_every_episodes,
        )
        positive_int(
            "checkpoint.retention_versions",
            self.checkpoint.retention_versions,
        )
        for name, value in (
            ("checkpoint.state_directory", self.checkpoint.state_directory),
            ("checkpoint.latest_pointer_path", self.checkpoint.latest_pointer_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise FDPSCConfigError(f"{name} must be a non-empty path string")
        resume_value = self.checkpoint.resume_path
        if resume_value is not None and (
            not isinstance(resume_value, str) or not resume_value.strip()
        ):
            raise FDPSCConfigError(
                "checkpoint.resume_path must be null or a non-empty path string"
            )
        if resume_value is not None and not self.checkpoint.enabled:
            raise FDPSCConfigError(
                "checkpoint.resume_path requires checkpoint.enabled=true"
            )

        runtime_root = Path.cwd() if runtime_output_dir is None else Path(runtime_output_dir)
        paths = self.resolve_v2_paths(runtime_root)
        state_directory = paths["state_directory"]
        latest_pointer = paths["latest_pointer"]
        resume_path = paths["resume"]
        assert state_directory is not None and latest_pointer is not None
        if resume_path is not None:
            explicit_version = (
                resume_path.parent == state_directory and resume_path.suffix == ".pt"
            )
            if resume_path != latest_pointer and not explicit_version:
                raise FDPSCConfigError(
                    "checkpoint.resume_path must be the configured latest pointer "
                    "or a .pt version directly inside checkpoint.state_directory"
                )
            if require_files and not resume_path.is_file():
                raise FDPSCConfigError(
                    f"FSD V2 resume checkpoint is not readable: {resume_path}"
                )

    def validate(self, runtime_output_dir: Optional[Path] = None, require_files: bool = True) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise FDPSCConfigError("fd_psc.seed must be a non-negative integer")
        if not self.enabled:
            return

        if self.run_mode not in {
            "fd_psc",
            "fsd_v2",
            "episodic_reset",
            "accumulate",
            "plain_svd",
        }:
            raise FDPSCConfigError(
                "fd_psc.run_mode must be fd_psc, fsd_v2, episodic_reset, "
                "accumulate, or plain_svd"
            )
        if self.run_mode == "fsd_v2":
            self._validate_fsd_v2(
                runtime_output_dir=runtime_output_dir,
                require_files=require_files,
            )
            return
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


def minimal_fsd_v2_config(**overrides: Any) -> FDPSCConfig:
    """Return a dependency-light FSD V2 config with no external data inputs."""

    raw: Dict[str, Any] = {
        "enabled": True,
        "run_mode": "fsd_v2",
        "slow_lora": {"persistent_rank": 8},
        "gradient_geometry": {"enabled": False},
        "slice": {"enabled": False},
        "sdc": {"enabled": False},
        "spectral_surgery": {"enabled": False},
        "activation_subspace": {"enabled": False},
        "merge": {"soft_ness_enabled": False},
        "repair": {"enabled": False},
        "exception": {
            "enabled": False,
            "maximum_adapters": 0,
            "local_replay_windows": 0,
        },
        "gates": {
            "allow_unsafe_ablation": False,
            "current_gain_enabled": False,
            "history_enabled": False,
            "anchor_enabled": False,
            "plasticity_enabled": False,
            "functional_error_enabled": False,
            "spectral_drift_enabled": False,
        },
        "canary": {"enabled": False},
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
    "AdaptiveBudgetConfig",
    "CoreStorageConfig",
    "DeepSleepConfig",
    "FDPSCConfig",
    "FDPSCConfigError",
    "RTRCConfig",
    "RawReplayConfig",
    "minimal_fsd_v2_config",
    "minimal_test_config",
]
