"""Runtime integration for Frozen-Dynamics Plasticity-Safe Consolidation.

The implementation in this module is deliberately orchestration-heavy.  The
mathematics, data guards, state machine, replay policy, gates, and checkpoint
protocol live in small independently-tested modules; :class:`FDPSCSystem`
connects them to AdaJEPA without changing its JEPA objective or replan timing.

Two invariants are treated as fatal throughout:

* every pre-injection parameter and persistent buffer (theta_0) remains
  bitwise identical for the lifetime of the process; and
* commit-query data is obtained only after exactly one final proposal has
  been selected, through the registry's single-use proposal-bound token.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .activation_subspace import (
    ActivationSubspace,
    ActivationSubspaceBank,
    apply_soft_ness,
    conv2d_group_activation_matrices,
)
from .checkpoint import CheckpointValidationError, SidecarCheckpointManager, sha256_file
from .canary import (
    CanaryManifest,
    CanaryPhase,
    CanaryRunner,
    CanaryScheduler,
    CanaryStatus,
    CanaryTriggerDecision,
)
from .commit_gates import CommitGateEvaluator, CommitGateInputs
from .config import FDPSCConfig
from .diagnostics import Diagnostics, assert_finite_tree
from .encoder_adapters import FrozenVisualLatent, LATENT_SCHEMA_VERSION
from .exception_router import ExceptionRouter, RouteDecision
from .external_data import DataIdentity, ExternalDataRegistry, ExternalRecord, canonical_json_hash
from .gradient_geometry import (
    GradientGeometryTracker,
    c_pcgrad,
    dual_constraint_projection,
    global_weighted_cosine,
    gradient_cosine,
)
from .gradient_hooks import EffectiveWeightGradientHooks
from .injector import InjectionResult, ManifestEntry, inject_fd_psc_adapters
from .low_rank_merge import (
    LowRankFactors,
    concatenate_factors,
    factor_frobenius_norm_sq,
    functional_error,
    select_rank,
    truncate_factors,
)
from .metrics import StructuredMetrics
from .repair import RepairEngine, ScreeningResult
from .replay_memory import ClusterBalancedReplay, ReplayWindow
from .slice_initializer import initialize_slice, simulate_factor_first_step
from .spectral_control import (
    BaseSpectrum,
    SDCEventTracker,
    compute_base_spectrum,
    effective_gradient_proxy,
    sdc_correct_gradient,
    spectral_drift,
    spectral_surgery,
)
from .state_machine import FDPSCState, FDPSCStateMachine, FinalProposal, ProposalType
from .transaction import Participant, RNGSnapshot, StateTransaction


class FDPSCIntegrationError(RuntimeError):
    """Raised when an integration-level FD-PSC invariant is violated."""


class _RepairGeometryInfeasible(RuntimeError):
    """Internal signal: reject this repair trajectory, never silently use raw G."""


class _PeriodicCanaryFailure(FDPSCIntegrationError):
    """A scheduled post-commit canary rejected a cumulative commit period."""

    def __init__(
        self,
        message: str,
        *,
        report: Mapping[str, Any],
        attempted_commit_id: str,
        attempted_commit_sequence: int,
        reverted_commit_ids: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.report = copy.deepcopy(dict(report))
        self.attempted_commit_id = str(attempted_commit_id)
        self.attempted_commit_sequence = int(attempted_commit_sequence)
        self.reverted_commit_ids = tuple(str(value) for value in reverted_commit_ids)


@dataclass
class _BaseTensor:
    name: str
    tensor: Tensor
    value: Tensor
    is_parameter: bool


@dataclass
class _SupportSegment:
    identity: DataIdentity
    obs: Mapping[str, Tensor]
    actions: Tensor
    iteration: int
    episode_id: str
    context_identifier: str
    preprocess_hash: str
    latent_adapter_schema: str
    source_record_ids: Tuple[str, ...] = ()
    source_iterations: Tuple[int, ...] = ()


@dataclass
class _Candidate:
    proposal_type: ProposalType
    factors_by_layer: Dict[str, LowRankFactors]
    task_factors_by_layer: Dict[str, LowRankFactors]
    functional_error_by_layer: Dict[str, float]
    selected_rank_by_layer: Dict[str, int]
    alpha_shared: float
    alpha_safe: float
    spectral_variant: str
    calibration_loss: float = float("inf")
    calibration_gain: float = -float("inf")
    history_loss: Optional[float] = None
    worst_context_regression: Optional[float] = None
    anchor_loss: Optional[float] = None
    anchor_regression: Optional[float] = None
    plasticity_gain: Optional[float] = None
    maximum_drift_increase: Optional[float] = None
    screening_reason: str = "not_screened"
    repair_step: Optional[int] = None
    spectral_calibration_safe: Optional[bool] = None
    # Immutable, untruncated factor-space merge used for rank selection.
    # Keeping this separate from ``factors_by_layer`` is essential: the
    # latter is the provisional compressed persistent state whose own
    # calibration activations define H_l.
    rank_reference_by_layer: Dict[str, LowRankFactors] = field(
        default_factory=dict,
        repr=False,
    )

    def summary(self) -> Dict[str, Any]:
        def finite_or_none(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        return {
            "proposal_type": self.proposal_type.value,
            "alpha_shared": float(self.alpha_shared),
            "alpha_safe": float(self.alpha_safe),
            "spectral_variant": self.spectral_variant,
            # ``inf`` is useful internally as the ordering sentinel for an
            # unevaluated candidate, but it is not a valid JSON metric.  A
            # null plus the screening reason is the auditable wire format.
            "calibration_loss": finite_or_none(self.calibration_loss),
            "calibration_gain": finite_or_none(self.calibration_gain),
            "calibration_evaluated": (
                math.isfinite(float(self.calibration_loss))
                and math.isfinite(float(self.calibration_gain))
            ),
            "history_loss": finite_or_none(self.history_loss),
            "worst_context_regression": finite_or_none(
                self.worst_context_regression
            ),
            "anchor_loss": finite_or_none(self.anchor_loss),
            "anchor_regression": finite_or_none(self.anchor_regression),
            "plasticity_gain": finite_or_none(self.plasticity_gain),
            "maximum_drift_increase": finite_or_none(
                self.maximum_drift_increase
            ),
            "screening_reason": self.screening_reason,
            "repair_step": self.repair_step,
            "spectral_calibration_safe": self.spectral_calibration_safe,
            "selected_rank_by_layer": dict(sorted(self.selected_rank_by_layer.items())),
            "functional_error_by_layer": {
                key: finite_or_none(value)
                for key, value in sorted(self.functional_error_by_layer.items())
            },
        }


@dataclass(frozen=True)
class _MergeSimilaritySignals:
    """Frozen, pre-query signals used only to prune the coefficient grid."""

    gradient: Optional[float] = None
    context: Optional[float] = None
    residual: Optional[float] = None


@dataclass(frozen=True)
class _CoefficientGridDecision:
    shared_coefficients: Tuple[float, ...]
    reason: str
    signal_decisions: Mapping[str, str]


class _AdapterParticipant:
    """StateTransaction participant for dynamic-rank adapter tensors."""

    def __init__(self, injection: InjectionResult) -> None:
        self.injection = injection

    def state_dict(self) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(adapter.adapter_state_dict())
            for key, adapter in sorted(self.injection.adapters.items())
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.injection.adapters):
            raise FDPSCIntegrationError("adapter transaction registry changed")
        for key, adapter in sorted(self.injection.adapters.items()):
            adapter.load_adapter_state_dict(copy.deepcopy(state[key]))


class _CounterParticipant:
    def __init__(self, owner: "FDPSCSystem") -> None:
        self.owner = owner

    def state_dict(self) -> Dict[str, int]:
        return {
            "episode_sequence": self.owner._episode_sequence,
            "commit_sequence": self.owner._commit_sequence,
            "successful_slow_commit_count": self.owner._successful_slow_commit_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.owner._episode_sequence = int(state["episode_sequence"])
        self.owner._commit_sequence = int(state["commit_sequence"])
        self.owner._successful_slow_commit_count = int(
            state["successful_slow_commit_count"]
        )


class _GradientReferenceParticipant:
    def __init__(self, owner: "FDPSCSystem") -> None:
        self.owner = owner

    def state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        return {
            "history": {
                key: value.detach().clone()
                for key, value in sorted(self.owner._history_gradients.items())
            },
            "anchor": {
                key: value.detach().clone()
                for key, value in sorted(self.owner._anchor_gradients.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.owner._history_gradients = {
            str(key): value.detach().clone()
            for key, value in state.get("history", {}).items()
        }
        self.owner._anchor_gradients = {
            str(key): value.detach().clone()
            for key, value in state.get("anchor", {}).items()
        }


class _CanaryPeriodParticipant:
    """Make canary-period bookkeeping part of the atomic commit boundary."""

    def __init__(self, owner: "FDPSCSystem") -> None:
        self.owner = owner

    def state_dict(self) -> Dict[str, Any]:
        return self.owner._canary_period_state()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.owner._load_canary_period_state(state, validate_runtime=False)


def _stable_seed(seed: int, *parts: object) -> int:
    text = "\0".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    result = torch.Generator(device=device)
    result.manual_seed(int(seed))
    return result


def _tensor_bytes(tensor: Tensor) -> bytes:
    # A CUDA convolution may retain an elementwise-contiguous layout whose
    # trailing singleton dimensions have strides greater than one (for
    # example ``(2, 1, 2, 2)`` for a ``[2, 2, 1, 1]`` kernel).  PyTorch's
    # dtype-view requires the final stride to be exactly one even though that
    # layout is otherwise contiguous.  Flatten first so byte reinterpretation
    # is valid on both CPU and CUDA without weakening the bitwise invariant.
    return (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


def _tensor_byte_view(tensor: Tensor) -> Tensor:
    return tensor.detach().contiguous().reshape(-1).contiguous().view(torch.uint8)


def _hash_tensor_tree(value: Any) -> str:
    def canonical(item: Any) -> Any:
        if isinstance(item, Tensor):
            if not torch.isfinite(item).all():
                raise FDPSCIntegrationError("support tensors must be finite before identity audit")
            return item.detach().cpu().tolist()
        if isinstance(item, Mapping):
            return {str(key): canonical(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [canonical(child) for child in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)

    # The same canonical JSON algorithm is used by inline external manifests,
    # so identical raw support cannot evade leakage detection merely because it
    # originated as a Tensor rather than a JSON list.
    return canonical_json_hash(canonical(value))


def _tensor_tree_nbytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(item) for item in value)
    if hasattr(value, "__dict__"):
        return _tensor_tree_nbytes(vars(value))
    return 0


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class FDPSCSystem:
    """One persistent FD-PSC memory attached to an AdaJEPA world model."""

    STATE_SCHEMA_VERSION = 1
    _ACTIVATION_ROW_LIMIT = 4096

    def __init__(
        self,
        *,
        wm: nn.Module,
        config: Any,
        runtime_output_dir: Path,
        canary_evaluator: Optional[Any] = None,
        runtime_preprocess_hash: Optional[str] = None,
    ) -> None:
        self.wm = wm
        self.config = FDPSCConfig.from_mapping(config)
        if not self.config.enabled:
            raise FDPSCIntegrationError("FDPSCSystem cannot be constructed for a disabled config")
        self.runtime_output_dir = Path(runtime_output_dir).expanduser().resolve()
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)
        self._full_protocol = self.config.run_mode == "fd_psc"
        self.config.validate(
            self.runtime_output_dir,
            require_files=self._full_protocol,
        )
        self.paths = self.config.resolve_paths(self.runtime_output_dir)

        self.device = next(wm.parameters()).device
        self._base_tensors = self._capture_base_tensors()
        computed_base_hash = self._compute_base_state_hash()
        supplied_hash = str(getattr(wm, "_base_checkpoint_hash", "")).lower()
        self.base_checkpoint_hash = supplied_hash or computed_base_hash
        if len(self.base_checkpoint_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.base_checkpoint_hash
        ):
            raise FDPSCIntegrationError("base checkpoint hash is not a SHA-256 digest")
        supplied_preprocess_hash = (
            runtime_preprocess_hash
            if runtime_preprocess_hash is not None
            else getattr(wm, "_fd_psc_preprocess_hash", None)
        )
        if supplied_preprocess_hash is None:
            if self._full_protocol:
                raise FDPSCIntegrationError(
                    "full FD-PSC requires an explicit runtime preprocessing identity"
                )
            self.preprocess_hash: Optional[str] = None
        else:
            self.preprocess_hash = str(supplied_preprocess_hash).strip().lower()
            if len(self.preprocess_hash) != 64 or any(
                char not in "0123456789abcdef" for char in self.preprocess_hash
            ):
                raise FDPSCIntegrationError(
                    "runtime preprocessing identity is not a SHA-256 digest"
                )

        self.injection = inject_fd_psc_adapters(
            wm,
            self.config,
            schema_sample=self._schema_only_dry_run,
        )
        if not self.injection.adapters:
            raise FDPSCIntegrationError("enabled FD-PSC injected no logical adapters")
        self.target_manifest = self.injection.manifest
        self._entry_by_id = self.target_manifest.by_logical_id()
        self._adapter_participant = _AdapterParticipant(self.injection)
        self.assert_base_frozen()

        self.external: Optional[ExternalDataRegistry] = None
        manifest_path = self.paths.get("external_manifest")
        if manifest_path is not None:
            self.external = ExternalDataRegistry(
                manifest_path,
                verify_checksums=self.config.external_eval_data.verify_checksums,
                expected_base_checkpoint_hash=self.base_checkpoint_hash,
                expected_preprocess_hash=self.preprocess_hash,
                require_all_splits=False,
            )
            if self.external.latent_adapter_schema != LATENT_SCHEMA_VERSION:
                raise FDPSCIntegrationError(
                    "external latent_adapter_schema does not match the runtime frozen-latent protocol"
                )
            self._verify_configured_split_paths()
        elif self._full_protocol:
            raise FDPSCIntegrationError("full FD-PSC requires external_eval_data.manifest_path")

        self.canary_scheduler = CanaryScheduler.from_config(self.config.canary)
        self.canary_runner: Optional[CanaryRunner] = None
        if self.config.canary.enabled:
            canary_manifest_path = self.paths.get("canary_manifest")
            if canary_manifest_path is None:
                raise FDPSCIntegrationError("enabled Gate-7 canary requires a fixed manifest")
            canary_manifest = CanaryManifest.load(
                canary_manifest_path,
                expected_base_checkpoint_hash=self.base_checkpoint_hash,
                expected_preprocess_hash=self.preprocess_hash,
            )
            self.canary_runner = CanaryRunner(
                canary_manifest,
                rollout_count=self.config.canary.rollout_count,
                evaluator=canary_evaluator,
                unavailable_policy=self.config.canary.unavailable_policy,
            )

        self.state_machine = FDPSCStateMachine()
        self.metrics = StructuredMetrics()
        self.diagnostics = Diagnostics()
        self.replay = ClusterBalancedReplay(
            self.config.replay.historical_windows,
            maximum_context_clusters=self.config.replay.maximum_context_clusters,
            new_cluster_similarity_threshold=self.config.replay.new_cluster_similarity_threshold,
            minimum_windows_per_cluster=self.config.replay.minimum_windows_per_cluster,
            seed=_stable_seed(self.config.seed, "historical-replay"),
        )
        self.subspaces = ActivationSubspaceBank()
        self.exception_router = ExceptionRouter(
            maximum_adapters=(
                self.config.exception.maximum_adapters if self.config.exception.enabled else 0
            ),
            minimum_route_similarity=self.config.exception.minimum_route_similarity,
            local_replay_windows=self.config.exception.local_replay_windows,
            seed=_stable_seed(self.config.seed, "exception-router"),
            no_match_behavior=self.config.exception.no_match_behavior,
        )
        self.gates = CommitGateEvaluator(
            self.config.gates,
            functional_error_threshold=self.config.slow_lora.functional_error_threshold,
            minimum_exception_commit_fast_gain=self.config.exception.minimum_commit_fast_gain,
        )
        self.repair_engine = (
            RepairEngine(
                maximum_steps=self.config.repair.maximum_steps,
                candidate_steps=self.config.repair.candidate_steps,
                windows_per_batch=self.config.repair.windows_per_batch,
                current_weight=self.config.repair.current_weight,
                replay_weight=self.config.repair.replay_weight,
                proximal_enabled=self.config.repair.proximal_enabled,
                proximal_weight=self.config.repair.proximal_weight,
                pcgrad_enabled=self.config.repair.pcgrad_enabled,
                seed=_stable_seed(self.config.seed, "repair"),
                sampling=self.config.replay.repair_sampling,
            )
            if self.config.repair.enabled
            else None
        )
        self.geometry = GradientGeometryTracker(
            self.config.gradient_geometry.ema_beta,
            self.config.gradient_geometry.conflict_threshold,
            self.config.gradient_geometry.consecutive_conflicts,
        )
        self._sdc_trackers = {
            logical_id: SDCEventTracker(
                check_every_replans=self.config.sdc.check_every_replans,
                drift_threshold=self.config.sdc.drift_threshold,
                drift_consecutive_checks=self.config.sdc.drift_consecutive_checks,
                drift_increase_tolerance=self.config.sdc.drift_increase_tolerance,
                anchor_regression_trigger=self.config.sdc.anchor_regression_trigger,
            )
            for logical_id in self.injection.adapters
        }
        self._base_spectra = self._build_base_spectra()
        self._history_gradients: Dict[str, Tensor] = {}
        self._anchor_gradients: Dict[str, Tensor] = {}
        self._latest_online_gradients: Dict[str, Tensor] = {}
        self._latest_corrected_gradients: Dict[str, Tensor] = {}
        self._latest_gradient_cosines: Dict[str, Dict[str, Optional[float]]] = {}
        self._sdc_active: Dict[str, bool] = {
            logical_id: (self.config.sdc.enabled and not self.config.sdc.event_triggered)
            for logical_id in self.injection.adapters
        }
        # Episode-local SDC safety baseline.  Records are fixed on the first
        # scheduled check and the P_before loss is evaluated once, then reused
        # for every later check in the same episode.  Checkpoints are legal only
        # between episodes, so this transient state is deliberately not part of
        # the persistent sidecar schema.
        self._sdc_anchor_records: Tuple[ExternalRecord, ...] = ()
        self._sdc_anchor_before_loss: Optional[float] = None
        self._sdc_anchor_current_loss: Optional[float] = None
        self._sdc_anchor_regression_value: Optional[float] = None

        self._episode_sequence = 0
        self._commit_sequence = 0
        # Ephemeral deduplication only: a persistent commit checkpoint already
        # contains the projected between-episode state, so terminal cleanup
        # must not immediately write a duplicate cadence snapshot.
        self._last_checkpoint_episode_sequence = 0
        # Gate 2 is cold-start N/A only until the first *successful slow*
        # commit.  Replay contents cannot serve as this evidence: a legal
        # capacity-zero replay (or a corrupted/empty restored replay) must not
        # silently put later commits back into cold-start mode.
        self._successful_slow_commit_count = 0
        self._active_episode_id: Optional[str] = None
        self._active_context: Optional[str] = None
        self._episode_metadata: Dict[str, Any] = {}
        self._context_embedding: Optional[Tensor] = None
        self._route: Optional[RouteDecision] = None
        self._support_segments: List[_SupportSegment] = []
        self._support_context_descriptors: List[Tensor] = []
        self._support_hashes: set[str] = set()
        self._support_transition_cursor = 0
        self._episode_start_adapter_states: Optional[Dict[str, Any]] = None
        self._episode_start_repair_state: Optional[Dict[str, Any]] = None
        self._plasticity_probe_rng: Optional[RNGSnapshot] = None
        self._online_hooks: Optional[EffectiveWeightGradientHooks] = None
        self._online_mode_depth = 0
        self._online_update_started_at: Optional[float] = None
        self._sleep_started_at: Optional[float] = None
        self._replan_index = 0
        self._online_loss = 0.0
        self._online_step_count = 0
        self._episode_early_losses: List[float] = []
        self._episode_start_persistent_commit_count = 0
        self._jepa_loss_threshold: Optional[float] = None
        self._jepa_loss_threshold_reason = "threshold_not_provided"
        self._loss_threshold_reached = False
        self._centered_reason: Optional[str] = None
        self._calibration_candidate_count = 0

        # A periodic Gate-7 compares the whole not-yet-certified commit period
        # with this baseline.  The baseline is a detached algorithm-memory
        # snapshot, not merely the immediately preceding slow factors.
        self._canary_known_good: Optional[Dict[str, Any]] = None
        self._canary_pending_commit_ids: List[str] = []
        self._canary_last_rollback: Optional[Dict[str, Any]] = None

        self.checkpoints: Optional[SidecarCheckpointManager] = None
        if self.config.checkpoint.enabled:
            state_directory = self.paths["state_directory"]
            latest_pointer = self.paths["latest_pointer"]
            assert state_directory is not None and latest_pointer is not None
            self.checkpoints = SidecarCheckpointManager(
                state_directory=state_directory,
                latest_pointer_path=latest_pointer,
                base_checkpoint_hash=self.base_checkpoint_hash,
                manifest_hash=self.target_manifest.hash,
                schema_version=self.STATE_SCHEMA_VERSION,
                retention_versions=self.config.checkpoint.retention_versions,
                state_validator=self._validate_checkpoint_state,
            )
            if self.paths.get("resume") is not None:
                self._resume(self.paths["resume"])

        if self.canary_runner is not None and self._canary_known_good is None:
            self._promote_canary_known_good(
                commit_id=None,
                commit_sequence=self._commit_sequence,
                persistent_commit_count=self.state_machine.persistent_commit_count,
            )

        self._write_runtime_manifest()
        self.metrics.record(
            "adapter_parameter_count",
            sum(
                int(entry.actual_rank) * (int(entry.in_features) + int(entry.out_features))
                for entry in self.target_manifest.entries
                if entry.injected
            ),
            tags={"scope": "initial_episode_capacity"},
        )
        self.assert_base_frozen()

    # ------------------------------------------------------------------
    # Construction and immutable-base verification
    # ------------------------------------------------------------------
    def _capture_base_tensors(self) -> List[_BaseTensor]:
        result: List[_BaseTensor] = []
        seen: set[int] = set()
        for name, parameter in self.wm.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            result.append(_BaseTensor(name, parameter, parameter.detach().clone(), True))
        for module_path, module in self.wm.named_modules():
            for name, buffer in module._buffers.items():
                if buffer is None or name in module._non_persistent_buffers_set or id(buffer) in seen:
                    continue
                seen.add(id(buffer))
                qualified = f"{module_path}.{name}" if module_path else name
                result.append(_BaseTensor(qualified, buffer, buffer.detach().clone(), False))
        return result

    def _compute_base_state_hash(self) -> str:
        digest = hashlib.sha256()
        for item in sorted(self._base_tensors, key=lambda value: value.name):
            digest.update(item.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(item.value.dtype).encode("ascii"))
            digest.update(str(tuple(item.value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(item.value))
        return digest.hexdigest()

    def assert_base_frozen(self) -> None:
        changed: List[str] = []
        trainable: List[str] = []
        for item in self._base_tensors:
            current = item.tensor.detach()
            if current.shape != item.value.shape or current.dtype != item.value.dtype:
                changed.append(item.name)
            elif not torch.equal(
                _tensor_byte_view(current),
                _tensor_byte_view(item.value.to(device=current.device)),
            ):
                changed.append(item.name)
            if item.is_parameter and bool(getattr(item.tensor, "requires_grad", False)):
                trainable.append(item.name)
        if changed or trainable:
            raise FDPSCIntegrationError(
                "theta_0 invariant failed; "
                f"changed={changed[:8]}, requires_grad={trainable[:8]}"
            )

    def _schema_only_dry_run(self, root: nn.Module) -> None:
        """Exercise predictor/head schemas without observations or environment I/O."""

        predictor = getattr(root, "predictor", None)
        if predictor is not None:
            reference = next(predictor.parameters(), None)
            positional = getattr(predictor, "pos_embedding", None)
            if isinstance(positional, Tensor) and positional.ndim == 3:
                tokens, dimension = int(positional.shape[1]), int(positional.shape[2])
            else:
                first = next((m for m in predictor.modules() if isinstance(m, nn.Linear)), None)
                if first is None:
                    raise FDPSCIntegrationError("predictor schema has no Linear input")
                tokens, dimension = 1, int(first.in_features)
            device = reference.device if reference is not None else self.device
            dtype = reference.dtype if reference is not None else torch.float32
            predictor(torch.zeros(1, tokens, dimension, device=device, dtype=dtype))

        encoder = getattr(root, "encoder", None)
        adapter = self.injection.encoder_adapter if hasattr(self, "injection") else None
        paths: Sequence[str]
        if adapter is not None:
            paths = adapter.projection_module_paths
        elif encoder is not None and hasattr(encoder, "projector"):
            paths = ("projector",)
        elif encoder is not None and hasattr(encoder, "projection"):
            paths = ("projection",)
        elif encoder is not None and hasattr(encoder, "to_out"):
            paths = ("to_out",)
        else:
            paths = ()
        for path in paths:
            head = encoder.get_submodule(path)
            first_conv = next((m for m in head.modules() if isinstance(m, nn.Conv2d)), None)
            first_linear = next((m for m in head.modules() if isinstance(m, nn.Linear)), None)
            reference = next(head.parameters(), None)
            device = reference.device if reference is not None else self.device
            dtype = reference.dtype if reference is not None else torch.float32
            if first_conv is not None:
                # 64x64 is schema-only and remains valid for the official
                # channel/global projector stride stacks.
                head(
                    torch.zeros(
                        1, first_conv.in_channels, 64, 64, device=device, dtype=dtype
                    )
                )
            elif first_linear is not None:
                head(torch.zeros(2, first_linear.in_features, device=device, dtype=dtype))

        for name, enabled in (
            ("action_encoder", self.config.target_modules.action_encoder_linear),
            ("proprio_encoder", self.config.target_modules.proprio_encoder_linear),
        ):
            if not enabled:
                continue
            module = getattr(root, name, None)
            first = next((m for m in module.modules() if isinstance(m, nn.Linear)), None)
            if first is None:
                raise FDPSCIntegrationError(f"enabled {name} has no Linear schema")
            reference = next(module.parameters())
            module(torch.zeros(1, 2, first.in_features, device=reference.device, dtype=reference.dtype))

    def _base_matrix(self, entry: ManifestEntry) -> Tensor:
        physical = self.injection.physical_modules[entry.module_path]
        base_layer = getattr(physical, "base_layer", None)
        if isinstance(base_layer, nn.Linear):
            return base_layer.weight.detach()
        if isinstance(base_layer, nn.Conv2d):
            group = int(entry.logical_group or 0)
            output_per_group = base_layer.out_channels // base_layer.groups
            start = group * output_per_group
            return base_layer.weight.detach()[start : start + output_per_group].reshape(
                output_per_group, -1
            )
        raise FDPSCIntegrationError(f"cannot resolve base matrix for {entry.logical_layer_id}")

    def _build_base_spectra(self) -> Dict[str, BaseSpectrum]:
        if not (self.config.sdc.enabled or self.config.gates.spectral_drift_enabled):
            return {}
        result = {}
        for logical_id, entry in sorted(self._entry_by_id.items()):
            if not entry.injected:
                continue
            result[logical_id] = compute_base_spectrum(
                self._base_matrix(entry),
                energy_threshold=self.config.sdc.base_energy_threshold,
            )
        return result

    # ------------------------------------------------------------------
    # Fixed external data and runtime identity
    # ------------------------------------------------------------------
    def _verify_configured_split_paths(self) -> None:
        assert self.external is not None
        raw = self.external._manifest_raw
        split_specs = raw["splits"]
        configured = {
            "calibration": self.paths.get("calibration"),
            "commit_query": self.paths.get("commit_query"),
            "plasticity_support": self.paths.get("plasticity_support"),
            "plasticity_query": self.paths.get("plasticity_query"),
            "report_test": self.paths.get("report_test"),
            "anchor": self.paths.get("anchor_data"),
        }
        for split_name, configured_path in configured.items():
            if configured_path is None:
                continue
            spec = split_specs.get(split_name, {})
            raw_path = spec.get("path") if isinstance(spec, Mapping) else None
            if raw_path is None:
                raise FDPSCIntegrationError(
                    f"configured {split_name}_path cannot refer to an inline manifest split"
                )
            actual = Path(str(raw_path)).expanduser()
            actual = (
                actual if actual.is_absolute() else self.external.manifest_path.parent / actual
            ).resolve()
            if actual != configured_path:
                raise FDPSCIntegrationError(
                    f"configured path for {split_name} differs from fixed manifest: "
                    f"{configured_path} != {actual}"
                )
        anchor_manifest = self.paths.get("anchor_manifest")
        if anchor_manifest is not None and anchor_manifest != self.external.manifest_path:
            if sha256_file(anchor_manifest) != sha256_file(self.external.manifest_path):
                raise FDPSCIntegrationError(
                    "anchor manifest differs from the external split manifest"
                )

    def _write_runtime_manifest(self) -> None:
        value = {
            "schema_version": 1,
            "config_identity": self.config.identity_hash(),
            "persistence_config_identity": self.config.persistence_identity_hash(),
            "run_mode": self.config.run_mode,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "external_manifest_hash": (
                self.external.manifest_hash if self.external is not None else None
            ),
            "canary_manifest_hash": (
                self.canary_runner.manifest.manifest_hash
                if self.canary_runner is not None
                else None
            ),
            "target_manifest_hash": self.target_manifest.hash,
            "target_manifest": self.target_manifest.to_dict(),
            "latent_adapter_schema": LATENT_SCHEMA_VERSION,
        }
        _atomic_json(self.runtime_output_dir / "fd_psc_runtime_manifest.json", value)

    @property
    def next_episode_id(self) -> str:
        return f"episode-{self._episode_sequence:08d}"

    def resolve_context_identifier(self, metadata: Mapping[str, Any]) -> str:
        key = self.config.external_eval_data.context_key
        explicit_values = {
            str(metadata[name])
            for name in {key, "context_identifier"}
            if metadata.get(name) is not None
        }
        if any(not value for value in explicit_values):
            raise FDPSCIntegrationError("episode context metadata must be non-empty")
        if len(explicit_values) > 1:
            raise FDPSCIntegrationError(
                f"episode context metadata fields disagree: {sorted(explicit_values)}"
            )
        if explicit_values:
            return next(iter(explicit_values))
        if self.external is None:
            return "default"
        episode_contexts = self.external.episode_contexts
        lookups: List[str] = []
        if metadata.get("sample_idx") is not None:
            sample = str(metadata["sample_idx"])
            lookups.extend((sample, f"sample:{sample}"))
        if metadata.get("seed") is not None:
            lookups.append(f"seed:{metadata['seed']}")
        mapped = {
            episode_contexts[lookup]
            for lookup in lookups
            if lookup in episode_contexts
        }
        if len(mapped) > 1:
            raise FDPSCIntegrationError(
                f"evaluation manifest has conflicting episode context mappings: {sorted(mapped)}"
            )
        if mapped:
            return next(iter(mapped))
        policy = self.config.external_eval_data.missing_context_policy
        raise FDPSCIntegrationError(
            f"episode context is missing; policy={policy!r}. "
            f"Provide metadata[{key!r}] or manifest.episode_contexts."
        )

    def materialize_external_payload(self, record: Any) -> Mapping[str, Any]:
        if isinstance(record, Mapping):
            return copy.deepcopy(dict(record))
        if not isinstance(record, ExternalRecord):
            payload = getattr(record, "payload", None)
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
            raise TypeError("external record is neither a mapping nor ExternalRecord")
        if record.payload is not None:
            if not isinstance(record.payload, Mapping):
                raise FDPSCIntegrationError("external payload must be an object")
            return copy.deepcopy(dict(record.payload))
        if record.payload_path is None or self.external is None:
            raise FDPSCIntegrationError(f"external record {record.record_id} has no payload")
        path = Path(record.payload_path).expanduser()
        path = (path if path.is_absolute() else self.external.manifest_path.parent / path).resolve()
        if not path.is_file():
            raise FDPSCIntegrationError(f"external payload is unreadable: {path}")
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            actual = canonical_json_hash(value)
        elif path.suffix.lower() in {".pt", ".pth"}:
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(path, map_location="cpu")
            actual = sha256_file(path)
        else:
            raise FDPSCIntegrationError(
                f"unsupported external payload format for {record.record_id}: {path.suffix}"
            )
        if actual != record.identity.content_hash:
            raise FDPSCIntegrationError(
                f"payload content hash mismatch for {record.record_id}: {actual}"
            )
        if not isinstance(value, Mapping):
            raise FDPSCIntegrationError("external payload file must contain a mapping")
        return value

    # ------------------------------------------------------------------
    # Episode lifecycle and incremental support audit
    # ------------------------------------------------------------------
    def _episode_generator(self, logical_id: str, phase: str = "pilot") -> torch.Generator:
        adapter = self.injection.adapters[logical_id]
        return _generator(
            adapter._reference().device,
            _stable_seed(self.config.seed, self._active_episode_id, logical_id, phase),
        )

    def _begin_zero_pilot(self) -> None:
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            adapter.begin_episode(
                generator=self._episode_generator(logical_id),
                clear_exception=True,
            )

    @contextlib.contextmanager
    def _all_adapters_disabled(self) -> Iterator[None]:
        enabled = {
            key: adapter.adapters_enabled
            for key, adapter in self.injection.adapters.items()
        }
        try:
            for adapter in self.injection.adapters.values():
                adapter.disable_all_adapters()
            yield
        finally:
            for key, adapter in self.injection.adapters.items():
                adapter.adapters_enabled = enabled[key]

    def _context_descriptor(self, initial_obs: Any, context: str) -> Optional[Tensor]:
        if initial_obs is not None:
            try:
                prepared = initial_obs
                if isinstance(initial_obs, Mapping):
                    prepared = {
                        key: value.to(self.device) if isinstance(value, Tensor) else value
                        for key, value in initial_obs.items()
                    }
                with self._all_adapters_disabled():
                    latent = self.wm.extract_frozen_visual_latent(prepared)
                vector = (
                    latent.tensor.detach()
                    .to(dtype=torch.float32)
                    .reshape(-1, latent.tensor.shape[-1])
                    .mean(0)
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                return None
        else:
            # Used only by headless/unit integrations. Production planning
            # always supplies initial_obs, so routed descriptors are theta_0
            # visual features there.
            raw = hashlib.sha256(str(context).encode("utf-8")).digest()
            vector = torch.tensor(list(raw), dtype=torch.float32)
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector).all() or float(norm) <= 1.0e-12:
            return None
        return (vector / norm).detach().cpu()

    def _current_context_prototype(self) -> Optional[Tensor]:
        values = self._support_context_descriptors
        if not values:
            return self._context_embedding
        dimensions = {int(value.numel()) for value in values}
        if len(dimensions) != 1:
            raise FDPSCIntegrationError("accepted support context descriptors changed schema")
        mean = torch.stack([value.detach().cpu().flatten() for value in values]).mean(0)
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(mean).all() or float(norm) <= 1.0e-12:
            return self._context_embedding
        return (mean / norm).detach().cpu()

    def _apply_exception_state(self, adapter_id: Optional[str]) -> None:
        for adapter in self.injection.adapters.values():
            adapter.clear_active_exception()
        if adapter_id is None:
            return
        record = self.exception_router.get(adapter_id)
        raw = record.adapter_state
        if not isinstance(raw, Mapping) or int(raw.get("schema_version", -1)) != 1:
            raise FDPSCIntegrationError("stored exception adapter has an unsupported schema")
        layers = raw.get("layers", {})
        if set(layers) != set(self.injection.adapters):
            raise FDPSCIntegrationError("stored exception layer registry differs from target manifest")
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            state = layers[logical_id]
            if int(state["B"].shape[1]) == 0:
                adapter.clear_active_exception()
            else:
                adapter.set_active_exception(
                    state["B"], state["A"], adapter_id=adapter_id
                )

    def _prepare_episode_metric_tracking(self) -> None:
        """Initialize Section-28 adaptation metrics from episode metadata."""

        self._episode_early_losses = []
        self._episode_start_persistent_commit_count = int(
            self.state_machine.persistent_commit_count
        )
        self._loss_threshold_reached = False
        raw_threshold = self._episode_metadata.get("jepa_loss_threshold")
        self._jepa_loss_threshold = None
        self._jepa_loss_threshold_reason = "threshold_not_provided"
        if raw_threshold is not None:
            try:
                parsed = float(raw_threshold)
            except (TypeError, ValueError):
                self._jepa_loss_threshold_reason = "threshold_is_not_numeric"
            else:
                if math.isfinite(parsed):
                    self._jepa_loss_threshold = parsed
                    self._jepa_loss_threshold_reason = ""
                else:
                    self._jepa_loss_threshold_reason = "threshold_is_not_finite"

        if self._jepa_loss_threshold is None:
            self.metrics.record_nullable(
                "time_to_threshold_replans",
                None,
                status="unavailable",
                reason=self._jepa_loss_threshold_reason,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
            )
        else:
            self.metrics.record_nullable(
                "time_to_threshold_replans",
                None,
                status="pending",
                reason="awaiting_online_loss",
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"loss_threshold": self._jepa_loss_threshold},
            )
        self.metrics.record_nullable(
            "next_episode_early_loss_decline",
            None,
            status="insufficient_observations",
            reason="requires_at_least_two_online_optimizer_steps",
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={
                "early_window_steps": 3,
                "memory_commit_count_at_episode_start": (
                    self._episode_start_persistent_commit_count
                ),
            },
        )

    def _record_route_metrics(self, decision: RouteDecision) -> None:
        matched = decision.adapter_id is not None
        self.metrics.record_nullable(
            "routed_exception_id",
            decision.adapter_id,
            status="available" if matched else "not_applicable",
            reason=None if matched else decision.reason,
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={"matched": matched, "route_reason": decision.reason},
        )
        similarity_status = "available" if decision.similarity is not None else "unavailable"
        self.metrics.record_nullable(
            "routed_exception_similarity",
            decision.similarity,
            status=similarity_status,
            reason=None if decision.similarity is not None else decision.reason,
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={"matched": matched, "route_reason": decision.reason},
        )
        if matched:
            self.metrics.record(
                "route_rejection_count",
                self.metrics.counter("route_rejection_count"),
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"kind": "counter_snapshot", "route_rejected": False},
            )
        else:
            self.metrics.increment(
                "route_rejection_count",
                episode_id=self._active_episode_id,
            )

    def _record_early_adaptation_metrics(self, loss: float, optimizer_step: int) -> None:
        value = float(loss)
        if len(self._episode_early_losses) < 3:
            self._episode_early_losses.append(value)
            observations = len(self._episode_early_losses)
            if observations >= 2:
                decline = (
                    self._episode_early_losses[0]
                    - self._episode_early_losses[-1]
                ) / float(observations - 1)
                self.metrics.record_nullable(
                    "next_episode_early_loss_decline",
                    decline,
                    status="available",
                    episode_id=self._active_episode_id,
                    replan_index=self._replan_index,
                    context_identifier=self._active_context,
                    tags={
                        "observed_optimizer_steps": observations,
                        "early_window_steps": 3,
                        "optimizer_step": int(optimizer_step),
                        "definition": "(first_loss-last_loss)/(observations-1)",
                        "memory_commit_count_at_episode_start": (
                            self._episode_start_persistent_commit_count
                        ),
                    },
                )

        threshold = self._jepa_loss_threshold
        if (
            threshold is not None
            and not self._loss_threshold_reached
            and value <= threshold
        ):
            self._loss_threshold_reached = True
            self.metrics.record_nullable(
                "time_to_threshold_replans",
                self._replan_index + 1,
                status="available",
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                context_identifier=self._active_context,
                tags={
                    "loss_threshold": threshold,
                    "observed_loss": value,
                    "optimizer_step": int(optimizer_step),
                    "counting": "one_based_completed_replan",
                },
            )

    def begin_episode(
        self,
        *,
        episode_id: str,
        context_identifier: str,
        initial_obs: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._active_episode_id is not None or self.state_machine.active:
            raise FDPSCIntegrationError("cannot overlap FD-PSC episodes")
        if str(episode_id) != self.next_episode_id:
            raise FDPSCIntegrationError(
                f"episode id must be the deterministic next id {self.next_episode_id!r}"
            )
        context = str(context_identifier)
        required_context_splits = ["calibration", "commit_query"]
        if self.config.gates.plasticity_enabled:
            required_context_splits.extend(("plasticity_support", "plasticity_query"))
        if self.external is not None:
            # This check precedes state-machine activation, support admission,
            # and every optimizer step.  A single calibration context is not
            # evidence that the current episode belongs to it.
            self.external.validate_context_splits(context, required_context_splits)
        self._active_episode_id = str(episode_id)
        self._active_context = context
        self._episode_metadata = dict(metadata or {})
        self._support_segments = []
        self._support_context_descriptors = []
        self._support_hashes = set()
        self._support_transition_cursor = 0
        self._replan_index = 0
        self._online_step_count = 0
        self._online_loss = 0.0
        self._episode_early_losses = []
        self._episode_start_persistent_commit_count = int(
            self.state_machine.persistent_commit_count
        )
        self._jepa_loss_threshold = None
        self._jepa_loss_threshold_reason = "threshold_not_provided"
        self._loss_threshold_reached = False
        self._centered_reason = None
        self._calibration_candidate_count = 0
        self._plasticity_probe_rng = None
        self._latest_online_gradients = {}
        self._latest_corrected_gradients = {}
        self._latest_gradient_cosines = {}
        self._sdc_anchor_records = ()
        self._sdc_anchor_before_loss = None
        self._sdc_anchor_current_loss = None
        self._sdc_anchor_regression_value = None
        self.geometry.reset_episode()
        for tracker in self._sdc_trackers.values():
            tracker.reset()
        self._sdc_active = {
            key: (self.config.sdc.enabled and not self.config.sdc.event_triggered)
            for key in self.injection.adapters
        }
        try:
            self.state_machine.begin_episode(self._active_episode_id, context)
            if self.external is not None:
                self.external.begin_episode(
                    self._active_episode_id,
                    context,
                    required_splits=required_context_splits,
                )
                needs_anchor = (
                    self.config.gates.anchor_enabled
                    or self.config.gradient_geometry.enabled
                    or (
                        self.config.spectral_surgery.enabled
                        and self.config.spectral_surgery.anchor_weight > 0
                    )
                )
                if needs_anchor and not (
                    self.external._by_context["anchor"].get(context)
                    or self.external._records["anchor"]
                ):
                    raise FDPSCIntegrationError("fixed anchor split is empty")
                if needs_anchor:
                    available_anchor = self.external.anchor(context)
                    if len(available_anchor) < self.config.anchor_data.windows:
                        raise FDPSCIntegrationError(
                            "fixed anchor split has fewer records than anchor_data.windows: "
                            f"{len(available_anchor)} < {self.config.anchor_data.windows}"
                        )
                if self.config.gradient_geometry.enabled:
                    needed_current = (
                        self.config.gradient_geometry.current_batches
                        * self.config.gradient_geometry.windows_per_batch
                    )
                    available_current = self.external.calibration(context)
                    if len(available_current) < needed_current:
                        raise FDPSCIntegrationError(
                            "external calibration cannot satisfy fixed gradient batches: "
                            f"{len(available_current)} < {needed_current}"
                        )
                    needed_anchor = (
                        self.config.gradient_geometry.anchor_batches
                        * self.config.gradient_geometry.windows_per_batch
                    )
                    available_anchor = self.external.anchor(context)
                    if len(available_anchor) < needed_anchor:
                        raise FDPSCIntegrationError(
                            "immutable anchor cannot satisfy fixed gradient batches: "
                            f"{len(available_anchor)} < {needed_anchor}"
                        )
            if self.config.run_mode != "accumulate" or self._episode_sequence == 0:
                self._begin_zero_pilot()
            else:
                # Baseline 4 is one fixed-rank full-depth adapter whose
                # weights continue across episodes. It is never reset/merged.
                for adapter in self.injection.adapters.values():
                    adapter.clear_active_exception()
            # From this point onward the production episode has been admitted.
            # Allocate its deterministic high-water mark before routing or
            # applying a routed adapter so any later failure is auditable by
            # the ordinary abort snapshot and cannot reuse the episode ID.
            self._episode_sequence += 1
            self._context_embedding = self._context_descriptor(initial_obs, context)
            self._route = self.exception_router.begin_episode(
                self._active_episode_id, self._context_embedding
            )
            self._record_route_metrics(self._route)
            self._apply_exception_state(
                None if self.config.run_mode == "accumulate" else self._route.adapter_id
            )
            self._prepare_episode_metric_tracking()
            self._episode_start_adapter_states = self._adapter_participant.state_dict()
            self._episode_start_repair_state = (
                copy.deepcopy(self.repair_engine.state_dict())
                if self.repair_engine is not None
                else None
            )
            for logical_id, adapter in sorted(self.injection.adapters.items()):
                self.metrics.record(
                    "episodic_rank",
                    adapter.pilot_actual_rank,
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                )
            self.metrics.record(
                "episode_begin",
                1,
                episode_id=self._active_episode_id,
                context_identifier=context,
                tags={
                    "route": self._route.adapter_id or "slow_only",
                    "route_reason": self._route.reason,
                    "run_mode": self.config.run_mode,
                },
            )
            self.assert_base_frozen()
            return {
                "episode_id": self._active_episode_id,
                "context_identifier": context,
                "route_adapter_id": self._route.adapter_id,
                "route_similarity": self._route.similarity,
            }
        except BaseException:
            # Use the ordinary abort path so an episode that was activated and
            # counted before a later begin failure gets the same auditable
            # between-episode snapshot as every other abort.
            self.abort_episode("begin_episode_failed")
            raise

    def require_active_episode(self) -> None:
        if self._active_episode_id is None or self.state_machine.state not in {
            FDPSCState.EPISODE_PILOT,
            FDPSCState.EPISODE_CENTERED,
        }:
            raise FDPSCIntegrationError("FD-PSC online update has no active episode")

    def register_support_segment(
        self,
        obs: Mapping[str, Tensor],
        actions: Tensor,
        *,
        iteration: int,
    ) -> DataIdentity:
        self.require_active_episode()
        if not isinstance(obs, Mapping) or not obs:
            raise FDPSCIntegrationError("support observation must be a non-empty mapping")
        if not isinstance(actions, Tensor) or actions.ndim < 2:
            raise FDPSCIntegrationError("support actions must be a batched time tensor")
        time = int(actions.shape[1])
        visual = obs.get("visual")
        if not isinstance(visual, Tensor) or visual.ndim < 2 or int(visual.shape[1]) != time + 1:
            raise FDPSCIntegrationError("support must contain T+1 visual frames for T actions")
        content_hash = _hash_tensor_tree({"obs": obs, "actions": actions})
        if content_hash in self._support_hashes:
            raise FDPSCIntegrationError("duplicate support content within an episode")
        explicit: Optional[Mapping[str, Any]] = None
        declared = self._episode_metadata.get("support_identities")
        if isinstance(declared, Mapping):
            candidate_identity = declared.get(str(int(iteration)), declared.get(int(iteration)))
            if isinstance(candidate_identity, Mapping):
                explicit = candidate_identity
        if explicit is not None:
            identity_payload = dict(explicit)
            supplied_hash = identity_payload.get("content_hash")
            if supplied_hash is not None and str(supplied_hash) != content_hash:
                raise FDPSCIntegrationError(
                    "declared support identity content_hash differs from audited tensors"
                )
            identity_payload["content_hash"] = content_hash
            identity_payload.setdefault("context_identifier", self._active_context)
        else:
            seed = self._episode_metadata.get("seed", "unknown")
            sample = self._episode_metadata.get("sample_idx", "unknown")
            environment = self._episode_metadata.get("environment", self._episode_metadata.get("env", "env"))
            task = self._episode_metadata.get("task", self._active_context)
            trajectory_id = str(
                self._episode_metadata.get(
                    "trajectory_id",
                    f"trajectory:{environment}:{task}:seed={seed}:sample={sample}",
                )
            )
            start = self._support_transition_cursor
            identity_payload = {
                "record_id": f"support:{trajectory_id}:replan={int(iteration)}",
                "context_identifier": self._active_context,
                "trajectory_id": trajectory_id,
                "transition_ids": [
                    f"{trajectory_id}:transition={start + index}"
                    for index in range(time)
                ],
                "frame_ids": [
                    f"{trajectory_id}:frame={start + index}"
                    for index in range(time + 1)
                ],
                "content_hash": content_hash,
            }
        identity = DataIdentity.from_mapping(identity_payload, split_name="support")
        if self.external is not None:
            self.external.open_support_registration()
            self.external.audit_and_register_support(identity)
        self._support_hashes.add(content_hash)
        self._support_transition_cursor += time
        segment = _SupportSegment(
            identity=identity,
            obs={key: value.detach().cpu().clone() for key, value in obs.items()},
            actions=actions.detach().cpu().clone(),
            iteration=int(iteration),
            episode_id=str(self._active_episode_id),
            context_identifier=str(self._active_context),
            preprocess_hash=str(self.preprocess_hash or ""),
            latent_adapter_schema=LATENT_SCHEMA_VERSION,
            source_record_ids=(identity.record_id,),
            source_iterations=(int(iteration),),
        )
        self._support_segments.append(segment)
        descriptor = self._context_descriptor(segment.obs, str(self._active_context))
        if descriptor is not None:
            self._support_context_descriptors.append(descriptor)
            # Accepted support, rather than a future outcome/query, refines the
            # immutable context prototype used by replay and exception commit.
            self._context_embedding = self._current_context_prototype()
        self.state_machine.note_support_window(1)
        self.metrics.record(
            "support_registered",
            1,
            episode_id=self._active_episode_id,
            replan_index=int(iteration),
            context_identifier=self._active_context,
            tags={"content_hash": content_hash},
        )
        return identity

    # ------------------------------------------------------------------
    # Online adapter optimization, conflict trigger, SLICE, and two-pass SDC
    # ------------------------------------------------------------------
    def prepare_online_mode(self, *, predictor_train: bool, encoder_train: bool) -> None:
        self.require_active_episode()
        if self._online_mode_depth:
            raise FDPSCIntegrationError("online adaptation mode is not re-entrant")
        self._online_update_started_at = time.perf_counter()
        if self.external is not None:
            self.external.seal_support_for_online_update()
        for parameter in self.wm.parameters():
            parameter.requires_grad_(False)
        for adapter in self.injection.adapters.values():
            if not adapter.pilot_frozen:
                if adapter.pilot_A is not None:
                    adapter.pilot_A.requires_grad_(True)
                if adapter.pilot_B is not None:
                    adapter.pilot_B.requires_grad_(True)
            if adapter.centered_active:
                if adapter.center_A is not None:
                    adapter.center_A.requires_grad_(True)
                if adapter.center_B is not None:
                    adapter.center_B.requires_grad_(True)
        # Preserve the legacy AdaJEPA choice of training only predictor and,
        # optionally, encoder.  Starting from global eval also prevents an
        # unrelated theta_0 BatchNorm in action/proprio/decoder branches from
        # changing a persistent buffer.
        self.wm.eval()
        self.wm.predictor.train(bool(predictor_train))
        self.wm.encoder.train(bool(encoder_train))
        self.injection.enforce_frozen_base_eval()
        if self.config.gradient_geometry.enabled or self.config.sdc.enabled:
            self._online_hooks = EffectiveWeightGradientHooks(
                self.injection.gradient_modules()
            )
        self._online_mode_depth = 1
        self.assert_base_frozen()

    def online_parameter_groups(self, *, include_encoder: bool) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        if not self._online_mode_depth:
            raise FDPSCIntegrationError("online parameter groups requested outside online mode")
        predictor = self.injection.predictor_parameters()
        encoder = self.injection.encoder_parameters() if include_encoder else []
        base_ids = {id(item.tensor) for item in self._base_tensors if item.is_parameter}
        if any(id(parameter) in base_ids for parameter in (*predictor, *encoder)):
            raise FDPSCIntegrationError("theta_0 parameter leaked into the adapter optimizer")
        return predictor, encoder

    def prepare_effective_loss(self, loss: Tensor) -> Tensor:
        """Compatibility shim; two-pass SDC is applied by ``backward_with_sdc``."""

        return loss

    @staticmethod
    def capture_update_rng() -> RNGSnapshot:
        """Capture RNG immediately before an online JEPA forward."""

        return RNGSnapshot.capture()

    def _sdc_proxy(self, gradients: Mapping[str, Tensor]) -> Optional[Tensor]:
        proxy: Optional[Tensor] = None
        for logical_id, gradient in sorted(gradients.items()):
            if not self._sdc_active.get(logical_id, False):
                continue
            spectrum = self._base_spectra.get(logical_id)
            if spectrum is None:
                continue
            correction = sdc_correct_gradient(
                gradient,
                spectrum,
                minimum_gamma=self.config.sdc.minimum_gamma,
                epsilon=self.config.gradient_geometry.epsilon,
                active=True,
            )
            if not correction.applied:
                continue
            difference = correction.gradient - gradient
            adapter = self.injection.adapters[logical_id]
            term: Optional[Tensor] = None
            if adapter.centered_active and adapter.center_B is not None and adapter.center_A is not None:
                term = effective_gradient_proxy(
                    difference,
                    adapter.center_B,
                    adapter.center_A,
                    scaling=adapter.centered_scaling,
                    centered_initial=(adapter.center_B0, adapter.center_A0),
                )
            elif adapter.pilot_B is not None and adapter.pilot_A is not None and not adapter.pilot_frozen:
                term = effective_gradient_proxy(
                    difference,
                    adapter.pilot_B,
                    adapter.pilot_A,
                    scaling=adapter.pilot_scaling,
                )
            if term is not None:
                proxy = term if proxy is None else proxy + term
                self.metrics.record(
                    "sdc_gamma",
                    correction.gamma,
                    episode_id=self._active_episode_id,
                    replan_index=self._replan_index,
                    logical_layer_id=logical_id,
                )
        return proxy

    def backward_with_sdc(
        self,
        loss: Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        loss_closure: Optional[Any] = None,
        forward_rng: Optional[RNGSnapshot] = None,
    ) -> None:
        """Run one ordinary backward or the event-triggered exact two-pass path.

        Pass 1 measures effective-weight gradients with exact module hooks.  If
        an active SDC layer needs correction, those parameter gradients are
        discarded and pass 2 differentiates the unchanged JEPA loss plus a
        stop-gradient factor proxy for ``G_corrected - G``.
        """

        active = self.config.sdc.enabled and any(self._sdc_active.values())
        if not active or self._online_hooks is None:
            loss.backward()
            return
        if loss_closure is None or forward_rng is None:
            raise FDPSCIntegrationError(
                "active two-pass SDC requires a loss closure and pre-forward RNG snapshot"
            )
        # Pass 1 is measurement only.
        loss.backward()
        first_gradients = self._online_hooks.gradients
        proxy = self._sdc_proxy(first_gradients)
        if proxy is None:
            return
        optimizer.zero_grad(set_to_none=True)
        self._online_hooks.reset()
        # Rewind to the exact pre-forward state, recompute the unchanged JEPA
        # forward with identical stochastic masks, then leave global RNG at
        # the state produced by one forward (the second pass).
        forward_rng.restore()
        second_loss = loss_closure()
        (second_loss + proxy).backward()
        self.metrics.increment("sdc_two_pass_updates", episode_id=self._active_episode_id)

    def note_optimizer_step(self, step: int, loss: float) -> None:
        self.require_active_episode()
        self.state_machine.note_online_update(1)
        self._online_step_count += 1
        self._online_loss = float(loss)
        if self._online_hooks is not None:
            gradients = self._online_hooks.gradients
            self._latest_online_gradients = {
                key: value.detach().clone() for key, value in gradients.items()
            }
            self._online_hooks.reset()
        self.metrics.record(
            "online_jepa_loss",
            float(loss),
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={"optimizer_step": int(step)},
        )
        self.metrics.record(
            "current_jepa_loss",
            float(loss),
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={"optimizer_step": int(step)},
        )
        self._record_early_adaptation_metrics(float(loss), int(step))
        self.assert_base_frozen()

    def _record_active_constraints(
        self,
        logical_layer_id: str,
        constraints: Optional[Sequence[str]],
        *,
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        dimensions = {
            "episode_id": self._active_episode_id,
            "replan_index": self._replan_index,
            "logical_layer_id": logical_layer_id,
        }
        if status == "available":
            names = tuple(str(item) for item in (constraints or ()))
            self.metrics.record_nullable(
                "active_constraints",
                ",".join(names) if names else "none",
                status="available",
                tags={"projection_method": "dual_constraint"},
                **dimensions,
            )
            self.metrics.record_nullable(
                "active_constraint_count",
                len(names),
                status="available",
                tags={"projection_method": "dual_constraint"},
                **dimensions,
            )
            return
        self.metrics.record_nullable(
            "active_constraints",
            None,
            status=status,
            reason=reason,
            tags={"projection_method": self.config.gradient_geometry.projection_method},
            **dimensions,
        )
        self.metrics.record_nullable(
            "active_constraint_count",
            None,
            status=status,
            reason=reason,
            tags={"projection_method": self.config.gradient_geometry.projection_method},
            **dimensions,
        )

    def _correct_online_gradients(
        self,
        history_gradients: Optional[Mapping[str, Tensor]] = None,
        anchor_gradients: Optional[Mapping[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], bool]:
        corrected: Dict[str, Tensor] = {}
        cfg = self.config.gradient_geometry
        for logical_id, current in sorted(self._latest_online_gradients.items()):
            history = (history_gradients or {}).get(logical_id)
            anchor = (anchor_gradients or {}).get(logical_id)
            history_status, anchor_status = self.geometry.update(
                logical_id,
                current,
                history,
                anchor,
                epsilon=cfg.epsilon,
            )
            self._latest_gradient_cosines[logical_id] = {
                "history": history_status.cosine,
                "anchor": anchor_status.cosine,
            }
            for reference_name, conflict_status in (
                ("history", history_status),
                ("anchor", anchor_status),
            ):
                self.metrics.record_nullable(
                    "conflict_ema",
                    conflict_status.ema,
                    status=(
                        "available"
                        if conflict_status.ema is not None
                        else "unavailable"
                    ),
                    reason=(
                        None
                        if conflict_status.ema is not None
                        else "no_finite_reference_cosine"
                    ),
                    episode_id=self._active_episode_id,
                    replan_index=self._replan_index,
                    logical_layer_id=logical_id,
                    tags={
                        "scope": "logical_layer",
                        "reference": reference_name,
                        "current_cosine_available": conflict_status.available,
                        "consecutive_conflicts": (
                            conflict_status.consecutive_conflicts
                        ),
                        "triggered": conflict_status.triggered,
                    },
                )
            value = current
            if cfg.projection_method == "dual_constraint":
                projection = dual_constraint_projection(
                    current,
                    history=history,
                    anchor=anchor,
                    history_slack=cfg.history_slack,
                    anchor_slack=cfg.anchor_slack,
                    epsilon=cfg.epsilon,
                )
                self._record_active_constraints(
                    logical_id,
                    projection.active_constraints if projection.feasible else None,
                    status="available" if projection.feasible else "unavailable",
                    reason=None if projection.feasible else projection.reason,
                )
                if projection.feasible:
                    value = projection.gradient
                else:
                    self.diagnostics.fallback(
                        "dual_constraint_projection_infeasible",
                        projection.reason,
                        episode_id=self._active_episode_id,
                        logical_layer_id=logical_id,
                    )
                    self.metrics.record(
                        "gradient_projection_feasible",
                        0,
                        episode_id=self._active_episode_id,
                        replan_index=self._replan_index,
                        logical_layer_id=logical_id,
                        tags={"reason": projection.reason},
                    )
                    # This layer is intentionally absent from the SLICE
                    # initializer map. Falling back to the conflicting raw
                    # gradient would violate both active constraints.
                    continue
            elif cfg.projection_method in {"c_pcgrad", "per_step_c_pcgrad"}:
                self._record_active_constraints(
                    logical_id,
                    None,
                    status="not_applicable",
                    reason="projection_method_has_no_active_set",
                )
                if history is not None:
                    value = c_pcgrad(
                        value,
                        history,
                        coefficient=cfg.c_pcgrad_coefficient,
                        epsilon=cfg.epsilon,
                    )
                if anchor is not None:
                    value = c_pcgrad(
                        value,
                        anchor,
                        coefficient=cfg.c_pcgrad_coefficient,
                        epsilon=cfg.epsilon,
                    )
            else:
                raise FDPSCIntegrationError(
                    f"unsupported gradient projection method {cfg.projection_method!r}"
                )
            corrected[logical_id] = value.detach().clone()
            self.metrics.record(
                "gradient_history_cosine",
                history_status.cosine,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                logical_layer_id=logical_id,
            )
            self.metrics.record(
                "gradient_anchor_cosine",
                anchor_status.cosine,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                logical_layer_id=logical_id,
            )
            self.metrics.record(
                "gradient_correction_norm",
                float(
                    torch.linalg.vector_norm(
                        (value - current).to(dtype=torch.float32)
                    ).detach().cpu()
                ),
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                logical_layer_id=logical_id,
            )

        history_global = global_weighted_cosine(
            self._latest_online_gradients,
            history_gradients or {},
            weighting=cfg.global_cosine_weighting,
            epsilon=cfg.epsilon,
        )
        anchor_global = global_weighted_cosine(
            self._latest_online_gradients,
            anchor_gradients or {},
            weighting=cfg.global_cosine_weighting,
            epsilon=cfg.epsilon,
        )
        history_trigger, anchor_trigger = self.geometry.update_global(
            history_global.value,
            anchor_global.value,
        )
        for reference_name, conflict_status in (
            ("history", history_trigger),
            ("anchor", anchor_trigger),
        ):
            self.metrics.record_nullable(
                "conflict_ema",
                conflict_status.ema,
                status=(
                    "available"
                    if conflict_status.ema is not None
                    else "unavailable"
                ),
                reason=(
                    None
                    if conflict_status.ema is not None
                    else "no_finite_global_reference_cosine"
                ),
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                tags={
                    "scope": "global_weighted",
                    "reference": reference_name,
                    "current_cosine_available": conflict_status.available,
                    "consecutive_conflicts": conflict_status.consecutive_conflicts,
                    "triggered": conflict_status.triggered,
                },
            )
        self.metrics.record(
            "rho_history",
            history_global.value,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={
                "scope": "global_weighted",
                "weighting": cfg.global_cosine_weighting,
                "available": history_global.available,
            },
        )
        self.metrics.record(
            "rho_anchor",
            anchor_global.value,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={
                "scope": "global_weighted",
                "weighting": cfg.global_cosine_weighting,
                "available": anchor_global.available,
            },
        )
        triggered = history_trigger.triggered or anchor_trigger.triggered
        return corrected, triggered

    def _activate_centered(self, trainer: Any, corrected: Mapping[str, Tensor]) -> bool:
        started = time.perf_counter()
        activated = 0
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            gradient = corrected.get(logical_id)
            if gradient is None or adapter.centered_active:
                continue
            group = self._entry_by_id[logical_id].module_group
            learning_rate = trainer.lr if group == "predictor" else trainer.encoder_lr
            baseline_rank = min(
                adapter.actual_rank,
                int(gradient.shape[0]),
                int(gradient.shape[1]),
            )
            reference = adapter._reference()
            baseline_a = torch.empty(
                (baseline_rank, int(gradient.shape[1])),
                device=reference.device,
                dtype=reference.dtype,
            )
            with torch.no_grad():
                baseline_a.uniform_(
                    -1.0 / math.sqrt(int(gradient.shape[1])),
                    1.0 / math.sqrt(int(gradient.shape[1])),
                    generator=self._episode_generator(logical_id, "slice-baseline"),
                )
            baseline_b = torch.zeros(
                (int(gradient.shape[0]), baseline_rank),
                device=reference.device,
                dtype=reference.dtype,
            )
            optimizer_betas = (0.9, 0.999)
            optimizer_epsilon = 1.0e-8
            optimizer_weight_decay = 0.01 if trainer.optimizer_name == "adamw" else 0.0
            first = simulate_factor_first_step(
                gradient,
                baseline_b,
                baseline_a,
                scaling=self.config.episodic_lora.alpha / baseline_rank,
                optimizer_name=trainer.optimizer_name,
                learning_rate=learning_rate,
                betas=optimizer_betas,
                optimizer_epsilon=optimizer_epsilon,
                weight_decay=optimizer_weight_decay,
            )
            baseline = first.norm if first.finite else None
            initialized = initialize_slice(
                gradient,
                requested_rank=self.config.slice.rank,
                mode=self.config.slice.initialization,
                fallback_mode=self.config.slice.fallback_initialization,
                oversampling=self.config.slice.randomized_svd_oversampling,
                power_iterations=self.config.slice.power_iterations,
                generator=self._episode_generator(logical_id, "slice"),
                alpha=self.config.episodic_lora.alpha,
                baseline_first_step_norm=baseline,
                magnitude_mode=self.config.slice.magnitude_mode,
                maximum_beta=self.config.slice.maximum_scale,
                optimizer_name=trainer.optimizer_name,
                learning_rate=learning_rate,
                betas=optimizer_betas,
                optimizer_epsilon=optimizer_epsilon,
                weight_decay=optimizer_weight_decay,
            )
            if not initialized.success or initialized.b0 is None or initialized.a0 is None:
                self.diagnostics.fallback(
                    "slice_fallback_to_pilot",
                    initialized.reason,
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                )
                continue
            adapter.activate_centered_branch(
                initialized.b0,
                initialized.a0,
                alpha=self.config.episodic_lora.alpha,
            )
            activated += 1
            self.metrics.record(
                "slice_actual_rank",
                initialized.actual_rank,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                logical_layer_id=logical_id,
                tags={"mode": initialized.mode, "beta": initialized.beta},
            )
        if activated:
            self.state_machine.activate_centered("gradient_conflict_trigger")
            self._centered_reason = "gradient_conflict_trigger"
            self.metrics.record(
                "slice_trigger",
                1,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                tags={"activated_layers": activated},
            )
            self.metrics.record(
                "slice_latency_s",
                time.perf_counter() - started,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
            )
            return True
        self.metrics.record(
            "slice_trigger",
            0,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={"reason": "no_layer_initialized"},
        )
        self.metrics.record(
            "slice_latency_s",
            time.perf_counter() - started,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
        )
        return False

    def _sdc_check_scheduled(self) -> bool:
        return (
            (self._replan_index + 1) % self.config.sdc.check_every_replans == 0
        )

    def _scheduled_sdc_anchor_regression(self, trainer: Any) -> float:
        """Return fixed-anchor ``L(P_fast) - L(P_before)`` for this check.

        ``P_before`` is the routed episode-start state captured after the
        zero-function Pilot and fixed exception route were installed;
        ``P_fast`` is the live episodic state after the current finetune event.
        Both losses use the exact same immutable, deterministically ordered
        anchor records.  ``_evaluate_state`` performs frozen eval under
        ``_preserve_adapter_runtime``, which restores module modes, adapter
        values/Parameter identities, requires-grad flags, and all process RNG.
        Production configuration requires this split, so absence or a
        non-finite loss fails closed instead of fabricating a safety signal.
        """

        self.require_active_episode()
        if not self._sdc_check_scheduled():
            raise FDPSCIntegrationError(
                "SDC anchor regression requested outside a scheduled check"
            )
        if not self._sdc_anchor_records:
            self._sdc_anchor_records = self._fixed_anchor_records()
        records = self._sdc_anchor_records
        if self._sdc_anchor_before_loss is None:
            self._sdc_anchor_before_loss = self._evaluate_state(
                trainer,
                records,
                state="before",
            )
            self.metrics.record(
                "sdc_anchor_loss",
                self._sdc_anchor_before_loss,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                tags={"state": "before", "record_count": len(records)},
            )
        current = self._evaluate_state(trainer, records, state="fast")
        regression = current - self._sdc_anchor_before_loss
        if not math.isfinite(current) or not math.isfinite(regression):
            raise FDPSCIntegrationError("non-finite scheduled SDC anchor regression")
        self._sdc_anchor_current_loss = current
        self._sdc_anchor_regression_value = regression
        self.metrics.record(
            "sdc_anchor_loss",
            current,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={"state": "fast", "record_count": len(records)},
        )
        self.metrics.record(
            "sdc_anchor_regression",
            regression,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
            tags={
                "before_loss": self._sdc_anchor_before_loss,
                "current_loss": current,
                "record_count": len(records),
            },
        )
        return regression

    def _update_sdc_events(self, trainer: Any) -> None:
        if not self.config.sdc.enabled:
            return
        anchor_regression: Optional[float] = None
        if self.config.sdc.event_triggered and self._sdc_check_scheduled():
            # One shared anchor evaluation serves every logical-layer tracker;
            # only drift and rho_anchor are layer-specific.
            anchor_regression = self._scheduled_sdc_anchor_regression(trainer)
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            if not self.config.sdc.event_triggered:
                self._sdc_active[logical_id] = True
                continue
            spectrum = self._base_spectra.get(logical_id)
            drift_value: Optional[float] = None
            if spectrum is not None:
                drift = spectral_drift(
                    LowRankFactors(
                        adapter.get_episodic_factors().B,
                        adapter.get_episodic_factors().A,
                    ),
                    spectrum,
                    epsilon=self.config.gradient_geometry.epsilon,
                )
                drift_value = drift.value if drift.available else None
            cosine = self._latest_gradient_cosines.get(logical_id, {}).get("anchor")
            status = self._sdc_trackers[logical_id].update(
                self._replan_index,
                drift_value,
                anchor_regression=anchor_regression,
                anchor_cosine=cosine,
            )
            self._sdc_active[logical_id] = status.active
            if status.checked:
                self.metrics.record(
                    "sdc_trigger_active",
                    int(status.active),
                    episode_id=self._active_episode_id,
                    replan_index=self._replan_index,
                    logical_layer_id=logical_id,
                    tags={
                        "reason": status.reason,
                        "drift": drift_value,
                        "anchor_regression": anchor_regression,
                        "anchor_cosine": cosine,
                    },
                )

    @staticmethod
    def _stable_record_key(record: Any) -> str:
        if isinstance(record, ExternalRecord):
            return record.record_id
        if isinstance(record, ReplayWindow):
            return record.window_id
        if isinstance(record, Mapping):
            return str(
                record.get(
                    "record_id",
                    record.get("__fd_psc_context_identifier", repr(sorted(record))),
                )
            )
        return str(getattr(record, "record_id", getattr(record, "window_id", repr(record))))

    def _fixed_external_gradient_records(
        self,
        records: Sequence[Any],
        *,
        batches: int,
        source: str,
    ) -> Tuple[Any, ...]:
        count = int(batches) * self.config.gradient_geometry.windows_per_batch
        ordered = sorted(records, key=self._stable_record_key)
        if len(ordered) < count:
            raise FDPSCIntegrationError(
                f"{source} has {len(ordered)} records but fixed gradient plan requires {count}"
            )
        # Stable-ID, without-replacement selection. Equal-size batches may be
        # flattened because external_loss_tensor applies the same mean JEPA
        # reduction; this is exactly the sample-weighted aggregate gradient.
        return tuple(ordered[:count])

    def _balanced_history_gradient_windows(
        self,
        windows: Sequence[ReplayWindow],
    ) -> Tuple[ReplayWindow, ...]:
        if not windows:
            return ()
        count = (
            self.config.gradient_geometry.history_batches
            * self.config.gradient_geometry.windows_per_batch
        )
        buckets: Dict[str, List[ReplayWindow]] = {}
        for window in sorted(windows, key=lambda item: (item.context_identifier, item.window_id)):
            buckets.setdefault(window.context_identifier, []).append(window)
        contexts = sorted(buckets)
        rng = random.Random(
            _stable_seed(
                self.config.seed,
                self._active_episode_id,
                self._replan_index,
                "history-gradient-sampler",
            )
        )
        available = {key: list(value) for key, value in buckets.items()}
        selected: List[ReplayWindow] = []
        for index in range(count):
            context = contexts[index % len(contexts)]
            pool = available[context]
            if pool:
                chosen = pool.pop(rng.randrange(len(pool)))
            else:
                source = buckets[context]
                chosen = source[rng.randrange(len(source))]
            selected.append(chosen.clone())
        duplicate_rate = 1.0 - len({item.window_id for item in selected}) / float(len(selected))
        self.metrics.record(
            "gradient_history_duplicate_rate",
            duplicate_rate,
            episode_id=self._active_episode_id,
            replan_index=self._replan_index,
        )
        return tuple(selected)

    def _fixed_anchor_records(self) -> Tuple[ExternalRecord, ...]:
        if self.external is None:
            raise FDPSCIntegrationError("fixed anchor registry is unavailable")
        records = sorted(
            self.external.anchor(self._active_context),
            key=lambda record: record.record_id,
        )
        count = self.config.anchor_data.windows
        if len(records) < count:
            raise FDPSCIntegrationError(
                f"fixed anchor has {len(records)} records but anchor_data.windows={count}"
            )
        return tuple(records[:count])

    def _evaluate_conflict_trigger(
        self,
        trainer: Any,
        segments: Sequence[Tuple[Mapping[str, Tensor], Tensor]],
        step_losses: Sequence[float],
    ) -> bool:
        self.require_active_episode()
        # Trigger gradients are evaluated on fixed calibration/history/anchor
        # under the same P_fast clone and the exact same mean JEPA reduction.
        # The online hook gradient remains a fallback only for external-free
        # comparison baselines.
        history_now: Dict[str, Tensor] = {}
        anchor_now: Dict[str, Tensor] = {}
        if self.config.gradient_geometry.enabled and self.external is not None:
            current_records = self._fixed_external_gradient_records(
                self.external.calibration(self._active_context),
                batches=self.config.gradient_geometry.current_batches,
                source="external calibration",
            )
            self._latest_online_gradients = self._collect_effective_gradients(
                trainer,
                current_records,
                state="fast",
                fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
            )
            replay_windows = self.replay.windows()
            if replay_windows:
                sampled_history = self._balanced_history_gradient_windows(replay_windows)
                history_now = self._collect_effective_gradients(
                    trainer,
                    self._replay_records(sampled_history),
                    state="fast",
                    fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                )
            if self.config.gradient_geometry.anchor_batches > 0:
                anchor_records = self._fixed_external_gradient_records(
                    self._fixed_anchor_records(),
                    batches=self.config.gradient_geometry.anchor_batches,
                    source="immutable anchor",
                )
                anchor_now = self._collect_effective_gradients(
                    trainer,
                    anchor_records,
                    state="fast",
                    fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                )
        corrected, conflict = self._correct_online_gradients(
            history_now,
            anchor_now,
        )
        self._latest_corrected_gradients = corrected
        enough_support = sum(
            max(0, int(segment.actions.shape[1])) for segment in self._support_segments
        ) >= self.config.gradient_geometry.minimum_transitions
        activated = False
        if (
            self.config.slice.enabled
            and conflict
            and enough_support
            and self.state_machine.state == FDPSCState.EPISODE_PILOT
        ):
            activated = self._activate_centered(trainer, corrected)
        return activated

    def after_optimizer_step(
        self,
        trainer: Any,
        segments: Sequence[Tuple[Mapping[str, Tensor], Tensor]],
        step_losses: Sequence[float],
    ) -> bool:
        """Evaluate the conflict trigger immediately after one real update.

        Fixed-split gradient collection installs cloned adapter states while it
        runs.  The live online hook is reset afterwards so those measurement
        backwards cannot leak into the next actual optimizer step.
        """

        activated = self._evaluate_conflict_trigger(
            trainer,
            segments,
            step_losses,
        )
        if self._online_hooks is not None:
            self._online_hooks.reset()
        self.assert_base_frozen()
        return activated

    def after_finetune_event(
        self,
        trainer: Any,
        segments: Sequence[Tuple[Mapping[str, Tensor], Tensor]],
        step_losses: Sequence[float],
        *,
        conflict_evaluated_per_step: bool = False,
    ) -> None:
        self.require_active_episode()
        # Compatibility for callers outside the official trainer loop.  The
        # official multi-step path has already evaluated exactly once after
        # each actual optimizer step and must not evaluate again here.
        if not conflict_evaluated_per_step:
            self.after_optimizer_step(trainer, segments, step_losses)
        self._update_sdc_events(trainer)
        if self._jepa_loss_threshold is not None and not self._loss_threshold_reached:
            self.metrics.record_nullable(
                "time_to_threshold_replans",
                None,
                status="not_reached",
                reason="latest_completed_replan_above_threshold",
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                context_identifier=self._active_context,
                tags={
                    "loss_threshold": self._jepa_loss_threshold,
                    "replans_observed": self._replan_index + 1,
                },
            )
        self._replan_index += 1
        self.assert_base_frozen()

    def finish_online_mode(self) -> None:
        if self._online_hooks is not None:
            self._online_hooks.close()
            self._online_hooks = None
        for parameter in self.wm.parameters():
            parameter.requires_grad_(False)
        self.wm.eval()
        self.injection.enforce_frozen_base_eval()
        self._online_mode_depth = 0
        if self._online_update_started_at is not None:
            self.metrics.record(
                "online_update_latency_s",
                time.perf_counter() - self._online_update_started_at,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
            )
            self._online_update_started_at = None
        self.assert_base_frozen()

    def online_metrics(self) -> Dict[str, Any]:
        return {
            "fd_psc/episode_id": self._active_episode_id,
            "fd_psc/state": self.state_machine.state.value,
            "fd_psc/online_steps": self._online_step_count,
            "fd_psc/support_windows": len(self._support_segments),
            "fd_psc/centered_active": self.state_machine.centered_activated,
            "fd_psc/sdc_active_layers": sum(self._sdc_active.values()),
            "fd_psc/route": self._route.adapter_id if self._route is not None else None,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers: states, activations, and exact effective gradients
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _preserve_adapter_runtime(self) -> Iterator[None]:
        states = self._adapter_participant.state_dict()
        # Adapter state loading supports dynamic ranks by replacing Parameter
        # objects.  During a multi-step online update an already-constructed
        # optimizer must keep owning the original Pilot/Centered objects, so
        # evaluation restores their values in place and reattaches them.
        parameter_slots = {
            logical_id: {
                name: (
                    getattr(adapter, name),
                    (
                        bool(getattr(adapter, name).requires_grad)
                        if isinstance(getattr(adapter, name), nn.Parameter)
                        else False
                    ),
                )
                for name in ("pilot_A", "pilot_B", "center_A", "center_B")
            }
            for logical_id, adapter in sorted(self.injection.adapters.items())
        }
        parameter_requires_grad = tuple(
            (parameter, bool(parameter.requires_grad))
            for parameter in self.wm.parameters()
        )
        modes = {module: module.training for module in self.wm.modules()}
        rng = RNGSnapshot.capture()
        try:
            yield
        finally:
            self._adapter_participant.load_state_dict(states)
            for logical_id, adapter in sorted(self.injection.adapters.items()):
                for name, (original, requires_grad) in parameter_slots[logical_id].items():
                    if original is None:
                        continue
                    restored = getattr(adapter, name)
                    if not isinstance(restored, nn.Parameter):
                        raise FDPSCIntegrationError(
                            f"adapter runtime restore lost {logical_id}.{name}"
                        )
                    if restored.shape != original.shape:
                        raise FDPSCIntegrationError(
                            f"adapter runtime restore changed {logical_id}.{name} shape"
                        )
                    with torch.no_grad():
                        original.copy_(restored)
                    original.requires_grad_(requires_grad)
                    setattr(adapter, name, original)
            # Runtime evaluation temporarily freezes every model parameter.
            # Restore the flags captured on entry after the original episodic
            # Parameter objects have been rebound; an optimizer constructed by
            # the online loop must retain both ownership and trainability.
            for parameter, requires_grad in parameter_requires_grad:
                parameter.requires_grad_(requires_grad)
            for module, training in modes.items():
                module.train(training)
            rng.restore()
            self.injection.enforce_frozen_base_eval()

    def _zero_episodic(self) -> None:
        with torch.no_grad():
            for adapter in self.injection.adapters.values():
                if adapter.pilot_B is not None:
                    adapter.pilot_B.zero_()
                if adapter.centered_active and adapter.center_B is not None:
                    adapter.center_B.copy_(adapter.center_B0)

    @staticmethod
    def _canonical(value: Any) -> LowRankFactors:
        return LowRankFactors(value.B.detach().clone(), value.A.detach().clone())

    def _persistent_effective_factors(self) -> Dict[str, LowRankFactors]:
        result = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            slow = self._canonical(adapter.get_slow_factors())
            exception = self._canonical(adapter.get_exception_factors())
            result[logical_id] = concatenate_factors((slow, exception))
        return result

    def _apply_candidate(self, candidate: _Candidate, *, clear_episode: bool = True) -> None:
        if clear_episode:
            self._zero_episodic()
        if candidate.proposal_type == ProposalType.GLOBAL_SLOW:
            for logical_id, adapter in sorted(self.injection.adapters.items()):
                factors = candidate.factors_by_layer[logical_id]
                adapter.replace_slow_adapter(factors.B, factors.A)
                adapter.clear_active_exception()
        else:
            for logical_id, adapter in sorted(self.injection.adapters.items()):
                factors = candidate.factors_by_layer[logical_id]
                if factors.rank == 0:
                    adapter.clear_active_exception()
                else:
                    adapter.set_active_exception(
                        factors.B,
                        factors.A,
                        adapter_id=(
                            self._route.adapter_id
                            if candidate.proposal_type == ProposalType.REPLACE_EXCEPTION
                            and self._route is not None
                            else self.exception_router.preview_next_adapter_id()
                        ),
                    )

    def _load_episode_start(self) -> None:
        if self._episode_start_adapter_states is None:
            raise FDPSCIntegrationError("episode start state is unavailable")
        self._adapter_participant.load_state_dict(
            copy.deepcopy(self._episode_start_adapter_states)
        )

    def _evaluate_state(
        self,
        trainer: Any,
        records: Sequence[Any],
        *,
        state: str,
        candidate: Optional[_Candidate] = None,
    ) -> float:
        if not records:
            raise FDPSCIntegrationError("cannot evaluate an empty fixed split")
        with self._preserve_adapter_runtime():
            fast_state = self._adapter_participant.state_dict()
            weighted_loss = 0.0
            total = len(records)
            for context, descriptor, context_records in self._group_records_for_routing(records):
                self._load_routed_evaluation_state(
                    state=state,
                    candidate=candidate,
                    context=context,
                    descriptor=descriptor,
                    fast_state=fast_state,
                )
                self.wm.eval()
                self.injection.enforce_frozen_base_eval()
                group_loss = trainer.evaluate_external_records(context_records)
                weighted_loss += len(context_records) * float(group_loss)
            value = weighted_loss / float(total)
            if not math.isfinite(value):
                raise FDPSCIntegrationError(f"non-finite {state} evaluation loss")
            return value

    def _replay_payload(self, window: ReplayWindow) -> Mapping[str, Any]:
        return {
            "frozen_visual_latent": copy.deepcopy(window.visual_latent),
            "proprio": copy.deepcopy(window.proprio),
            "actions": copy.deepcopy(window.actions),
            # Private routing metadata is ignored by AdaJEPA's payload parser
            # but keeps the production exception route reproducible when a
            # mixed-context replay batch is evaluated.
            "__fd_psc_context_identifier": window.context_identifier,
            "__fd_psc_context_embedding": copy.deepcopy(window.context_embedding),
        }

    def _replay_records(self, windows: Sequence[ReplayWindow]) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._replay_payload(window) for window in windows)

    @staticmethod
    def _support_records(
        segments: Sequence[_SupportSegment],
    ) -> Tuple[Mapping[str, Any], ...]:
        """Expose complete online windows to the unchanged JEPA loss.

        Repair deliberately consumes the already-audited support objects; it
        never manufactures a second view of the episode or consults any
        external split while taking an optimizer step.
        """

        return tuple(
            {
                "obs": dict(segment.obs),
                "actions": segment.actions,
            }
            for segment in segments
        )

    @staticmethod
    def _unit_context_descriptor(value: Any) -> Optional[Tensor]:
        if value is None:
            return None
        try:
            vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten()
        except (TypeError, ValueError, RuntimeError):
            return None
        if vector.numel() == 0 or not torch.isfinite(vector).all():
            return None
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or float(norm) <= 1.0e-12:
            return None
        return (vector / norm).clone()

    def _record_context_descriptor(self, record: Any) -> Tuple[str, Optional[Tensor]]:
        context = str(self._active_context or "")
        descriptor: Any = None
        if isinstance(record, ReplayWindow):
            context = record.context_identifier
            descriptor = record.context_embedding
        elif isinstance(record, ExternalRecord):
            context = record.context_identifier
            for key in (
                "frozen_context_embedding",
                "context_embedding",
                "context_descriptor",
            ):
                if key in record.metadata:
                    descriptor = record.metadata[key]
                    break
            if descriptor is None:
                payload = (
                    record.payload
                    if isinstance(record.payload, Mapping)
                    else self.materialize_external_payload(record)
                )
                descriptor = self._descriptor_from_payload(payload, context)
        elif isinstance(record, Mapping):
            context = str(
                record.get(
                    "__fd_psc_context_identifier",
                    record.get("context_identifier", context),
                )
            )
            descriptor = record.get(
                "__fd_psc_context_embedding",
                record.get(
                    "frozen_context_embedding",
                    record.get("context_embedding", record.get("context_descriptor")),
                ),
            )
            if descriptor is None:
                descriptor = self._descriptor_from_payload(record, context)
        if descriptor is None and context == str(self._active_context):
            descriptor = self._current_context_prototype()
        return context, self._unit_context_descriptor(descriptor)

    def _descriptor_from_payload(
        self,
        payload: Mapping[str, Any],
        context: str,
    ) -> Optional[Tensor]:
        for key in (
            "frozen_context_embedding",
            "context_embedding",
            "context_descriptor",
        ):
            if key in payload:
                return self._unit_context_descriptor(payload[key])
        obs = payload.get("obs")
        if isinstance(obs, Mapping):
            return self._context_descriptor(obs, context)
        latent_payload = payload.get(
            "frozen_visual_latent",
            payload.get("visual_latent"),
        )
        if isinstance(latent_payload, Mapping):
            try:
                latent = FrozenVisualLatent.from_payload(latent_payload)
                value = latent.tensor.detach().to(dtype=torch.float32)
                vector = value.reshape(-1, value.shape[-1]).mean(0)
                return self._unit_context_descriptor(vector)
            except (KeyError, TypeError, ValueError, RuntimeError):
                return None
        return None

    def _group_records_for_routing(
        self,
        records: Sequence[Any],
    ) -> List[Tuple[str, Optional[Tensor], Tuple[Any, ...]]]:
        grouped: Dict[str, List[Any]] = {}
        descriptors: Dict[str, Optional[Tensor]] = {}
        for record in records:
            context, descriptor = self._record_context_descriptor(record)
            if not context:
                raise FDPSCIntegrationError("fixed evaluation record has no context identifier")
            if (
                descriptor is None
                and len(self.exception_router) > 0
                and context != str(self._active_context)
            ):
                raise FDPSCIntegrationError(
                    f"context {context!r} lacks the frozen descriptor required by production routing"
                )
            grouped.setdefault(context, []).append(record)
            previous = descriptors.get(context)
            if previous is None:
                descriptors[context] = descriptor
            elif descriptor is not None:
                if previous.shape != descriptor.shape or not torch.allclose(
                    previous,
                    descriptor,
                    atol=1.0e-6,
                    rtol=1.0e-5,
                ):
                    raise FDPSCIntegrationError(
                        f"context {context!r} has inconsistent frozen descriptors"
                    )
        return [
            (context, descriptors.get(context), tuple(grouped[context]))
            for context in sorted(grouped)
        ]

    def _evaluation_route(
        self,
        context: str,
        descriptor: Optional[Tensor],
        candidate: Optional[_Candidate],
    ) -> Tuple[str, Optional[str]]:
        if context == str(self._active_context):
            fixed_id = self._route.adapter_id if self._route is not None else None
            if candidate is not None and candidate.proposal_type in {
                ProposalType.NEW_EXCEPTION,
                ProposalType.REPLACE_EXCEPTION,
            }:
                return "proposal", fixed_id
            return "existing", fixed_id
        existing = self.exception_router.route(descriptor, production=False)
        if candidate is None or candidate.proposal_type == ProposalType.GLOBAL_SLOW:
            return "existing", existing.adapter_id
        if candidate.proposal_type == ProposalType.REPLACE_EXCEPTION:
            target = self._route.adapter_id if self._route is not None else None
            if target is not None and existing.adapter_id == target:
                return "proposal", target
            return "existing", existing.adapter_id

        # NEW_EXCEPTION participates in the clone's production nearest-prototype
        # decision without mutating bank usage or allocating an adapter ID.
        prototype = self._current_context_prototype()
        if prototype is None or descriptor is None or prototype.shape != descriptor.shape:
            return "existing", existing.adapter_id
        similarity = float(torch.dot(prototype, descriptor))
        if (
            math.isfinite(similarity)
            and similarity >= self.config.exception.minimum_route_similarity
            and (existing.similarity is None or similarity > existing.similarity)
        ):
            return "proposal", None
        return "existing", existing.adapter_id

    def _load_routed_evaluation_state(
        self,
        *,
        state: str,
        candidate: Optional[_Candidate],
        context: str,
        descriptor: Optional[Tensor],
        fast_state: Mapping[str, Any],
    ) -> None:
        if state == "fast":
            self._adapter_participant.load_state_dict(copy.deepcopy(fast_state))
        else:
            self._load_episode_start()

        if state == "theta0":
            for adapter in self.injection.adapters.values():
                adapter.disable_all_adapters()
            return
        if state not in {"before", "fast", "candidate"}:
            raise ValueError(f"unknown evaluation state {state!r}")

        if state == "candidate":
            if candidate is None:
                raise ValueError("candidate state requires a candidate")
            self._zero_episodic()
            if candidate.proposal_type == ProposalType.GLOBAL_SLOW:
                self._apply_candidate(candidate)
                _, adapter_id = self._evaluation_route(context, descriptor, candidate)
                self._apply_exception_state(adapter_id)
                return
            route_kind, adapter_id = self._evaluation_route(context, descriptor, candidate)
            if route_kind == "proposal":
                self._apply_candidate(candidate)
            else:
                self._apply_exception_state(adapter_id)
            return

        # Pbefore/Pfast use the immutable production bank route for each fixed
        # record context.  This changes only the cloned active exception slot.
        _, adapter_id = self._evaluation_route(context, descriptor, None)
        self._apply_exception_state(adapter_id)

    def _evaluate_by_context(
        self,
        trainer: Any,
        windows: Sequence[ReplayWindow],
        *,
        state: str,
        candidate: Optional[_Candidate] = None,
    ) -> Dict[str, float]:
        grouped: Dict[str, List[ReplayWindow]] = {}
        for window in windows:
            grouped.setdefault(window.context_identifier, []).append(window)
        return {
            context: self._evaluate_state(
                trainer,
                self._replay_records(grouped[context]),
                state=state,
                candidate=candidate,
            )
            for context in sorted(grouped)
        }

    def _emit_context_retention_metrics(
        self,
        before_by_context: Mapping[str, float],
        candidate_by_context: Mapping[str, float],
    ) -> None:
        """Emit loss-based forgetting/BWT against the pre-proposal memory.

        Positive backward transfer means the proposal lowers historical loss.
        Forgetting is the largest positive context regression.  These are
        online proposal diagnostics, not cross-run report-test measurements.
        """

        if not before_by_context and not candidate_by_context:
            for name in ("forgetting", "backward_transfer"):
                self.metrics.record_nullable(
                    name,
                    None,
                    status="not_applicable",
                    reason="historical_replay_is_empty",
                    episode_id=self._active_episode_id,
                    context_identifier=self._active_context,
                    tags={"reference_state": "before", "state": "candidate"},
                )
            return
        if set(before_by_context) != set(candidate_by_context):
            for name in ("forgetting", "backward_transfer"):
                self.metrics.record_nullable(
                    name,
                    None,
                    status="unavailable",
                    reason="historical_context_sets_differ",
                    episode_id=self._active_episode_id,
                    context_identifier=self._active_context,
                    tags={"reference_state": "before", "state": "candidate"},
                )
            return

        regressions: List[float] = []
        for context in sorted(before_by_context):
            before = float(before_by_context[context])
            candidate = float(candidate_by_context[context])
            self.metrics.record(
                "per_context_loss",
                before,
                episode_id=self._active_episode_id,
                context_identifier=context,
                tags={"split": "historical_replay", "state": "before"},
            )
            self.metrics.record(
                "per_context_loss",
                candidate,
                episode_id=self._active_episode_id,
                context_identifier=context,
                tags={"split": "historical_replay", "state": "candidate"},
            )
            regressions.append(candidate - before)
        forgetting = max(0.0, max(regressions, default=0.0))
        backward_transfer = -sum(regressions) / float(len(regressions))
        self.metrics.record_nullable(
            "forgetting",
            forgetting,
            status="available",
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={
                "definition": "max(0,max(candidate_loss-before_loss))",
                "context_count": len(regressions),
            },
        )
        self.metrics.record_nullable(
            "backward_transfer",
            backward_transfer,
            status="available",
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={
                "definition": "mean(before_loss-candidate_loss)",
                "context_count": len(regressions),
            },
        )

    def _collect_activations(
        self,
        trainer: Any,
        records: Sequence[Any],
        *,
        state: str = "fast",
        candidate: Optional[_Candidate] = None,
    ) -> Dict[str, Tensor]:
        captured: Dict[str, List[Tensor]] = {key: [] for key in self.injection.adapters}
        handles = []
        entries_by_path: Dict[str, List[ManifestEntry]] = {}
        for entry in self.target_manifest.entries:
            if entry.injected:
                entries_by_path.setdefault(entry.module_path, []).append(entry)

        for path, wrapper in sorted(self.injection.physical_modules.items()):
            entries = sorted(entries_by_path[path], key=lambda item: item.logical_layer_id)

            def hook(
                module: nn.Module,
                args: Tuple[Any, ...],
                path_entries: Sequence[ManifestEntry] = entries,
            ) -> None:
                if not args or not isinstance(args[0], Tensor):
                    return
                value = args[0].detach()
                base = getattr(module, "base_layer", None)
                if isinstance(base, nn.Linear):
                    matrix = value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
                    captured[path_entries[0].logical_layer_id].append(matrix)
                elif isinstance(base, nn.Conv2d):
                    matrices = conv2d_group_activation_matrices(base, value)
                    by_group = {int(entry.logical_group or 0): entry for entry in path_entries}
                    for group, matrix in enumerate(matrices):
                        captured[by_group[group].logical_layer_id].append(matrix)

            handles.append(wrapper.register_forward_pre_hook(hook))
        try:
            self._evaluate_state(
                trainer,
                records,
                state=state,
                candidate=candidate,
            )
        finally:
            for handle in handles:
                handle.remove()
        result: Dict[str, Tensor] = {}
        for logical_id, values in sorted(captured.items()):
            if not values:
                continue
            matrix = torch.cat(
                [value.to(device=self.injection.adapters[logical_id]._reference().device) for value in values],
                dim=0,
            )
            if matrix.shape[0] > self._ACTIVATION_ROW_LIMIT:
                # Evenly spaced deterministic rows avoid a prefix-only bias.
                indices = torch.linspace(
                    0,
                    matrix.shape[0] - 1,
                    self._ACTIVATION_ROW_LIMIT,
                    device=matrix.device,
                ).round().long()
                matrix = matrix.index_select(0, indices)
            if not torch.isfinite(matrix).all():
                raise FDPSCIntegrationError(
                    f"non-finite calibration activations for {logical_id}"
                )
            result[logical_id] = matrix
        missing = sorted(set(self.injection.adapters) - set(result))
        if missing:
            raise FDPSCIntegrationError(
                f"fixed evaluation did not traverse injected targets: {missing}"
            )
        return result

    def _proximal_key_layer_ids(self) -> Tuple[str, ...]:
        configured_tags = set(self.config.repair.proximal_layer_tags)
        selected: List[str] = []
        for entry in self.target_manifest.entries:
            if not entry.injected:
                continue
            tags = {entry.module_group, entry.role}
            if entry.attention_output:
                tags.add("attention_output")
            if entry.mlp_output:
                tags.add("mlp_output")
            if entry.final_projection:
                tags.add("final_projection")
            if tags & configured_tags:
                selected.append(entry.logical_layer_id)
        return tuple(sorted(set(selected)))

    def _bind_live_candidate_factors(self, candidate: _Candidate) -> None:
        """Bind repair Parameters as persistent buffers without detaching them.

        Candidate evaluation normally installs immutable buffer clones.  LPR
        needs the same numerical persistent state *and* an autograd path to the
        cumulative repair factors.  Directly replacing the registered buffer
        values inside ``_preserve_adapter_runtime`` provides that path; the
        context restores ordinary detached persistent buffers on exit.
        """

        if candidate.proposal_type == ProposalType.GLOBAL_SLOW:
            names = ("slow_B", "slow_A")
        else:
            names = ("exception_B", "exception_A")
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            factors = candidate.factors_by_layer[logical_id]
            if names[0] not in adapter._buffers or names[1] not in adapter._buffers:
                raise FDPSCIntegrationError(
                    f"candidate factor buffers missing for {logical_id}"
                )
            adapter._buffers[names[0]] = factors.B
            adapter._buffers[names[1]] = factors.A

    def _collect_key_layer_outputs(
        self,
        trainer: Any,
        records: Sequence[Any],
        *,
        logical_ids: Sequence[str],
        state: str,
        candidate: Optional[_Candidate] = None,
        differentiable_candidate: bool = False,
    ) -> Dict[str, Tensor]:
        """Collect actual key-layer outputs for LPR on fixed old records."""

        if not records:
            raise FDPSCIntegrationError("LPR output collection requires old replay")
        requested = set(logical_ids)
        if not requested:
            raise FDPSCIntegrationError("LPR matched no injected key layer")
        entries_by_path: Dict[str, List[ManifestEntry]] = {}
        for entry in self.target_manifest.entries:
            if entry.injected and entry.logical_layer_id in requested:
                entries_by_path.setdefault(entry.module_path, []).append(entry)
        captured: Dict[str, List[Tensor]] = {key: [] for key in sorted(requested)}
        handles = []
        for path, entries_value in sorted(entries_by_path.items()):
            wrapper = self.injection.physical_modules[path]
            entries = tuple(
                sorted(entries_value, key=lambda item: item.logical_layer_id)
            )

            def hook(
                module: nn.Module,
                args: Tuple[Any, ...],
                output: Any,
                path_entries: Sequence[ManifestEntry] = entries,
            ) -> None:
                if not isinstance(output, Tensor):
                    raise FDPSCIntegrationError(
                        "LPR key-layer hook requires a Tensor output"
                    )
                base = getattr(module, "base_layer", None)
                if isinstance(base, nn.Linear):
                    value = output.reshape(-1, output.shape[-1])
                    if not differentiable_candidate:
                        value = value.detach()
                    captured[path_entries[0].logical_layer_id].append(value)
                    return
                if isinstance(base, nn.Conv2d):
                    width = int(base.out_channels // base.groups)
                    for entry in path_entries:
                        group = int(entry.logical_group or 0)
                        value = output[:, group * width : (group + 1) * width]
                        value = value.movedim(1, -1).reshape(-1, width)
                        if not differentiable_candidate:
                            value = value.detach()
                        captured[entry.logical_layer_id].append(value)
                    return
                raise FDPSCIntegrationError(
                    f"unsupported LPR key-layer wrapper at {path}"
                )

            handles.append(wrapper.register_forward_hook(hook))

        try:
            with self._preserve_adapter_runtime():
                fast_state = self._adapter_participant.state_dict()
                grad_context = (
                    contextlib.nullcontext()
                    if differentiable_candidate
                    else torch.no_grad()
                )
                with grad_context:
                    for context, descriptor, context_records in self._group_records_for_routing(
                        records
                    ):
                        self._load_routed_evaluation_state(
                            state=state,
                            candidate=candidate,
                            context=context,
                            descriptor=descriptor,
                            fast_state=fast_state,
                        )
                        if differentiable_candidate:
                            if state != "candidate" or candidate is None:
                                raise FDPSCIntegrationError(
                                    "differentiable LPR outputs require candidate state"
                                )
                            uses_proposal = candidate.proposal_type == ProposalType.GLOBAL_SLOW
                            if not uses_proposal:
                                route_kind, _ = self._evaluation_route(
                                    context,
                                    descriptor,
                                    candidate,
                                )
                                uses_proposal = route_kind == "proposal"
                            if uses_proposal:
                                self._bind_live_candidate_factors(candidate)
                        for parameter in self.wm.parameters():
                            parameter.requires_grad_(False)
                        self.wm.eval()
                        self.injection.enforce_frozen_base_eval()
                        trainer.external_loss_tensor(context_records)
        finally:
            for handle in handles:
                handle.remove()

        result: Dict[str, Tensor] = {}
        for logical_id, values in sorted(captured.items()):
            if not values:
                raise FDPSCIntegrationError(
                    f"LPR key layer was not traversed: {logical_id}"
                )
            result[logical_id] = torch.cat(values, dim=0)
            if not torch.isfinite(result[logical_id]).all():
                raise FDPSCIntegrationError(
                    f"non-finite LPR key-layer output: {logical_id}"
                )
        return result

    def _collect_effective_gradients(
        self,
        trainer: Any,
        records: Sequence[Any],
        *,
        state: str = "fast",
        candidate: Optional[_Candidate] = None,
        include_loss: bool = False,
        fixed_batch_size: Optional[int] = None,
    ) -> Any:
        collection_started = time.perf_counter()
        if not records:
            raise FDPSCIntegrationError("cannot collect gradients from an empty split")
        with self._preserve_adapter_runtime():
            if state not in {"before", "fast", "candidate"}:
                raise ValueError(f"unsupported gradient state {state!r}")
            fast_state = self._adapter_participant.state_dict()
            hooks = EffectiveWeightGradientHooks(self.injection.gradient_modules())
            try:
                total = len(records)
                loss_value = 0.0
                if fixed_batch_size is not None and int(fixed_batch_size) <= 0:
                    raise ValueError("fixed_batch_size must be positive")
                size = total if fixed_batch_size is None else int(fixed_batch_size)
                record_batches = [tuple(records[index : index + size]) for index in range(0, total, size)]
                if fixed_batch_size is not None and any(
                    len(batch) != size for batch in record_batches
                ):
                    raise FDPSCIntegrationError(
                        "fixed gradient records do not form complete configured batches"
                    )
                for record_batch in record_batches:
                    for context, descriptor, context_records in self._group_records_for_routing(
                        record_batch
                    ):
                        self._load_routed_evaluation_state(
                            state=state,
                            candidate=candidate,
                            context=context,
                            descriptor=descriptor,
                            fast_state=fast_state,
                        )
                        for parameter in self.wm.parameters():
                            parameter.requires_grad_(False)
                        # Persistent candidate factors are buffers.  A zero episodic
                        # branch is retained solely as a differentiable carrier for
                        # the exact effective-weight hook; it contributes no value.
                        for adapter in self.injection.adapters.values():
                            if adapter.centered_active and adapter.center_A is not None:
                                adapter.center_A.requires_grad_(True)
                                adapter.center_B.requires_grad_(True)
                            elif adapter.pilot_A is not None:
                                adapter.pilot_A.requires_grad_(True)
                                adapter.pilot_B.requires_grad_(True)
                        self.wm.eval()
                        self.injection.enforce_frozen_base_eval()
                        group_loss = trainer.external_loss_tensor(context_records)
                        weight = len(context_records) / float(total)
                        (group_loss * weight).backward()
                        loss_value += weight * float(group_loss.detach().cpu())
                result = {
                    key: value.detach().clone() for key, value in hooks.gradients.items()
                }
            finally:
                hooks.close()
                for parameter in self.wm.parameters():
                    parameter.grad = None
                    parameter.requires_grad_(False)
            missing = sorted(set(self.injection.adapters) - set(result))
            if missing:
                raise FDPSCIntegrationError(
                    f"effective-gradient hooks missed logical targets: {missing}"
                )
            self.metrics.record(
                "gradient_collection_latency_s",
                time.perf_counter() - collection_started,
                episode_id=self._active_episode_id,
                replan_index=self._replan_index,
                tags={"state": state, "record_count": len(records)},
            )
            if include_loss:
                return result, loss_value
            return result

    # ------------------------------------------------------------------
    # Calibration-only proposal generation and repair
    # ------------------------------------------------------------------
    def _merge_signal_has_threshold(self, signal: str) -> bool:
        cfg = self.config.merge
        if signal == "gradient":
            return bool(cfg.use_gradient_similarity) and (
                cfg.gradient_conflict_threshold is not None
                or cfg.gradient_match_threshold is not None
            )
        if signal == "context":
            return bool(cfg.use_context_similarity) and (
                cfg.context_conflict_threshold is not None
                or cfg.context_match_threshold is not None
            )
        if signal == "residual":
            return bool(cfg.use_residual_similarity) and (
                cfg.residual_match_threshold is not None
            )
        raise ValueError(f"unknown merge-pruning signal {signal!r}")

    @staticmethod
    def _bounded_similarity(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(float(value)):
            return None
        number = float(value)
        # Dot products of normalized float32 descriptors can leave [-1, 1]
        # by a few ulps.  Larger excursions indicate an invalid signal.
        if number < -1.000001 or number > 1.000001:
            return None
        return min(1.0, max(-1.0, number))

    @staticmethod
    def _unit_descriptor(value: Any, expected_dim: Optional[int] = None) -> Optional[Tensor]:
        try:
            vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten()
        except (TypeError, ValueError, RuntimeError):
            return None
        if (
            vector.numel() == 0
            or (expected_dim is not None and vector.numel() != int(expected_dim))
            or not torch.isfinite(vector).all()
        ):
            return None
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or float(norm) <= 1.0e-12:
            return None
        return vector / norm

    @classmethod
    def _nearest_context_similarity(
        cls,
        current_descriptor: Any,
        history_windows: Sequence[ReplayWindow],
    ) -> Optional[float]:
        current = cls._unit_descriptor(current_descriptor)
        if current is None or not history_windows:
            return None
        similarities: List[float] = []
        for window in sorted(history_windows, key=lambda item: item.window_id):
            history = cls._unit_descriptor(
                window.context_embedding,
                expected_dim=current.numel(),
            )
            if history is None:
                # A configured but partially unavailable signal is not allowed
                # to prune from a selectively observed subset.
                return None
            similarities.append(float(torch.dot(current, history).item()))
        return cls._bounded_similarity(max(similarities))

    @classmethod
    def _nearest_residual_similarity(
        cls,
        current_windows: Sequence[ReplayWindow],
        history_windows: Sequence[ReplayWindow],
    ) -> Optional[float]:
        if not current_windows or not history_windows:
            return None

        def audited_descriptor(window: ReplayWindow, expected_dim: Optional[int]) -> Optional[Tensor]:
            metadata = dict(window.metadata or {})
            if (
                metadata.get("theta0_residual") is not True
                or metadata.get("residual_descriptor_schema")
                != "theta0_jepa_pattern_v1"
            ):
                return None
            return cls._unit_descriptor(window.residual, expected_dim=expected_dim)

        ordered_current = sorted(current_windows, key=lambda item: item.window_id)
        raw_current: List[Tensor] = []
        dimension: Optional[int] = None
        for window in ordered_current:
            # Average raw residual-pattern descriptors exactly as the exception
            # prototype protocol does, then normalize once.
            metadata = dict(window.metadata or {})
            if (
                metadata.get("theta0_residual") is not True
                or metadata.get("residual_descriptor_schema")
                != "theta0_jepa_pattern_v1"
            ):
                return None
            try:
                vector = torch.as_tensor(
                    window.residual,
                    dtype=torch.float32,
                ).detach().cpu().flatten()
            except (TypeError, ValueError, RuntimeError):
                return None
            if (
                vector.numel() == 0
                or (dimension is not None and vector.numel() != dimension)
                or not torch.isfinite(vector).all()
            ):
                return None
            dimension = int(vector.numel())
            raw_current.append(vector)
        current = cls._unit_descriptor(torch.stack(raw_current).mean(0), dimension)
        if current is None:
            # Near-zero theta_0 residuals have an explicit unavailable state;
            # they cannot be treated as either a match or a conflict.
            return None

        similarities: List[float] = []
        for window in sorted(history_windows, key=lambda item: item.window_id):
            history = audited_descriptor(window, current.numel())
            if history is None:
                return None
            similarities.append(float(torch.dot(current, history).item()))
        return cls._bounded_similarity(max(similarities))

    def _merge_similarity_signals(
        self,
        *,
        current_gradients: Mapping[str, Tensor],
        history_gradients: Mapping[str, Tensor],
        current_windows: Sequence[ReplayWindow],
        history_windows: Sequence[ReplayWindow],
    ) -> _MergeSimilaritySignals:
        gradient: Optional[float] = None
        if self._merge_signal_has_threshold("gradient"):
            result = global_weighted_cosine(
                current_gradients,
                history_gradients,
                weighting=self.config.gradient_geometry.global_cosine_weighting,
                epsilon=self.config.gradient_geometry.epsilon,
            )
            if result.available:
                gradient = self._bounded_similarity(result.value)

        context: Optional[float] = None
        if self._merge_signal_has_threshold("context"):
            context = self._nearest_context_similarity(
                self._current_context_prototype(),
                history_windows,
            )

        residual: Optional[float] = None
        if self._merge_signal_has_threshold("residual"):
            residual = self._nearest_residual_similarity(
                current_windows,
                history_windows,
            )
        return _MergeSimilaritySignals(
            gradient=gradient,
            context=context,
            residual=residual,
        )

    def _pruned_coefficient_grid(
        self,
        signals: _MergeSimilaritySignals,
    ) -> _CoefficientGridDecision:
        """Apply V2 §14.1 conservatively and deterministically.

        Null thresholds are absent controls.  Conversely, if a configured
        signal cannot be computed, or available signals disagree between
        MATCH and CONFLICT, the complete configured grid is retained.
        """

        cfg = self.config.merge
        full = tuple(float(value) for value in cfg.shared_coefficients)
        specifications = (
            (
                "gradient",
                bool(cfg.use_gradient_similarity),
                cfg.gradient_conflict_threshold,
                cfg.gradient_match_threshold,
                signals.gradient,
            ),
            (
                "context",
                bool(cfg.use_context_similarity),
                cfg.context_conflict_threshold,
                cfg.context_match_threshold,
                signals.context,
            ),
            (
                "residual",
                bool(cfg.use_residual_similarity),
                None,
                cfg.residual_match_threshold,
                signals.residual,
            ),
        )
        decisions: Dict[str, str] = {}
        for name, enabled, conflict_threshold, match_threshold, raw_value in specifications:
            if not enabled or (conflict_threshold is None and match_threshold is None):
                continue
            value = self._bounded_similarity(raw_value)
            if value is None:
                decisions[name] = "missing"
                return _CoefficientGridDecision(
                    full,
                    f"missing_signal:{name}",
                    dict(sorted(decisions.items())),
                )
            if conflict_threshold is not None and value <= float(conflict_threshold):
                decisions[name] = "conflict"
            elif match_threshold is not None and value >= float(match_threshold):
                decisions[name] = "match"
            else:
                decisions[name] = "neutral"

        kinds = set(decisions.values())
        if "match" in kinds and "conflict" in kinds:
            return _CoefficientGridDecision(
                full,
                "contradictory_signals",
                dict(sorted(decisions.items())),
            )
        allowed: Optional[Tuple[float, ...]] = None
        reason = "no_decisive_signal"
        if "conflict" in kinds:
            allowed = (0.0, 0.1, 0.25)
            reason = "conflict_pruned"
        elif "match" in kinds:
            allowed = (0.25, 0.5, 0.75, 1.0)
            reason = "match_pruned"
        if allowed is None:
            return _CoefficientGridDecision(
                full,
                reason,
                dict(sorted(decisions.items())),
            )
        pruned = tuple(
            value
            for value in full
            if any(math.isclose(value, target, rel_tol=0.0, abs_tol=1.0e-12) for target in allowed)
        )
        if not pruned:
            return _CoefficientGridDecision(
                full,
                "pruning_would_empty_grid",
                dict(sorted(decisions.items())),
            )
        return _CoefficientGridDecision(
            pruned,
            reason,
            dict(sorted(decisions.items())),
        )

    def _task_variants(
        self,
        calibration_gradients: Mapping[str, Tensor],
    ) -> List[Tuple[str, Dict[str, LowRankFactors]]]:
        raw = {
            logical_id: self._canonical(adapter.get_episodic_factors())
            for logical_id, adapter in sorted(self.injection.adapters.items())
        }
        variants: List[Tuple[str, Dict[str, LowRankFactors]]] = [("none", raw)]
        if not self.config.spectral_surgery.enabled:
            return variants
        operated: Dict[str, LowRankFactors] = {}
        applied = False
        for logical_id, factors in raw.items():
            entry = self._entry_by_id[logical_id]
            eligible = (
                not self.config.spectral_surgery.output_writing_layers_only
                or entry.attention_output
                or entry.mlp_output
                or entry.final_projection
            )
            if eligible and factors.rank and logical_id in calibration_gradients:
                result = spectral_surgery(
                    factors,
                    calibration_gradients[logical_id],
                    steps=self.config.spectral_surgery.steps,
                    learning_rate=self.config.spectral_surgery.learning_rate,
                    minimum_scale=self.config.spectral_surgery.minimum_scale,
                    maximum_scale=self.config.spectral_surgery.maximum_scale,
                    preserve_spectral_l2_norm=self.config.spectral_surgery.preserve_spectral_l2_norm,
                    epsilon=self.config.gradient_geometry.epsilon,
                )
                operated[logical_id] = result.factors
                applied = applied or result.applied
            else:
                operated[logical_id] = factors
        if applied:
            variants.append(("spectral_surgery", operated))
        return variants

    def _proposal_base_factors(self, proposal_type: ProposalType) -> Dict[str, LowRankFactors]:
        result = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            if proposal_type == ProposalType.GLOBAL_SLOW:
                result[logical_id] = self._canonical(adapter.get_slow_factors())
            else:
                result[logical_id] = self._canonical(adapter.get_exception_factors())
        return result

    def _make_candidates(
        self,
        proposal_type: ProposalType,
        activations: Mapping[str, Tensor],
        task_variants: Sequence[Tuple[str, Mapping[str, LowRankFactors]]],
        *,
        shared_coefficients: Optional[Sequence[float]] = None,
    ) -> List[_Candidate]:
        candidates: List[_Candidate] = []
        base = self._proposal_base_factors(proposal_type)
        if (
            self.config.run_mode in {"plain_svd", "accumulate"}
            or not self.config.merge.soft_ness_enabled
        ):
            coefficient_grid = [(1.0, 1.0)]
        else:
            shared_grid = (
                self.config.merge.shared_coefficients
                if shared_coefficients is None
                else shared_coefficients
            )
            coefficient_grid = [
                (float(shared), float(safe))
                for shared in shared_grid
                for safe in self.config.merge.safe_coefficients
            ]
        maximum_rank = (
            self.config.slow_lora.maximum_rank
            if proposal_type == ProposalType.GLOBAL_SLOW
            else self.config.exception.maximum_rank
        )
        # ``select_rank`` deterministically maps every configured rank through
        # min(rank, maximum_rank, dout, din).  Configuration ranks are bounded
        # by maximum_rank; the remaining clipping is strictly per-layer dmax.
        allowed = list(self.config.slow_lora.allowed_ranks)
        for spectral_name, task_by_layer in task_variants:
            for alpha_shared, alpha_safe in coefficient_grid:
                factors_by_layer: Dict[str, LowRankFactors] = {}
                committed_task: Dict[str, LowRankFactors] = {}
                errors: Dict[str, float] = {}
                ranks: Dict[str, int] = {}
                references: Dict[str, LowRankFactors] = {}
                feasible = True
                for logical_id, task in sorted(task_by_layer.items()):
                    transformed = task
                    if self.config.merge.soft_ness_enabled and self.config.activation_subspace.enabled:
                        subspace = self.subspaces.get(
                            logical_id,
                            task.in_features,
                            task.B.device,
                        )
                        weights = subspace.soft_ness_weights(
                            mode=self.config.activation_subspace.soft_ness_tau_mode,
                            fixed_tau=self.config.activation_subspace.soft_ness_tau_fixed,
                            quantile=self.config.activation_subspace.soft_ness_tau_quantile,
                            minimum_energy=self.config.activation_subspace.minimum_energy,
                        ).weights
                        transformed = apply_soft_ness(
                            task,
                            subspace,
                            alpha_shared,
                            alpha_safe,
                            weights=weights,
                        )
                    # Disabling soft-NESS is the literal full-task ablation:
                    # no coefficient grid and no hidden scalar attenuation.
                    reference = concatenate_factors((base[logical_id], transformed))
                    references[logical_id] = LowRankFactors(
                        reference.B.detach().clone(),
                        reference.A.detach().clone(),
                    )
                    selection = select_rank(
                        reference,
                        activations[logical_id],
                        allowed_ranks=allowed,
                        maximum_rank=maximum_rank,
                        spectral_energy_threshold=self.config.slow_lora.spectral_energy_threshold,
                        # Pfast H is only a deterministic provisional state.
                        # It may choose a starting rank but must never discard
                        # a candidate that is feasible under that candidate's
                        # own persistent-state H.  The real functional gate is
                        # enforced by ``_stabilize_candidate_rank``.
                        functional_error_threshold=float("inf"),
                        epsilon=self.config.gradient_geometry.epsilon,
                        absolute_tolerance=self.config.gates.absolute_numerical_tolerance,
                        output_dtype=base[logical_id].B.dtype,
                    )
                    if not selection.feasible or selection.factors is None or selection.rank is None:
                        if (
                            selection.reason
                            != "rank_cap_failed_spectral_or_functional_threshold"
                            or not selection.diagnostics
                        ):
                            feasible = False
                            break
                        # A saturated rank cap is not terminal.  Section 16.3
                        # requires this state to seed bounded repair, whose JEPA
                        # updates may make the capped representation feasible.
                        # Keep the largest legal capped approximation so Path A
                        # can reject it normally and Path B still has a complete
                        # all-layer candidate to optimize.  Commit-query data is
                        # not involved in either construction or repair.
                        fallback = selection.diagnostics[-1]
                        factors_by_layer[logical_id] = truncate_factors(
                            reference,
                            fallback.rank,
                            dtype=base[logical_id].B.dtype,
                        )
                        ranks[logical_id] = int(fallback.rank)
                        errors[logical_id] = float(
                            fallback.functional_error.relative_error
                        )
                    else:
                        factors_by_layer[logical_id] = selection.factors
                        ranks[logical_id] = selection.rank
                        if selection.rank == 0 and not selection.diagnostics:
                            # Canonical-zero layers are represented by empty
                            # factors and have exactly zero compression error.
                            errors[logical_id] = 0.0
                        else:
                            diagnostic = next(
                                item
                                for item in selection.diagnostics
                                if item.rank == selection.rank
                            )
                            errors[logical_id] = (
                                diagnostic.functional_error.relative_error
                            )
                    committed_task[logical_id] = transformed
                if feasible:
                    candidates.append(
                        _Candidate(
                            proposal_type=proposal_type,
                            factors_by_layer=factors_by_layer,
                            task_factors_by_layer=committed_task,
                            functional_error_by_layer=errors,
                            selected_rank_by_layer=ranks,
                            alpha_shared=alpha_shared,
                            alpha_safe=alpha_safe,
                            spectral_variant=spectral_name,
                            rank_reference_by_layer=references,
                        )
                    )
        return candidates

    @staticmethod
    def _factors_bitwise_equal(
        left: Mapping[str, LowRankFactors],
        right: Mapping[str, LowRankFactors],
    ) -> bool:
        if set(left) != set(right):
            return False
        return all(
            torch.equal(left[key].B.detach(), right[key].B.detach())
            and torch.equal(left[key].A.detach(), right[key].A.detach())
            for key in sorted(left)
        )

    def _stabilize_candidate_rank(
        self,
        trainer: Any,
        candidate: _Candidate,
        calibration: Sequence[Any],
    ) -> bool:
        """Solve the candidate-state activation/rank fixed point.

        ``H_l`` must come from the *candidate persistent state*, while the
        compression target remains the immutable untruncated factor-space
        merge.  Rank changes can themselves change downstream activations, so
        a candidate is screenable only after both ranks and factors stabilize.
        Cycles or a bounded non-convergence are explicit candidate rejection,
        never an excuse to reuse Pfast activations.
        """

        references = candidate.rank_reference_by_layer
        if not references:
            references = {
                logical_id: LowRankFactors(
                    factors.B.detach().clone(),
                    factors.A.detach().clone(),
                )
                for logical_id, factors in sorted(candidate.factors_by_layer.items())
            }
            candidate.rank_reference_by_layer = references
        expected = set(self.injection.adapters)
        if set(references) != expected or set(candidate.factors_by_layer) != expected:
            candidate.screening_reason = "candidate_rank_reference_mismatch"
            return False

        maximum_rank = (
            self.config.slow_lora.maximum_rank
            if candidate.proposal_type == ProposalType.GLOBAL_SLOW
            else self.config.exception.maximum_rank
        )
        allowed = [int(rank) for rank in self.config.slow_lora.allowed_ranks]
        # A deterministic finite-state iteration normally converges in one or
        # two passes.  The bound prevents pathological calibration-dependent
        # oscillations from making sleep unbounded; repeated factor states are
        # rejected immediately as a cycle.
        maximum_iterations = 32
        seen_states = set()
        for iteration in range(maximum_iterations):
            digest = hashlib.sha256()
            for key, factors in sorted(candidate.factors_by_layer.items()):
                digest.update(key.encode("utf-8"))
                for tensor in (factors.B, factors.A):
                    digest.update(str(tuple(tensor.shape)).encode("ascii"))
                    digest.update(str(tensor.dtype).encode("ascii"))
                    digest.update(_tensor_bytes(tensor))
            current_state = digest.hexdigest()
            if current_state in seen_states:
                candidate.screening_reason = "candidate_rank_activation_cycle"
                self.diagnostics.fallback(
                    "candidate_rank_activation_cycle",
                    f"iteration={iteration}",
                    episode_id=self._active_episode_id,
                )
                return False
            seen_states.add(current_state)

            activations = self._collect_activations(
                trainer,
                calibration,
                state="candidate",
                candidate=candidate,
            )
            compressed: Dict[str, LowRankFactors] = {}
            ranks: Dict[str, int] = {}
            errors: Dict[str, float] = {}
            for logical_id, reference in sorted(references.items()):
                selection = select_rank(
                    reference,
                    activations[logical_id],
                    allowed_ranks=allowed,
                    maximum_rank=maximum_rank,
                    spectral_energy_threshold=self.config.slow_lora.spectral_energy_threshold,
                    functional_error_threshold=self.config.slow_lora.functional_error_threshold,
                    epsilon=self.config.gradient_geometry.epsilon,
                    absolute_tolerance=self.config.gates.absolute_numerical_tolerance,
                    output_dtype=candidate.factors_by_layer[logical_id].B.dtype,
                )
                if (
                    not selection.feasible
                    or selection.factors is None
                    or selection.rank is None
                ):
                    candidate.screening_reason = (
                        f"candidate_rank_selection_failed:{logical_id}:{selection.reason}"
                    )
                    return False
                compressed[logical_id] = selection.factors
                ranks[logical_id] = int(selection.rank)
                if selection.rank == 0 and not selection.diagnostics:
                    errors[logical_id] = 0.0
                else:
                    diagnostic = next(
                        item
                        for item in selection.diagnostics
                        if item.rank == selection.rank
                    )
                    errors[logical_id] = float(
                        diagnostic.functional_error.relative_error
                    )

            stable = self._factors_bitwise_equal(
                candidate.factors_by_layer,
                compressed,
            )
            candidate.factors_by_layer = compressed
            candidate.selected_rank_by_layer = ranks
            candidate.functional_error_by_layer = errors
            if stable:
                self.metrics.record(
                    "candidate_rank_fixed_point_iterations",
                    iteration + 1,
                    episode_id=self._active_episode_id,
                    tags={
                        "proposal_type": candidate.proposal_type.value,
                        "spectral_variant": candidate.spectral_variant,
                    },
                )
                return True

        candidate.screening_reason = "candidate_rank_activation_nonconvergent"
        self.diagnostics.fallback(
            "candidate_rank_activation_nonconvergent",
            f"iterations={maximum_iterations}",
            episode_id=self._active_episode_id,
        )
        return False

    def _plasticity_gain(
        self,
        trainer: Any,
        *,
        state: str,
        candidate: Optional[_Candidate] = None,
    ) -> Optional[float]:
        if not self.config.gates.plasticity_enabled:
            return None
        assert self.external is not None
        support = self.external.plasticity_support(self._active_context)
        query = self.external.plasticity_query(self._active_context)
        with self._preserve_adapter_runtime():
            # Gate-4 is a paired clone experiment.  Capture the first probe's
            # complete process RNG once, then rewind every before/candidate
            # clone to that exact state.  The surrounding context restores the
            # caller's RNG afterwards, so neither probe order nor the number
            # of screened candidates can alter live stochastic state.
            if self._plasticity_probe_rng is None:
                self._plasticity_probe_rng = RNGSnapshot.capture()
            self._plasticity_probe_rng.restore()
            if state == "before":
                self._load_episode_start()
            elif state == "candidate":
                if candidate is None:
                    raise ValueError("candidate plasticity requires a candidate")
                self._load_episode_start()
                self._apply_candidate(candidate)
            elif state != "fast":
                raise ValueError(f"unsupported plasticity state {state!r}")

            # A fresh zero-function Pilot is adapted on the fixed plasticity
            # support using the original optimizer class, LR split, number of
            # steps, and JEPA loss. The whole clone (including RNG) is restored
            # on exit, so candidate screening cannot change live memory.
            # This phase must be identical for before and *every* candidate;
            # state/alpha-dependent seeds would compare different Pilots.
            phase = "plasticity-probe-paired"
            for logical_id, adapter in sorted(self.injection.adapters.items()):
                adapter.begin_episode(
                    generator=self._episode_generator(logical_id, phase),
                    clear_exception=False,
                )
            for parameter in self.wm.parameters():
                parameter.requires_grad_(False)
            for adapter in self.injection.adapters.values():
                if adapter.pilot_A is not None:
                    adapter.pilot_A.requires_grad_(True)
                    adapter.pilot_B.requires_grad_(True)
            predictor = self.injection.predictor_parameters()
            encoder = self.injection.encoder_parameters() if trainer.finetune_encoder else []
            if not predictor:
                raise FDPSCIntegrationError("plasticity clone has no predictor Pilot parameters")
            groups = [{"params": predictor, "lr": trainer.lr}]
            if encoder:
                groups.append({"params": encoder, "lr": trainer.encoder_lr})
            optimizer_type = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
            }.get(trainer.optimizer_name)
            if optimizer_type is None:
                raise FDPSCIntegrationError(
                    f"unsupported plasticity optimizer {trainer.optimizer_name!r}"
                )
            optimizer = optimizer_type(groups)
            plasticity_hooks: Optional[EffectiveWeightGradientHooks] = None
            saved_sdc_active = dict(self._sdc_active)
            always_on_sdc = self.config.sdc.enabled and not self.config.sdc.event_triggered
            try:
                if always_on_sdc:
                    if self._online_hooks is not None:
                        raise FDPSCIntegrationError(
                            "plasticity probe cannot borrow SDC hooks during an online update"
                        )
                    # The always-on ablation applies from the very first clone
                    # step. Reuse the exact online two-pass helper and effective
                    # weight hooks rather than approximating factor gradients.
                    self._sdc_active = {
                        logical_id: True for logical_id in self.injection.adapters
                    }
                    plasticity_hooks = EffectiveWeightGradientHooks(
                        self.injection.gradient_modules()
                    )
                    self._online_hooks = plasticity_hooks

                self.wm.eval()
                query_before = trainer.evaluate_external_records(query)
                self.wm.eval()
                self.wm.predictor.train()
                if trainer.finetune_encoder:
                    self.wm.encoder.train()
                self.injection.enforce_frozen_base_eval()
                for _ in range(trainer.steps):
                    optimizer.zero_grad(set_to_none=True)

                    def loss_closure() -> Tensor:
                        return trainer.external_loss_tensor(support)

                    forward_rng = self.capture_update_rng()
                    loss = loss_closure()
                    if always_on_sdc:
                        self.backward_with_sdc(
                            loss,
                            optimizer,
                            loss_closure=loss_closure,
                            forward_rng=forward_rng,
                        )
                    else:
                        loss.backward()
                    optimizer.step()
                    if plasticity_hooks is not None:
                        plasticity_hooks.reset()
                self.wm.eval()
                query_after = trainer.evaluate_external_records(query)
                self.assert_base_frozen()
                return query_before - query_after
            finally:
                optimizer.zero_grad(set_to_none=True)
                if plasticity_hooks is not None:
                    plasticity_hooks.close()
                    self._online_hooks = None
                self._sdc_active = saved_sdc_active

    def _plasticity_screening_feasible(
        self,
        before: float,
        candidate: float,
    ) -> bool:
        """Mirror Gate 4 exactly while commit-query is still inaccessible."""

        before_value = float(before)
        candidate_value = float(candidate)
        if not (math.isfinite(before_value) and math.isfinite(candidate_value)):
            return False
        absolute = float(self.config.gates.absolute_numerical_tolerance)
        relative = float(self.config.gates.relative_numerical_tolerance)
        if before_value > absolute:
            required = float(self.config.gates.plasticity_retention) * before_value
            return bool(
                candidate_value + absolute >= required
                or math.isclose(
                    candidate_value,
                    required,
                    abs_tol=absolute,
                    rel_tol=relative,
                )
            )
        # Gate 4's explicit absolute fallback for a non-positive or near-zero
        # baseline.  A less-negative candidate is not automatically valid: it
        # must also be non-negative within tolerance and preserve the baseline
        # in the gate's stated absolute sense.
        if candidate_value < -absolute:
            return False
        return bool(
            before_value - candidate_value <= absolute
            or math.isclose(
                before_value,
                candidate_value,
                abs_tol=absolute,
                rel_tol=relative,
            )
        )

    def _emit_plasticity_gate_ratio(
        self,
        before_gain: Optional[float],
        candidate_gain: Optional[float],
    ) -> None:
        common = {
            "episode_id": self._active_episode_id,
            "context_identifier": self._active_context,
        }
        if not self.config.gates.plasticity_enabled:
            self.metrics.record_nullable(
                "plasticity_gate_ratio",
                None,
                status="not_applicable",
                reason="plasticity_gate_disabled",
                **common,
            )
            return
        if before_gain is None or candidate_gain is None:
            self.metrics.record_nullable(
                "plasticity_gate_ratio",
                None,
                status="unavailable",
                reason="plasticity_gain_missing",
                **common,
            )
            return
        denominator = float(before_gain)
        candidate = float(candidate_gain)
        if not (math.isfinite(denominator) and math.isfinite(candidate)):
            self.metrics.record_nullable(
                "plasticity_gate_ratio",
                None,
                status="unavailable",
                reason="plasticity_gain_non_finite",
                **common,
            )
            return
        if denominator <= self.config.gates.absolute_numerical_tolerance:
            self.metrics.record_nullable(
                "plasticity_gate_ratio",
                None,
                status="not_applicable",
                reason="gate_uses_absolute_fallback_for_nonpositive_baseline",
                tags={
                    "before_gain": denominator,
                    "candidate_gain": candidate,
                },
                **common,
            )
            return
        self.metrics.record_nullable(
            "plasticity_gate_ratio",
            candidate / denominator,
            status="available",
            tags={
                "before_gain": denominator,
                "candidate_gain": candidate,
                "definition": "candidate_gain/before_gain",
            },
            **common,
        )

    def _screen_candidates(
        self,
        trainer: Any,
        candidates: Sequence[_Candidate],
        *,
        calibration: Sequence[Any],
        calibration_before: float,
        calibration_fast: float,
        history_windows: Sequence[ReplayWindow],
        history_before: Optional[float],
        anchor: Sequence[Any],
        anchor_before: Optional[float],
        plasticity_before: Optional[float],
    ) -> Tuple[Optional[_Candidate], List[Dict[str, Any]]]:
        fast_gain = calibration_before - calibration_fast
        if fast_gain <= self.config.gates.absolute_numerical_tolerance:
            for candidate in candidates:
                candidate.screening_reason = "calibration_fast_gain_not_positive"
            return None, [candidate.summary() for candidate in candidates]

        if (
            self.config.gates.history_enabled
            and self._successful_slow_commit_count > 0
            and not history_windows
        ):
            # After the first successful slow commit Gate 2 is mandatory.  An
            # empty/capacity-zero/corrupt replay makes its metrics unavailable;
            # reject before selecting a proposal so the one-shot query is not
            # wasted on a final gate that is known to fail.
            for candidate in candidates:
                candidate.screening_reason = (
                    "historical_replay_required_after_slow_commit"
                )
            return None, [candidate.summary() for candidate in candidates]

        # Spectral surgery is only a calibration-derived proposal.  Compare
        # the untruncated operated merge with the matching untruncated
        # original before rank feasibility is considered, so a rank-infeasible
        # original cannot accidentally authorize a calibration-worse surgery.
        spectral_rejections = set()
        if self.config.spectral_surgery.enabled:
            original_losses: Dict[Tuple[str, float, float], float] = {}

            def untruncated_loss(value: _Candidate) -> float:
                untruncated = copy.deepcopy(value)
                untruncated.factors_by_layer = {
                    key: LowRankFactors(
                        factors.B.detach().clone(),
                        factors.A.detach().clone(),
                    )
                    for key, factors in sorted(value.rank_reference_by_layer.items())
                }
                untruncated.selected_rank_by_layer = {
                    key: factors.rank
                    for key, factors in sorted(untruncated.factors_by_layer.items())
                }
                untruncated.functional_error_by_layer = {
                    key: 0.0 for key in untruncated.factors_by_layer
                }
                return self._evaluate_state(
                    trainer,
                    calibration,
                    state="candidate",
                    candidate=untruncated,
                )

            for candidate in candidates:
                if candidate.spectral_variant != "none":
                    continue
                candidate.spectral_calibration_safe = True
                key = (
                    candidate.proposal_type.value,
                    float(candidate.alpha_shared),
                    float(candidate.alpha_safe),
                )
                try:
                    original_losses[key] = untruncated_loss(candidate)
                except Exception as exc:
                    candidate.spectral_calibration_safe = False
                    candidate.screening_reason = (
                        f"original_calibration_error:{type(exc).__name__}:{exc}"
                    )
            for candidate in candidates:
                if candidate.spectral_variant == "none":
                    continue
                if candidate.spectral_calibration_safe is True:
                    # A cumulative repair clone inherits the calibration-only
                    # surgery decision already made for its seed before repair.
                    continue
                key = (
                    candidate.proposal_type.value,
                    float(candidate.alpha_shared),
                    float(candidate.alpha_safe),
                )
                try:
                    operated_loss = untruncated_loss(candidate)
                    original_loss = original_losses[key]
                except Exception as exc:
                    candidate.spectral_calibration_safe = False
                    candidate.screening_reason = (
                        f"spectral_calibration_error:{type(exc).__name__}:{exc}"
                    )
                    spectral_rejections.add(id(candidate))
                    continue
                if (
                    operated_loss
                    > original_loss + self.config.gates.absolute_numerical_tolerance
                ):
                    candidate.spectral_calibration_safe = False
                    candidate.screening_reason = "spectral_calibration_regression"
                    spectral_rejections.add(id(candidate))
                    self.diagnostics.fallback(
                        "spectral_calibration_regression",
                        f"original={original_loss},operated={operated_loss}",
                        episode_id=self._active_episode_id,
                    )
                else:
                    candidate.spectral_calibration_safe = True

        history_before_by_context = (
            self._evaluate_by_context(trainer, history_windows, state="before")
            if history_windows and self.config.gates.history_enabled
            else {}
        )
        proposal_types = {candidate.proposal_type for candidate in candidates}
        if len(proposal_types) > 1:
            raise FDPSCIntegrationError(
                "one screening batch cannot mix global and exception proposals"
            )
        drift_before = (
            self._drift_by_layer(
                self._factor_maps_for_state(
                    proposal_type=next(iter(proposal_types))
                )
            )
            if self.config.gates.spectral_drift_enabled and proposal_types
            else {}
        )
        feasible: List[_Candidate] = []
        for candidate in candidates:
            if id(candidate) in spectral_rejections:
                continue
            try:
                if not self._stabilize_candidate_rank(
                    trainer,
                    candidate,
                    calibration,
                ):
                    continue
                candidate.calibration_loss = self._evaluate_state(
                    trainer,
                    calibration,
                    state="candidate",
                    candidate=candidate,
                )
                candidate.calibration_gain = calibration_before - candidate.calibration_loss
                required = self.config.gates.fast_gain_retention * fast_gain
                if candidate.calibration_gain + self.config.gates.absolute_numerical_tolerance < required:
                    candidate.screening_reason = "calibration_fast_gain_retention"
                    continue

                maximum_error = max(
                    candidate.functional_error_by_layer.values(),
                    default=0.0,
                )
                if self.config.gates.functional_error_enabled and (
                    not math.isfinite(maximum_error)
                    or maximum_error
                    > self.config.slow_lora.functional_error_threshold
                    + self.config.gates.absolute_numerical_tolerance
                ):
                    candidate.screening_reason = "functional_error"
                    continue

                if history_windows and self.config.gates.history_enabled:
                    candidate.history_loss = self._evaluate_state(
                        trainer,
                        self._replay_records(history_windows),
                        state="candidate",
                        candidate=candidate,
                    )
                    if (
                        history_before is None
                        or candidate.history_loss - history_before
                        > self.config.gates.history_loss_tolerance
                        + self.config.gates.absolute_numerical_tolerance
                    ):
                        candidate.screening_reason = "history_regression"
                        continue
                    candidate_by_context = self._evaluate_by_context(
                        trainer,
                        history_windows,
                        state="candidate",
                        candidate=candidate,
                    )
                    if set(candidate_by_context) != set(history_before_by_context):
                        candidate.screening_reason = "history_context_mismatch"
                        continue
                    candidate.worst_context_regression = max(
                        (
                            candidate_by_context[key] - history_before_by_context[key]
                            for key in sorted(candidate_by_context)
                        ),
                        default=0.0,
                    )
                    if (
                        candidate.worst_context_regression
                        > self.config.gates.worst_context_loss_tolerance
                        + self.config.gates.absolute_numerical_tolerance
                    ):
                        candidate.screening_reason = "worst_context_regression"
                        continue

                if self.config.gates.anchor_enabled:
                    candidate.anchor_loss = self._evaluate_state(
                        trainer,
                        anchor,
                        state="candidate",
                        candidate=candidate,
                    )
                    if anchor_before is None:
                        candidate.screening_reason = "anchor_missing"
                        continue
                    candidate.anchor_regression = candidate.anchor_loss - anchor_before
                    if (
                        candidate.anchor_regression
                        > self.config.gates.anchor_loss_tolerance
                        + self.config.gates.absolute_numerical_tolerance
                    ):
                        candidate.screening_reason = "anchor_regression"
                        continue

                if self.config.gates.plasticity_enabled:
                    candidate.plasticity_gain = self._plasticity_gain(
                        trainer,
                        state="candidate",
                        candidate=candidate,
                    )
                    if plasticity_before is None or candidate.plasticity_gain is None:
                        candidate.screening_reason = "plasticity_missing"
                        continue
                    if not self._plasticity_screening_feasible(
                        plasticity_before,
                        candidate.plasticity_gain,
                    ):
                        candidate.screening_reason = "plasticity_regression"
                        continue

                if self.config.gates.spectral_drift_enabled:
                    candidate_drift = self._drift_by_layer(
                        self._factor_maps_for_state(candidate=candidate)
                    )
                    if set(candidate_drift) != set(drift_before):
                        candidate.screening_reason = "spectral_drift_missing"
                        continue
                    candidate.maximum_drift_increase = max(
                        (
                            candidate_drift[key] - drift_before[key]
                            for key in sorted(candidate_drift)
                        ),
                        default=0.0,
                    )
                    if (
                        candidate.maximum_drift_increase
                        > self.config.gates.drift_tolerance
                        + self.config.gates.absolute_numerical_tolerance
                    ):
                        candidate.screening_reason = "spectral_drift"
                        continue

                candidate.screening_reason = "calibration_feasible"
                feasible.append(candidate)
            except Exception as exc:
                candidate.screening_reason = f"screening_error:{type(exc).__name__}:{exc}"

        feasible.sort(
            key=lambda item: (
                -item.calibration_gain,
                item.worst_context_regression or 0.0,
                item.anchor_regression or 0.0,
                max(item.functional_error_by_layer.values(), default=0.0),
                sum(item.selected_rank_by_layer.values()),
                item.proposal_type.value,
                item.alpha_shared,
                item.alpha_safe,
                item.spectral_variant,
            )
        )
        selected = feasible[0] if feasible else None
        if selected is not None:
            selected.screening_reason = "selected_by_calibration_lexicographic"
        return selected, [candidate.summary() for candidate in candidates]

    def _repair_candidate(
        self,
        trainer: Any,
        seed_candidate: Optional[_Candidate],
        *,
        calibration: Sequence[Any],
        calibration_before: float,
        calibration_fast: float,
        history_windows: Sequence[ReplayWindow],
        history_before: Optional[float],
        anchor: Sequence[Any],
        anchor_before: Optional[float],
        plasticity_before: Optional[float],
    ) -> Optional[_Candidate]:
        """Run the bounded cumulative JEPA repair trajectory.

        Every optimizer gradient originates in the unchanged JEPA objective on
        audited current support and (when present) the correct replay bank.
        Geometry is applied to effective weight gradients and delivered to the
        trainable factors through the same stop-gradient proxy used by SDC.
        Commit-query data is structurally absent from this API.
        """

        if self.repair_engine is None or seed_candidate is None or not self._support_segments:
            return None
        if (
            self.config.gates.history_enabled
            and self._successful_slow_commit_count > 0
            and not history_windows
        ):
            return None
        fast_gain = calibration_before - calibration_fast
        if fast_gain <= self.config.gates.absolute_numerical_tolerance:
            return None
        self.state_machine.enter_repair("quick_candidates_failed")
        initial = copy.deepcopy(seed_candidate)

        # LPR is computed only when old replay exists.  Cache the actual
        # Pbefore key-layer outputs, not adapter inputs or delta-only proxies;
        # candidate outputs are recomputed with autograd on every repair step.
        proximal_before_outputs: Dict[str, Tensor] = {}
        proximal_records: Tuple[Mapping[str, Any], ...] = ()
        if self.config.repair.proximal_enabled and history_windows:
            proximal_ids = self._proximal_key_layer_ids()
            if not proximal_ids:
                raise FDPSCIntegrationError(
                    "repair.proximal_enabled matched no injected logical layer"
                )
            proximal_records = self._replay_records(history_windows)
            proximal_before_outputs = self._collect_key_layer_outputs(
                trainer,
                proximal_records,
                logical_ids=proximal_ids,
                state="before",
                differentiable_candidate=False,
            )

        optimizer: Optional[torch.optim.Optimizer] = None
        repair_parameters: Dict[str, Tuple[nn.Parameter, nn.Parameter]] = {}

        def train_step(
            live: _Candidate,
            batch: Any,
            step_index: int,
            use_pcgrad: bool,
        ) -> Mapping[str, float]:
            nonlocal optimizer
            if optimizer is None:
                for logical_id, factors in sorted(live.factors_by_layer.items()):
                    if factors.rank == 0:
                        continue
                    b = nn.Parameter(factors.B.detach().clone())
                    a = nn.Parameter(factors.A.detach().clone())
                    repair_parameters[logical_id] = (b, a)
                    live.factors_by_layer[logical_id] = LowRankFactors(b, a)
                parameters = [
                    parameter
                    for pair in repair_parameters.values()
                    for parameter in pair
                ]
                if not parameters:
                    return {
                        "repair_jepa_loss": 0.0,
                        "repair_proxy_loss": 0.0,
                        "current_weight": float(batch.normalized_weights.get("current", 0.0)),
                        "replay_weight": float(batch.normalized_weights.get("replay", 0.0)),
                    }
                optimizer_type = {
                    "adam": torch.optim.Adam,
                    "adamw": torch.optim.AdamW,
                    "sgd": torch.optim.SGD,
                }.get(self.config.repair.optimizer)
                if optimizer_type is None:
                    raise FDPSCIntegrationError(
                        f"unsupported repair optimizer {self.config.repair.optimizer!r}"
                    )
                optimizer = optimizer_type(
                    parameters,
                    lr=self.config.repair.learning_rate,
                )

            current_records = self._support_records(batch.current)
            current_gradients, current_loss = self._collect_effective_gradients(
                trainer,
                current_records,
                state="candidate",
                candidate=live,
                include_loss=True,
            )
            replay_gradients: Dict[str, Tensor] = {}
            replay_loss = 0.0
            if batch.replay:
                replay_gradients, replay_loss = self._collect_effective_gradients(
                    trainer,
                    self._replay_records(batch.replay),
                    state="candidate",
                    candidate=live,
                    include_loss=True,
                )
            anchor_gradients: Dict[str, Tensor] = {}
            if use_pcgrad and anchor:
                anchor_gradients = self._collect_effective_gradients(
                    trainer,
                    anchor,
                    state="candidate",
                    candidate=live,
                )

            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            proxy_terms: List[Tensor] = []
            correction_norms: List[float] = []
            current_weight = float(batch.normalized_weights.get("current", 0.0))
            replay_weight = float(batch.normalized_weights.get("replay", 0.0))
            proximal_weight = float(batch.normalized_weights.get("proximal", 0.0))
            for logical_id, (b, a) in sorted(repair_parameters.items()):
                effective_gradient = current_weight * current_gradients[logical_id]
                history_reference = replay_gradients.get(logical_id)
                if history_reference is not None:
                    effective_gradient = effective_gradient + replay_weight * history_reference
                raw_gradient = effective_gradient
                if use_pcgrad:
                    anchor_reference = anchor_gradients.get(logical_id)
                    method = self.config.gradient_geometry.projection_method
                    if method == "dual_constraint":
                        projected = dual_constraint_projection(
                            effective_gradient,
                            history=history_reference,
                            anchor=anchor_reference,
                            history_slack=self.config.gradient_geometry.history_slack,
                            anchor_slack=self.config.gradient_geometry.anchor_slack,
                            epsilon=self.config.gradient_geometry.epsilon,
                        )
                        if projected.feasible:
                            effective_gradient = projected.gradient
                        else:
                            self.diagnostics.fallback(
                                "repair_dual_projection_infeasible",
                                projected.reason,
                                episode_id=self._active_episode_id,
                                logical_layer_id=logical_id,
                            )
                            raise _RepairGeometryInfeasible(
                                f"{logical_id}:{projected.reason}"
                            )
                    elif method in {"c_pcgrad", "per_step_c_pcgrad"}:
                        for reference in (history_reference, anchor_reference):
                            if reference is not None:
                                effective_gradient = c_pcgrad(
                                    effective_gradient,
                                    reference,
                                    coefficient=self.config.gradient_geometry.c_pcgrad_coefficient,
                                    epsilon=self.config.gradient_geometry.epsilon,
                                )
                    else:
                        raise FDPSCIntegrationError(
                            f"unsupported repair projection method {method!r}"
                        )
                correction_norms.append(
                    float(
                        torch.linalg.vector_norm(
                            (effective_gradient - raw_gradient).to(dtype=torch.float32)
                        ).detach().cpu()
                    )
                )
                proxy_terms.append(effective_gradient_proxy(effective_gradient, b, a))
            if proximal_weight > 0 and proximal_before_outputs:
                candidate_outputs = self._collect_key_layer_outputs(
                    trainer,
                    proximal_records,
                    logical_ids=tuple(sorted(proximal_before_outputs)),
                    state="candidate",
                    candidate=live,
                    differentiable_candidate=True,
                )
                if set(candidate_outputs) != set(proximal_before_outputs):
                    raise FDPSCIntegrationError("LPR key-layer output set changed")
                for logical_id in sorted(proximal_before_outputs):
                    current_h = candidate_outputs[logical_id]
                    before_h = proximal_before_outputs[logical_id].to(
                        device=current_h.device,
                        dtype=current_h.dtype,
                    )
                    if current_h.shape != before_h.shape:
                        raise FDPSCIntegrationError(
                            f"LPR output shape changed for {logical_id}: "
                            f"{tuple(before_h.shape)} -> {tuple(current_h.shape)}"
                        )
                    proxy_terms.append(
                        proximal_weight
                        * torch.mean(
                            (
                                current_h.to(dtype=torch.float32)
                                - before_h.to(dtype=torch.float32)
                            )
                            ** 2
                        )
                    )
            if not proxy_terms:
                raise FDPSCIntegrationError("repair produced no trainable factor proxy")
            proxy_loss = torch.stack(proxy_terms).sum()
            proxy_loss.backward()
            optimizer.step()
            weighted_jepa = current_weight * current_loss + replay_weight * replay_loss
            return {
                "repair_jepa_loss": weighted_jepa,
                "repair_proxy_loss": float(proxy_loss.detach().cpu()),
                "repair_gradient_correction_norm": (
                    sum(correction_norms) / len(correction_norms)
                    if correction_norms
                    else 0.0
                ),
                "current_weight": current_weight,
                "replay_weight": replay_weight,
                "proximal_weight": proximal_weight,
            }

        def screen(value: _Candidate, step: int) -> ScreeningResult:
            # ``RepairEngine`` screens a clone, so this snapshot is the
            # immutable untruncated cumulative repair checkpoint.  Candidate
            # rank selection below may compress the clone repeatedly without
            # altering the live optimizer trajectory.
            value.rank_reference_by_layer = {
                logical_id: LowRankFactors(
                    factors.B.detach().clone(),
                    factors.A.detach().clone(),
                )
                for logical_id, factors in sorted(value.factors_by_layer.items())
            }
            selected, _ = self._screen_candidates(
                trainer,
                [value],
                calibration=calibration,
                calibration_before=calibration_before,
                calibration_fast=calibration_fast,
                history_windows=history_windows,
                history_before=history_before,
                anchor=anchor,
                anchor_before=anchor_before,
                plasticity_before=plasticity_before,
            )
            value.repair_step = step
            return ScreeningResult(
                passed=selected is not None,
                metrics={
                    "calibration_loss": float(value.calibration_loss),
                    "calibration_gain": float(value.calibration_gain),
                },
                reason=(
                    "full_calibration_screen_passed"
                    if selected is not None
                    else value.screening_reason
                ),
            )

        try:
            result = self.repair_engine.run(
                initial,
                current_windows=self._support_segments,
                replay_windows=history_windows,
                train_step=train_step,
                screen_candidate=screen,
            )
        except _RepairGeometryInfeasible as exc:
            self.metrics.record(
                "repair_projection_rejected",
                1,
                episode_id=self._active_episode_id,
                tags={"reason": str(exc)},
            )
            return None
        if not result.succeeded:
            return None
        repaired = result.final_state
        repaired.screening_reason = "repair_first_feasible_checkpoint"
        return repaired

    # ------------------------------------------------------------------
    # Commit gates, replay/Q updates, atomic persistence, and sleep
    # ------------------------------------------------------------------
    def _factor_maps_for_state(
        self,
        *,
        candidate: Optional[_Candidate] = None,
        proposal_type: Optional[ProposalType] = None,
    ) -> Dict[str, LowRankFactors]:
        with self._preserve_adapter_runtime():
            self._load_episode_start()
            if candidate is not None:
                self._apply_candidate(candidate)
                effective_type = candidate.proposal_type
            else:
                effective_type = proposal_type
            if effective_type is None:
                raise FDPSCIntegrationError(
                    "persistent factor projection requires a proposal type"
                )
            if effective_type == ProposalType.GLOBAL_SLOW:
                return {
                    key: self._canonical(adapter.get_slow_factors())
                    for key, adapter in sorted(self.injection.adapters.items())
                }
            return {
                key: LowRankFactors(value.B.detach().clone(), value.A.detach().clone())
                for key, value in self._persistent_effective_factors().items()
            }

    def _drift_by_layer(
        self,
        factors_by_layer: Mapping[str, LowRankFactors],
    ) -> Dict[str, float]:
        result = {}
        for logical_id, factors in sorted(factors_by_layer.items()):
            spectrum = self._base_spectra.get(logical_id)
            if spectrum is None:
                continue
            drift = spectral_drift(
                factors,
                spectrum,
                epsilon=self.config.gradient_geometry.epsilon,
            )
            if not drift.available:
                raise FDPSCIntegrationError(
                    f"spectral drift unavailable for {logical_id}: {drift.reason}"
                )
            result[logical_id] = drift.value
        return result

    def _exception_state(self, candidate: _Candidate) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "target_manifest_hash": self.target_manifest.hash,
            "layers": {
                logical_id: {
                    "B": factors.B.detach().cpu().clone(),
                    "A": factors.A.detach().cpu().clone(),
                }
                for logical_id, factors in sorted(candidate.factors_by_layer.items())
            },
        }

    def _validate_replay_support_segment(self, segment: _SupportSegment) -> None:
        """Validate one registered support segment before temporal composition.

        Online adaptation keeps the repository's original recent-buffer merge
        untouched.  Replay composition is a later, read-only operation, so it
        must re-prove the episode/data identity and the one-action/one-
        transition, T+1-observation alignment from the immutable registered
        copies.
        """

        if self._active_episode_id is None or self._active_context is None:
            raise FDPSCIntegrationError("support replay validation requires an active episode")
        expected_preprocess = str(self.preprocess_hash or "")
        if (
            segment.episode_id != str(self._active_episode_id)
            or segment.context_identifier != str(self._active_context)
            or segment.identity.context_identifier != str(self._active_context)
            or segment.preprocess_hash != expected_preprocess
            or str(segment.latent_adapter_schema) != str(LATENT_SCHEMA_VERSION)
        ):
            raise FDPSCIntegrationError(
                "support replay segment episode/context/preprocess/schema identity changed"
            )
        if len(segment.identity.trajectory_ids) != 1:
            raise FDPSCIntegrationError(
                "support replay segment must name exactly one continuous trajectory"
            )
        if not isinstance(segment.actions, Tensor) or segment.actions.ndim < 2:
            raise FDPSCIntegrationError("support replay actions must be [batch,time,...]")
        time_steps = int(segment.actions.shape[1])
        if time_steps <= 0:
            raise FDPSCIntegrationError("support replay segment must contain an action")
        if len(segment.identity.transition_ids) != time_steps:
            raise FDPSCIntegrationError(
                "support transition identity count does not match the action sequence"
            )
        if len(segment.identity.frame_ids) != time_steps + 1:
            raise FDPSCIntegrationError(
                "support frame identity count must be exactly actions+1"
            )
        if not torch.isfinite(segment.actions).all():
            raise FDPSCIntegrationError("support replay actions must be finite")
        if not segment.obs:
            raise FDPSCIntegrationError("support replay observations are empty")
        for key, value in sorted(segment.obs.items()):
            if not isinstance(value, Tensor) or value.ndim < 2:
                raise FDPSCIntegrationError(
                    f"support replay observation {key!r} must be [batch,time,...]"
                )
            if int(value.shape[0]) != int(segment.actions.shape[0]):
                raise FDPSCIntegrationError(
                    f"support replay observation {key!r} batch differs from actions"
                )
            if int(value.shape[1]) != time_steps + 1:
                raise FDPSCIntegrationError(
                    f"support replay observation {key!r} must contain actions+1 frames"
                )
            if not torch.isfinite(value).all():
                raise FDPSCIntegrationError(
                    f"support replay observation {key!r} must be finite"
                )
        actual_hash = _hash_tensor_tree(
            {"obs": segment.obs, "actions": segment.actions}
        )
        if actual_hash != segment.identity.content_hash:
            raise FDPSCIntegrationError(
                "registered support tensors no longer match their audited content hash"
            )
        if self.external is not None:
            registered = self.external._support_records.get(segment.identity.record_id)
            # A composed segment names its complete source provenance instead
            # of being inserted into the online support ledger a second time.
            if len(segment.source_record_ids) <= 1:
                if registered != segment.identity:
                    raise FDPSCIntegrationError(
                        "support replay identity is absent from the audited support ledger"
                    )
            else:
                self.external.audit_composed_support(
                    segment.identity,
                    segment.source_record_ids,
                )

    def _support_boundary_is_continuous(
        self,
        left: _SupportSegment,
        right: _SupportSegment,
    ) -> bool:
        """Return true only for a fully proved adjacent trajectory boundary.

        Identity gaps are ordinary non-contiguous input and start a new replay
        chain.  Once stable IDs claim adjacency, a tensor mismatch is corrupt
        provenance and fails closed rather than silently joining or ignoring
        the forged boundary.
        """

        if (
            left.episode_id != right.episode_id
            or left.context_identifier != right.context_identifier
            or left.preprocess_hash != right.preprocess_hash
            or left.latent_adapter_schema != right.latent_adapter_schema
            or left.identity.trajectory_ids != right.identity.trajectory_ids
            or int(right.iteration) != int(left.iteration) + 1
            or left.identity.frame_ids[-1] != right.identity.frame_ids[0]
        ):
            return False
        if set(left.identity.transition_ids) & set(right.identity.transition_ids):
            raise FDPSCIntegrationError(
                "adjacent support segments reuse a transition identity"
            )
        unexpected_frame_overlap = (
            set(left.identity.frame_ids) & set(right.identity.frame_ids)
        ) - {left.identity.frame_ids[-1]}
        if unexpected_frame_overlap:
            raise FDPSCIntegrationError(
                "adjacent support segments overlap non-boundary frame identities"
            )
        if set(left.obs) != set(right.obs):
            raise FDPSCIntegrationError(
                "adjacent support segments changed observation schema"
            )
        if (
            left.actions.dtype != right.actions.dtype
            or left.actions.shape[0] != right.actions.shape[0]
            or left.actions.shape[2:] != right.actions.shape[2:]
        ):
            raise FDPSCIntegrationError(
                "adjacent support segments changed action schema"
            )
        for key in sorted(left.obs):
            before = left.obs[key]
            after = right.obs[key]
            if (
                before.dtype != after.dtype
                or before.shape[0] != after.shape[0]
                or before.shape[2:] != after.shape[2:]
            ):
                raise FDPSCIntegrationError(
                    f"adjacent support segments changed observation schema for {key!r}"
                )
            if not torch.equal(before[:, -1], after[:, 0]):
                raise FDPSCIntegrationError(
                    f"support boundary frame content mismatch for stable ID "
                    f"{left.identity.frame_ids[-1]!r} ({key})"
                )
        return True

    def _compose_support_chain(
        self,
        chain: Sequence[_SupportSegment],
    ) -> _SupportSegment:
        if not chain:
            raise FDPSCIntegrationError("cannot compose an empty support chain")
        if len(chain) == 1:
            return chain[0]
        first = chain[0]
        observations = {
            key: torch.cat(
                [first.obs[key], *(segment.obs[key][:, 1:] for segment in chain[1:])],
                dim=1,
            )
            for key in sorted(first.obs)
        }
        actions = torch.cat([segment.actions for segment in chain], dim=1)
        transitions = tuple(
            transition
            for segment in chain
            for transition in segment.identity.transition_ids
        )
        frames = tuple(first.identity.frame_ids) + tuple(
            frame
            for segment in chain[1:]
            for frame in segment.identity.frame_ids[1:]
        )
        if len(transitions) != len(set(transitions)) or len(frames) != len(set(frames)):
            raise FDPSCIntegrationError(
                "composed support replay window has duplicate non-boundary identities"
            )
        source_record_ids = tuple(
            record_id
            for segment in chain
            for record_id in (
                segment.source_record_ids or (segment.identity.record_id,)
            )
        )
        source_iterations = tuple(
            iteration
            for segment in chain
            for iteration in (
                segment.source_iterations or (int(segment.iteration),)
            )
        )
        provenance_hash = hashlib.sha256(
            "\0".join(source_record_ids).encode("utf-8")
        ).hexdigest()
        content_hash = _hash_tensor_tree({"obs": observations, "actions": actions})
        composed = _SupportSegment(
            identity=DataIdentity(
                record_id=f"composed-support:{provenance_hash}",
                context_identifier=first.identity.context_identifier,
                trajectory_ids=first.identity.trajectory_ids,
                transition_ids=transitions,
                frame_ids=frames,
                content_hash=content_hash,
            ),
            obs=observations,
            actions=actions,
            iteration=int(chain[-1].iteration),
            episode_id=first.episode_id,
            context_identifier=first.context_identifier,
            preprocess_hash=first.preprocess_hash,
            latent_adapter_schema=first.latent_adapter_schema,
            source_record_ids=source_record_ids,
            source_iterations=source_iterations,
        )
        self._validate_replay_support_segment(composed)
        return composed

    def _eligible_replay_segments(self) -> Tuple[_SupportSegment, ...]:
        """Build non-overlapping complete windows from audited support order.

        A single MPC feedback segment can be shorter than the trained JEPA
        horizon (the shipped setup is one model action per replan versus three
        history actions).  Consecutive segments are therefore accumulated only
        until they cover ``num_hist + num_pred`` frames.  An identity gap
        flushes the incomplete chain; no observation is fabricated and the
        online recent-buffer/JEPA update path is not touched.
        """

        minimum_frames = int(getattr(self.wm, "num_hist", 1)) + int(
            getattr(self.wm, "num_pred", 1)
        )
        minimum_actions = max(1, minimum_frames - 1)
        eligible: List[_SupportSegment] = []
        chain: List[_SupportSegment] = []
        action_count = 0
        for segment in self._support_segments:
            self._validate_replay_support_segment(segment)
            if chain and not self._support_boundary_is_continuous(chain[-1], segment):
                chain = []
                action_count = 0
            chain.append(segment)
            action_count += int(segment.actions.shape[1])
            if action_count >= minimum_actions:
                eligible.append(self._compose_support_chain(chain))
                chain = []
                action_count = 0
        return tuple(eligible)

    def _build_replay_windows(self, trainer: Any) -> List[ReplayWindow]:
        if self._context_embedding is None:
            raise FDPSCIntegrationError("context embedding is unavailable")
        result = []
        visual_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(self.config.replay.visual_latent_dtype)
        if visual_dtype is None:
            raise FDPSCIntegrationError(
                f"unsupported replay visual dtype {self.config.replay.visual_latent_dtype!r}"
            )
        auxiliary_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(self.config.replay.auxiliary_dtype)
        if auxiliary_dtype is None:
            raise FDPSCIntegrationError(
                f"unsupported replay auxiliary dtype {self.config.replay.auxiliary_dtype!r}"
            )
        # Long-term residual similarity is defined only in theta_0 space.  Do
        # the entire extraction in one isolated eval-mode scope so persistent
        # slow/exception and episodic adapters, dropout, and train-mode buffers
        # cannot leak into either the stored latent or residual descriptor.
        with self._preserve_adapter_runtime(), self._all_adapters_disabled():
            self.wm.eval()
            self.injection.enforce_frozen_base_eval()
            with torch.no_grad():
                for segment in self._eligible_replay_segments():
                    obs_device = {
                        key: value.to(self.device) for key, value in segment.obs.items()
                    }
                    actions_device = segment.actions.to(self.device)
                    latent = self.wm.extract_frozen_visual_latent(obs_device).detach(clone=True)
                    prepared_obs, prepared_actions = trainer._prepare_segment(
                        segment.obs,
                        segment.actions,
                    )
                    prepared_proprio = prepared_obs.get("proprio")
                    if not isinstance(prepared_proprio, Tensor):
                        raise FDPSCIntegrationError("support replay requires proprio")
                    # Rebuild JEPA tokens from the exact frozen latent object
                    # being persisted, rather than running a second backbone
                    # pass that is merely expected to be equivalent.
                    theta0_z = self.wm.encode_from_frozen_visual_latent(
                        latent,
                        prepared_proprio,
                        prepared_actions,
                    )
                    residual_descriptor = trainer._prediction_residual_descriptor(theta0_z)
                    theta0_loss = trainer._prediction_loss(
                        theta0_z,
                        detach_src=True,
                        detach_tgt=True,
                    )
                    payload = latent.to_payload()
                    payload["tensor"] = payload["tensor"].detach().cpu().to(dtype=visual_dtype)
                    time = int(segment.actions.shape[1])
                    proprio = segment.obs.get("proprio")
                    if not isinstance(proprio, Tensor):
                        raise FDPSCIntegrationError("support replay requires proprio")
                    result.append(
                        ReplayWindow(
                            window_id=f"window:{segment.identity.record_id}",
                            trajectory_id=segment.identity.trajectory_ids[0],
                            transition_ids=segment.identity.transition_ids,
                            frame_ids=segment.identity.frame_ids,
                            timesteps=tuple(range(time + 1)),
                            content_hash=segment.identity.content_hash,
                            context_identifier=str(self._active_context),
                            context_embedding=self._context_embedding,
                            visual_latent=payload,
                            proprio=proprio.detach().cpu().to(dtype=auxiliary_dtype),
                            actions=actions_device.detach().cpu().to(dtype=auxiliary_dtype),
                            time_positions=torch.arange(time + 1),
                            prediction_mask=torch.ones(time, dtype=torch.bool),
                            residual=residual_descriptor,
                            source_episode=str(self._active_episode_id),
                            provenance="episode_support",
                            committed=True,
                            preprocess_hash=(self.preprocess_hash or ""),
                            base_checkpoint_hash=self.base_checkpoint_hash,
                            latent_adapter_schema=LATENT_SCHEMA_VERSION,
                            difficulty_score=float(theta0_loss.detach().cpu()),
                            metadata={
                                "replan_iteration": segment.iteration,
                                "replan_iterations": list(
                                    segment.source_iterations
                                    or (int(segment.iteration),)
                                ),
                                "source_support_record_ids": list(
                                    segment.source_record_ids
                                    or (segment.identity.record_id,)
                                ),
                                "composed_across_replans": (
                                    len(segment.source_record_ids) > 1
                                ),
                                "theta0_residual": True,
                                "residual_descriptor_schema": "theta0_jepa_pattern_v1",
                                "residual_descriptor_pooling": (
                                    "signed_mean+rms_over_batch_offset_token_and_"
                                    "signed_mean+rms_over_batch_offset_feature"
                                ),
                            },
                        )
                    )
        return result

    def _update_subspaces_after_slow_commit(
        self,
        trainer: Any,
        windows: Sequence[ReplayWindow],
    ) -> None:
        if not self.config.activation_subspace.enabled or not windows:
            return
        activations = self._collect_activations(
            trainer,
            self._replay_records(windows),
            state="fast",
        )
        cfg = self.config.activation_subspace
        for logical_id, matrix in sorted(activations.items()):
            old = self.subspaces.get(
                logical_id,
                matrix.shape[1],
                matrix.device,
            )
            if old.rank == 0:
                updated = ActivationSubspace.from_activations(
                    matrix,
                    maximum_rank=cfg.maximum_rank,
                    spectral_energy_threshold=cfg.spectral_energy_threshold,
                    minimum_energy=cfg.minimum_energy,
                )
            else:
                updated = old.update(
                    matrix,
                    forgetting_factor=cfg.forgetting_factor,
                    maximum_rank=cfg.maximum_rank,
                    spectral_energy_threshold=cfg.spectral_energy_threshold,
                    minimum_energy=cfg.minimum_energy,
                )
            self.subspaces.set(logical_id, updated)
            self.metrics.record(
                "activation_subspace_rank",
                updated.rank,
                episode_id=self._active_episode_id,
                logical_layer_id=logical_id,
                tags={
                    "state": "transaction_candidate_post_update",
                    "commit_status": "provisional_until_transaction_commit",
                },
            )
            energies = updated.energies.detach().to(dtype=torch.float64).cpu()
            self.metrics.record_nullable(
                "activation_energy_total",
                float(energies.sum().item()),
                status="available",
                episode_id=self._active_episode_id,
                logical_layer_id=logical_id,
                tags={
                    "definition": "sum_retained_activation_covariance_eigenvalues",
                    "component_count": int(energies.numel()),
                    "state": "transaction_candidate_post_update",
                    "commit_status": "provisional_until_transaction_commit",
                },
            )
            if energies.numel() == 0:
                self.metrics.record_nullable(
                    "lambda_distribution",
                    None,
                    status="unavailable",
                    reason="activation_subspace_rank_is_zero",
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                    tags={
                        "state": "transaction_candidate_post_update",
                        "commit_status": "provisional_until_transaction_commit",
                    },
                )
            else:
                for component, energy in enumerate(energies.tolist()):
                    self.metrics.record_nullable(
                        "lambda_distribution",
                        float(energy),
                        status="available",
                        episode_id=self._active_episode_id,
                        logical_layer_id=logical_id,
                        tags={
                            "component_index": component,
                            "component_count": int(energies.numel()),
                            "state": "transaction_candidate_post_update",
                            "commit_status": "provisional_until_transaction_commit",
                        },
                    )

            if not self.config.merge.soft_ness_enabled:
                self.metrics.record_nullable(
                    "p_distribution",
                    None,
                    status="not_applicable",
                    reason="soft_ness_disabled",
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                    tags={
                        "state": "transaction_candidate_post_update",
                        "commit_status": "provisional_until_transaction_commit",
                    },
                )
                continue
            weights = updated.soft_ness_weights(
                mode=cfg.soft_ness_tau_mode,
                fixed_tau=cfg.soft_ness_tau_fixed,
                quantile=cfg.soft_ness_tau_quantile,
                minimum_energy=cfg.minimum_energy,
            )
            if not weights.available or weights.weights.numel() == 0:
                self.metrics.record_nullable(
                    "p_distribution",
                    None,
                    status="unavailable",
                    reason=weights.reason or "soft_ness_weights_are_empty",
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                    tags={
                        "tau": weights.tau,
                        "state": "transaction_candidate_post_update",
                        "commit_status": "provisional_until_transaction_commit",
                    },
                )
            else:
                p_values = weights.weights.detach().to(dtype=torch.float64).cpu()
                for component, probability in enumerate(p_values.tolist()):
                    self.metrics.record_nullable(
                        "p_distribution",
                        float(probability),
                        status="available",
                        episode_id=self._active_episode_id,
                        logical_layer_id=logical_id,
                        tags={
                            "component_index": component,
                            "component_count": int(p_values.numel()),
                            "tau": weights.tau,
                            "state": "transaction_candidate_post_update",
                            "commit_status": "provisional_until_transaction_commit",
                        },
                    )

    @staticmethod
    def _canonicalize_canary_adapter_state(state: Mapping[str, Any]) -> Dict[str, Any]:
        """Strip episode-local and fixed-route state from a rollout snapshot."""

        result = copy.deepcopy(dict(state))
        slow_b = result.get("slow_B")
        slow_a = result.get("slow_A")
        if not isinstance(slow_b, Tensor) or not isinstance(slow_a, Tensor):
            raise FDPSCIntegrationError("canary adapter snapshot is missing slow factors")
        result["exception_B"] = slow_b.new_empty((slow_b.shape[0], 0))
        result["exception_A"] = slow_a.new_empty((0, slow_a.shape[1]))
        result["active_exception_id"] = None
        pilot_b = result.get("pilot_B")
        if isinstance(pilot_b, Tensor):
            result["pilot_B"] = torch.zeros_like(pilot_b)
        result["pilot_frozen"] = True
        result["centered_active"] = False
        result["center_B"] = None
        result["center_A"] = None
        result["center_B0"] = slow_b.new_empty((slow_b.shape[0], 0))
        result["center_A0"] = slow_a.new_empty((0, slow_a.shape[1]))
        result["adapters_enabled"] = True
        return result

    def _capture_canary_algorithm_state(self) -> Dict[str, Any]:
        router_state = copy.deepcopy(self.exception_router.state_dict())
        router_state["active_episode_id"] = None
        router_state["active_route"] = None
        state = {
            "schema_version": 1,
            "config_identity": self.config.persistence_identity_hash(),
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "successful_slow_commit_count": self._successful_slow_commit_count,
            "adapters": {
                key: self._canonicalize_canary_adapter_state(
                    adapter.adapter_state_dict()
                )
                for key, adapter in sorted(self.injection.adapters.items())
            },
            "replay": copy.deepcopy(self.replay.state_dict()),
            "activation_subspaces": copy.deepcopy(self.subspaces.state_dict()),
            "exception_router": router_state,
            "history_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._history_gradients.items())
            },
            "anchor_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._anchor_gradients.items())
            },
            "repair": (
                copy.deepcopy(self.repair_engine.state_dict())
                if self.repair_engine is not None
                else None
            ),
            "rng": RNGSnapshot.capture(),
        }
        assert_finite_tree(
            {
                "adapters": state["adapters"],
                "replay": state["replay"],
                "activation_subspaces": state["activation_subspaces"],
                "exception_router": state["exception_router"],
                "history_gradients": state["history_gradients"],
                "anchor_gradients": state["anchor_gradients"],
            },
            path="canary_known_good",
        )
        return state

    def _canary_rollout_state_from_algorithm(
        self,
        algorithm_state: Mapping[str, Any],
        *,
        episode_sequence: int,
        commit_sequence: int,
    ) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "episode_sequence": int(episode_sequence),
            "commit_sequence": int(commit_sequence),
            "successful_slow_commit_count": int(
                algorithm_state["successful_slow_commit_count"]
            ),
            "context_identifier": None,
            "adapters": copy.deepcopy(algorithm_state["adapters"]),
            "replay": copy.deepcopy(algorithm_state["replay"]),
            "activation_subspaces": copy.deepcopy(
                algorithm_state["activation_subspaces"]
            ),
            "exception_router": copy.deepcopy(algorithm_state["exception_router"]),
            "history_gradients": copy.deepcopy(algorithm_state["history_gradients"]),
            "anchor_gradients": copy.deepcopy(algorithm_state["anchor_gradients"]),
        }

    def _promote_canary_known_good(
        self,
        *,
        commit_id: Optional[str],
        commit_sequence: int,
        persistent_commit_count: int,
    ) -> None:
        if self.canary_runner is None:
            return
        algorithm_state = self._capture_canary_algorithm_state()
        self._canary_known_good = {
            "schema_version": 1,
            "commit_id": None if commit_id is None else str(commit_id),
            "commit_sequence": int(commit_sequence),
            "episode_sequence": int(self._episode_sequence),
            "persistent_commit_count": int(persistent_commit_count),
            "algorithm_state": algorithm_state,
            "rollout_state": self._canary_rollout_state_from_algorithm(
                algorithm_state,
                episode_sequence=self._episode_sequence,
                commit_sequence=commit_sequence,
            ),
        }
        self._canary_pending_commit_ids = []

    def _canary_period_state(self) -> Dict[str, Any]:
        if self.canary_runner is None:
            return {"schema_version": 1, "enabled": False}
        if self._canary_known_good is None:
            raise FDPSCIntegrationError("enabled canary has no known-good state")
        return {
            "schema_version": 1,
            "enabled": True,
            "known_good": copy.deepcopy(self._canary_known_good),
            "pending_commit_ids": list(self._canary_pending_commit_ids),
            "last_rollback": copy.deepcopy(self._canary_last_rollback),
        }

    def _load_canary_period_state(
        self,
        state: Mapping[str, Any],
        *,
        validate_runtime: bool,
    ) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise CheckpointValidationError("unsupported canary-period state schema")
        enabled = bool(state.get("enabled", False))
        if enabled != (self.canary_runner is not None):
            raise CheckpointValidationError("sidecar canary enablement mismatch")
        if not enabled:
            self._canary_known_good = None
            self._canary_pending_commit_ids = []
            self._canary_last_rollback = None
            return
        known_good = state.get("known_good")
        if not isinstance(known_good, Mapping):
            raise CheckpointValidationError("canary-period state is missing known_good")
        if int(known_good.get("schema_version", -1)) != 1:
            raise CheckpointValidationError("unsupported known-good state schema")
        algorithm = known_good.get("algorithm_state")
        rollout = known_good.get("rollout_state")
        if not isinstance(algorithm, Mapping) or not isinstance(rollout, Mapping):
            raise CheckpointValidationError("known-good algorithm/rollout state is missing")
        expected = {
            "config_identity": self.config.persistence_identity_hash(),
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
        }
        for key, value in expected.items():
            if algorithm.get(key) != value:
                raise CheckpointValidationError(f"known-good {key} mismatch")
        if rollout.get("base_checkpoint_hash") != self.base_checkpoint_hash:
            raise CheckpointValidationError("known-good rollout base hash mismatch")
        if rollout.get("target_manifest_hash") != self.target_manifest.hash:
            raise CheckpointValidationError("known-good rollout manifest hash mismatch")
        if set(algorithm.get("adapters", {})) != set(self.injection.adapters):
            raise CheckpointValidationError("known-good adapter registry mismatch")
        good_sequence = int(known_good.get("commit_sequence", -1))
        good_episode = int(known_good.get("episode_sequence", -1))
        good_count = int(known_good.get("persistent_commit_count", -1))
        if min(good_sequence, good_episode, good_count) < 0:
            raise CheckpointValidationError("known-good counters must be non-negative")
        if validate_runtime and good_sequence > self._commit_sequence:
            raise CheckpointValidationError("known-good sequence exceeds commit high-water mark")
        if int(rollout.get("commit_sequence", -1)) != good_sequence:
            raise CheckpointValidationError("known-good rollout commit sequence mismatch")
        if int(rollout.get("episode_sequence", -1)) != good_episode:
            raise CheckpointValidationError("known-good rollout episode sequence mismatch")
        algorithm_slow_count = int(
            algorithm.get("successful_slow_commit_count", -1)
        )
        rollout_slow_count = int(
            rollout.get("successful_slow_commit_count", -1)
        )
        if algorithm_slow_count < 0 or rollout_slow_count != algorithm_slow_count:
            raise CheckpointValidationError(
                "known-good successful slow-commit count is missing or inconsistent"
            )
        if algorithm_slow_count > good_count:
            raise CheckpointValidationError(
                "known-good slow-commit count exceeds persistent commit count"
            )
        if good_count > good_sequence:
            raise CheckpointValidationError(
                "known-good persistent commit count exceeds its sequence"
            )
        if not isinstance(algorithm.get("rng"), RNGSnapshot):
            raise CheckpointValidationError("known-good RNG snapshot is missing")
        pending = [str(value) for value in state.get("pending_commit_ids", ())]
        if len(pending) != len(set(pending)) or any(not value for value in pending):
            raise CheckpointValidationError("pending canary commit IDs must be unique")
        pending_sequences: List[int] = []
        for commit_id in pending:
            match = re.fullmatch(
                r"commit-(\d+)(?:-attempt-\d{8})?",
                commit_id,
            )
            if match is None:
                raise CheckpointValidationError(
                    "pending canary IDs must use commit-XXXXXXXX with an optional attempt suffix"
                )
            pending_sequences.append(int(match.group(1)))
        if pending_sequences != sorted(pending_sequences):
            raise CheckpointValidationError("pending canary commit IDs are not monotonic")
        if any(sequence <= good_sequence for sequence in pending_sequences):
            raise CheckpointValidationError(
                "pending canary commit ID does not follow known-good"
            )
        if validate_runtime and any(
            sequence > self._commit_sequence for sequence in pending_sequences
        ):
            raise CheckpointValidationError(
                "pending canary commit ID exceeds commit high-water mark"
            )
        self._canary_known_good = copy.deepcopy(dict(known_good))
        self._canary_pending_commit_ids = pending
        last_rollback = state.get("last_rollback")
        self._canary_last_rollback = (
            None if last_rollback is None else copy.deepcopy(dict(last_rollback))
        )

    def _restore_canary_known_good_algorithm(self) -> None:
        if self._canary_known_good is None:
            raise FDPSCIntegrationError("periodic canary rollback has no known-good state")
        state = self._canary_known_good["algorithm_state"]
        if not isinstance(state, Mapping):
            raise FDPSCIntegrationError("known-good algorithm state is invalid")
        self._adapter_participant.load_state_dict(copy.deepcopy(state["adapters"]))
        self.replay.load_state_dict(copy.deepcopy(state["replay"]))
        self.subspaces.load_state_dict(copy.deepcopy(state["activation_subspaces"]))
        self.exception_router.load_state_dict(copy.deepcopy(state["exception_router"]))
        self._history_gradients = {
            str(key): value.to(self.injection.adapters[str(key)]._reference().device)
            for key, value in state.get("history_gradients", {}).items()
        }
        self._anchor_gradients = {
            str(key): value.to(self.injection.adapters[str(key)]._reference().device)
            for key, value in state.get("anchor_gradients", {}).items()
        }
        repair_state = state.get("repair")
        if self.repair_engine is not None and repair_state is not None:
            self.repair_engine.load_state_dict(copy.deepcopy(repair_state))
        self._successful_slow_commit_count = int(
            state["successful_slow_commit_count"]
        )
        rng = state.get("rng")
        if not isinstance(rng, RNGSnapshot):
            raise FDPSCIntegrationError("known-good RNG snapshot is invalid")
        rng.restore()
        self._route = None
        self._episode_start_repair_state = (
            copy.deepcopy(self.repair_engine.state_dict())
            if self.repair_engine is not None
            else None
        )
        self.assert_base_frozen()

    def _canary_persistent_state(self) -> Dict[str, Any]:
        """Return a detached state bundle suitable for an isolated rollout worker.

        The bundle intentionally contains no external-data registry or query
        token.  A canary evaluator receives only model/persistent-memory state
        plus the independent fixed scenario supplied by :mod:`fd_psc.canary`.
        """

        return {
            "schema_version": 1,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "episode_sequence": self._episode_sequence,
            "commit_sequence": self._commit_sequence,
            "successful_slow_commit_count": self._successful_slow_commit_count,
            "context_identifier": self._active_context,
            "adapters": self._adapter_participant.state_dict(),
            "replay": self.replay.state_dict(),
            "activation_subspaces": self.subspaces.state_dict(),
            "exception_router": self.exception_router.state_dict(),
            "history_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._history_gradients.items())
            },
            "anchor_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._anchor_gradients.items())
            },
        }

    def _persistent_rank_map(
        self,
        proposal_type: ProposalType,
    ) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            factors = (
                adapter.get_slow_factors()
                if proposal_type == ProposalType.GLOBAL_SLOW
                else adapter.get_exception_factors()
            )
            result[logical_id] = int(factors.A.shape[0])
        return result

    @staticmethod
    def _canary_high_risk(candidate: _Candidate) -> bool:
        return bool(
            candidate.repair_step is not None
            or candidate.spectral_variant == "spectral_surgery"
        )

    def _run_canary_phase(
        self,
        *,
        phase: CanaryPhase,
        candidate: _Candidate,
        before_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        before_ranks: Mapping[str, int],
        commit_sequence: int,
        attempted_commit_id: Optional[str] = None,
        reverted_commit_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        if self.canary_runner is None:
            return {"status": "disabled", "phase": phase.value}
        trigger = self.canary_scheduler.decide(
            phase=phase,
            episode_count=self._episode_sequence,
            commit_sequence=commit_sequence,
            before_ranks=before_ranks,
            candidate_ranks=candidate.selected_rank_by_layer,
            high_risk_commit=self._canary_high_risk(candidate),
            exception_merge=(candidate.proposal_type == ProposalType.REPLACE_EXCEPTION),
        )
        if (
            phase is CanaryPhase.POST_COMMIT
            and self.canary_runner is not None
            and self._canary_known_good is not None
            and self._episode_sequence
            - int(self._canary_known_good.get("episode_sequence", 0))
            >= self.canary_scheduler.every_episodes
            and not trigger.should_run
        ):
            # If the exact K-th episode had no accepted proposal, run at the
            # first later commit instead of silently stretching the rollback
            # period to the next modulo boundary.
            trigger = CanaryTriggerDecision(
                should_run=True,
                phase=phase,
                episode_count=self._episode_sequence,
                commit_sequence=int(commit_sequence),
                reasons=("periodic",),
                rank_expanded=trigger.rank_expanded,
            )
        result = self.canary_runner.run(
            trigger,
            before_state=before_state,
            candidate_state=candidate_state,
        )
        report = result.to_dict()
        self.metrics.record(
            "canary_status",
            result.status.value,
            episode_id=self._active_episode_id,
            context_identifier=self._active_context,
            tags={"phase": phase.value, "reasons": ",".join(trigger.reasons)},
        )
        if (
            result.before_success_rate is not None
            and result.candidate_success_rate is not None
        ):
            self.metrics.record(
                "canary_planning_regression",
                result.before_success_rate - result.candidate_success_rate,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"phase": phase.value, "rollout_pairs": result.completed_pairs},
            )
        if result.status is CanaryStatus.UNRUN:
            self.diagnostics.record(
                "warning",
                "canary_unrun",
                result.reason,
                episode_id=self._active_episode_id,
                details={"phase": phase.value, "limitations": list(result.limitations)},
            )
        if result.status is CanaryStatus.FAIL:
            if phase is CanaryPhase.POST_COMMIT:
                if attempted_commit_id is None:
                    raise FDPSCIntegrationError(
                        "periodic Gate-7 failure is missing its attempted commit ID"
                    )
                raise _PeriodicCanaryFailure(
                    f"Gate-7 {phase.value} canary failed: {result.reason}",
                    report=report,
                    attempted_commit_id=attempted_commit_id,
                    attempted_commit_sequence=commit_sequence,
                    reverted_commit_ids=reverted_commit_ids,
                )
            raise FDPSCIntegrationError(
                f"Gate-7 {phase.value} canary failed: {result.reason}"
            )
        return report

    def _persistent_transaction(self) -> StateTransaction:
        return StateTransaction(
            {
                "adapters": self._adapter_participant,
                "replay": self.replay,
                "subspaces": self.subspaces,
                "exception_router": self.exception_router,
                "canary_period": _CanaryPeriodParticipant(self),
                "counters": _CounterParticipant(self),
                "gradient_references": _GradientReferenceParticipant(self),
            },
            name=f"fd_psc_commit:{self._active_episode_id}",
        )

    def _rollback_periodic_canary_failure(
        self,
        failure: _PeriodicCanaryFailure,
        *,
        query_token_id: str,
        candidate: _Candidate,
        gate_report: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Restore the last certified period and publish an auditable rollback.

        The surrounding per-commit transaction has already restored the
        immediately preceding live state.  This second restore intentionally
        crosses all earlier commits in the failed canary period.  Query and
        gate ledgers, the current episode number, and the commit-ID high-water
        mark are preserved so none of the reverted decisions can be replayed.
        """

        if self._canary_known_good is None:
            raise FDPSCIntegrationError(
                "periodic canary failed without a persisted known-good state"
            )
        known_good_sequence = int(self._canary_known_good["commit_sequence"])
        known_good_count = int(self._canary_known_good["persistent_commit_count"])
        current_episode_sequence = self._episode_sequence
        self._restore_canary_known_good_algorithm()
        self._episode_sequence = current_episode_sequence
        self._commit_sequence = int(failure.attempted_commit_sequence)
        self.state_machine.persistent_commit_count = known_good_count
        self.state_machine.rollback_count += 1
        self._canary_pending_commit_ids = []
        rollback_base_id = f"rollback-{self._commit_sequence:08d}"
        rollback_id = (
            self.checkpoints.next_available_id(rollback_base_id)
            if self.checkpoints is not None
            else rollback_base_id
        )
        rollback_metadata = {
            "schema_version": 1,
            "rollback_id": rollback_id,
            "failed_commit_id": failure.attempted_commit_id,
            "failed_commit_sequence": failure.attempted_commit_sequence,
            "known_good_commit_id": self._canary_known_good.get("commit_id"),
            "known_good_commit_sequence": known_good_sequence,
            "reverted_commit_ids": list(failure.reverted_commit_ids),
            "episode_id": self._active_episode_id,
            "query_token_id": str(query_token_id),
            "canary": copy.deepcopy(failure.report),
        }
        self._canary_last_rollback = copy.deepcopy(rollback_metadata)
        if self.state_machine.state == FDPSCState.FINAL_GATE:
            self.state_machine.reject_query(
                f"periodic_canary_rollback:{rollback_id}:{failure}"
            )
        self.metrics.increment(
            "rollback_count",
            episode_id=self._active_episode_id,
        )
        self.metrics.record(
            "canary_period_reverted_commit_count",
            len(failure.reverted_commit_ids),
            episode_id=self._active_episode_id,
            tags={"scope": "canary_period", "rollback_id": rollback_id},
        )
        self.diagnostics.record(
            "error",
            "periodic_canary_rollback",
            str(failure),
            episode_id=self._active_episode_id,
            details=rollback_metadata,
        )

        checkpoint_reference = None
        checkpoint_error: Optional[Exception] = None
        if self.checkpoints is not None:
            try:
                future_metric_events = (
                    int(self._sleep_started_at is not None)
                    + 5  # candidate/query/exception/proposal-type/terminal metrics
                )
                checkpoint_reference = self.checkpoints.save_committed(
                    self._checkpoint_state(
                        future_metric_events=future_metric_events
                    ),
                    commit_id=rollback_id,
                    commit_sequence=self._commit_sequence,
                    config_identity=self.config.persistence_identity_hash(),
                    journal_metadata={
                        "event_type": "periodic_canary_rollback",
                        **rollback_metadata,
                        "proposal_type": candidate.proposal_type.value,
                        "gate_passed": bool(gate_report.passed),
                    },
                )
                self._last_checkpoint_episode_sequence = self._episode_sequence
                self.checkpoints.mark_rolled_back(
                    failure.reverted_commit_ids,
                    rollback_id=rollback_id,
                    rollback_sequence=self._commit_sequence,
                    reason=str(failure),
                    metadata={
                        "episode_id": self._active_episode_id,
                        "known_good_commit_sequence": known_good_sequence,
                    },
                )
            except Exception as exc:  # live memory is already safely restored
                checkpoint_error = exc
                self.diagnostics.record(
                    "error",
                    "canary_rollback_checkpoint_failed",
                    f"{type(exc).__name__}: {exc}",
                    episode_id=self._active_episode_id,
                    details=rollback_metadata,
                )
        else:
            checkpoint_error = FDPSCIntegrationError(
                "periodic canary rollback requires sidecar checkpointing"
            )

        details: Dict[str, Any] = {
            "commit_id": None,
            "failed_commit_id": failure.attempted_commit_id,
            "rollback_commit_id": (
                checkpoint_reference.commit_id
                if checkpoint_reference is not None
                else None
            ),
            "reverted_commit_ids": list(failure.reverted_commit_ids),
            "known_good_commit_sequence": known_good_sequence,
            "canary": {"post_commit": copy.deepcopy(failure.report)},
            "commit_error": f"{type(failure).__name__}: {failure}",
        }
        if checkpoint_error is not None:
            details["rollback_checkpoint_error"] = (
                f"{type(checkpoint_error).__name__}: {checkpoint_error}"
            )
        self.assert_base_frozen()
        return False, details

    def _commit_candidate(
        self,
        trainer: Any,
        candidate: _Candidate,
        *,
        query_token_id: str,
        gate_report: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        windows = self._build_replay_windows(trainer)
        attempted_commit_sequence = self._commit_sequence + 1
        commit_base_id = f"commit-{attempted_commit_sequence:08d}"
        commit_id = (
            self.checkpoints.next_available_id(commit_base_id)
            if self.checkpoints is not None
            else commit_base_id
        )
        checkpoint_reference = None
        try:
            with self._persistent_transaction() as transaction:
                self._load_episode_start()
                before_ranks = self._persistent_rank_map(candidate.proposal_type)
                persistent_before = self._canary_persistent_state()
                self._apply_candidate(candidate)
                pre_commit_canary = self._run_canary_phase(
                    phase=CanaryPhase.PRE_COMMIT,
                    candidate=candidate,
                    before_state=persistent_before,
                    candidate_state=self._canary_persistent_state(),
                    before_ranks=before_ranks,
                    commit_sequence=self._commit_sequence + 1,
                )
                if candidate.proposal_type == ProposalType.GLOBAL_SLOW:
                    self.replay.add_committed_windows(windows, commit_kind="slow")
                    self._update_subspaces_after_slow_commit(trainer, windows)
                    if self.config.gradient_geometry.enabled:
                        gradient_windows = self._balanced_history_gradient_windows(
                            self.replay.windows() or tuple(windows)
                        )
                        self._history_gradients = self._collect_effective_gradients(
                            trainer,
                            self._replay_records(gradient_windows),
                            state="fast",
                            fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                        )
                    if self.config.gradient_geometry.enabled and self.external is not None:
                        anchor_records = self._fixed_external_gradient_records(
                            self._fixed_anchor_records(),
                            batches=self.config.gradient_geometry.anchor_batches,
                            source="post-commit immutable anchor",
                        )
                        self._anchor_gradients = self._collect_effective_gradients(
                            trainer,
                            anchor_records,
                            state="fast",
                            fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                        )
                    # This durable counter, rather than the current replay
                    # length, is the evidence that closes Gate 2 cold start.
                    # It is inside the transaction and is therefore restored
                    # on any canary/checkpoint/commit failure.
                    self._successful_slow_commit_count += 1
                else:
                    exception_state = self._exception_state(candidate)
                    residuals = [window.residual for window in windows if window.residual is not None]
                    context_descriptors = [
                        value.detach().cpu().clone()
                        for value in self._support_context_descriptors
                    ]
                    if not context_descriptors:
                        fallback_descriptor = self._current_context_prototype()
                        if fallback_descriptor is None:
                            raise FDPSCIntegrationError(
                                "exception commit has no accepted support context descriptor"
                            )
                        context_descriptors = [fallback_descriptor]
                    if candidate.proposal_type == ProposalType.REPLACE_EXCEPTION:
                        if self._route is None or self._route.adapter_id is None:
                            raise FDPSCIntegrationError("replacement proposal has no fixed route")
                        self.exception_router.commit_replace(
                            self._route.adapter_id,
                            adapter_state=exception_state,
                            context_descriptors=context_descriptors,
                            residual_descriptors=residuals,
                            local_windows=windows,
                            validation_gain=candidate.calibration_gain,
                            metadata={"episode_id": self._active_episode_id},
                        )
                    else:
                        self.exception_router.commit_new(
                            adapter_state=exception_state,
                            context_descriptors=context_descriptors,
                            residual_descriptors=residuals,
                            local_windows=windows,
                            validation_gain=candidate.calibration_gain,
                            metadata={"episode_id": self._active_episode_id},
                        )
                self._commit_sequence = attempted_commit_sequence
                if self.canary_runner is not None:
                    self._canary_pending_commit_ids.append(commit_id)
                    canary_period_commit_ids = tuple(
                        self._canary_pending_commit_ids
                    )
                    if self._canary_known_good is None:
                        raise FDPSCIntegrationError(
                            "periodic Gate-7 is missing its known-good baseline"
                        )
                    periodic_before = copy.deepcopy(
                        self._canary_known_good["rollout_state"]
                    )
                    periodic_candidate_algorithm = (
                        self._capture_canary_algorithm_state()
                    )
                    periodic_candidate = self._canary_rollout_state_from_algorithm(
                        periodic_candidate_algorithm,
                        episode_sequence=self._episode_sequence,
                        commit_sequence=self._commit_sequence,
                    )
                else:
                    canary_period_commit_ids = ()
                    periodic_before = persistent_before
                    periodic_candidate = self._canary_persistent_state()
                post_commit_canary = self._run_canary_phase(
                    phase=CanaryPhase.POST_COMMIT,
                    candidate=candidate,
                    before_state=periodic_before,
                    candidate_state=periodic_candidate,
                    before_ranks=before_ranks,
                    commit_sequence=self._commit_sequence,
                    attempted_commit_id=commit_id,
                    reverted_commit_ids=canary_period_commit_ids,
                )
                if post_commit_canary.get("status") == CanaryStatus.PASS.value:
                    self._promote_canary_known_good(
                        commit_id=commit_id,
                        commit_sequence=self._commit_sequence,
                        persistent_commit_count=(
                            self.state_machine.persistent_commit_count + 1
                        ),
                    )
                canary = {
                    "pre_commit": dict(pre_commit_canary),
                    "post_commit": dict(post_commit_canary),
                    "period_commit_ids": list(canary_period_commit_ids),
                }
                self.assert_base_frozen()
                # Every persistent mutation is a journaled transaction.  The
                # periodic save cadence may govern non-mutating snapshots in
                # callers, but it can never skip a commit journal/version.
                if self.checkpoints is not None:
                    checkpoint_reference = self.checkpoints.save_committed(
                        self._checkpoint_state(
                            future_metric_events=self._post_checkpoint_metric_event_count(
                                checkpoint_saved=True
                            )
                        ),
                        commit_id=commit_id,
                        commit_sequence=self._commit_sequence,
                        config_identity=self.config.persistence_identity_hash(),
                        journal_metadata={
                            "episode_id": self._active_episode_id,
                            "proposal_type": candidate.proposal_type.value,
                            "query_token_id": query_token_id,
                            "gate_passed": bool(gate_report.passed),
                            "gates": {
                                result.gate: {
                                    "status": result.status.value,
                                    "reason": result.reason,
                                    "metrics": dict(result.metrics),
                                }
                                for result in gate_report.results
                            },
                            "alpha_shared": candidate.alpha_shared,
                            "alpha_safe": candidate.alpha_safe,
                            "selected_rank_by_layer": dict(
                                sorted(candidate.selected_rank_by_layer.items())
                            ),
                            "repair_step": candidate.repair_step,
                            "spectral_variant": candidate.spectral_variant,
                            "slice_activated": bool(self._centered_reason),
                            "sdc_active_layers": sorted(
                                key for key, active in self._sdc_active.items() if active
                            ),
                            "canary": dict(canary),
                            "canary_known_good_commit_id": (
                                None
                                if self._canary_known_good is None
                                else self._canary_known_good.get("commit_id")
                            ),
                            "canary_pending_commit_ids": list(
                                self._canary_pending_commit_ids
                            ),
                            "canary_period_commit_ids": list(
                                canary_period_commit_ids
                            ),
                        },
                    )
                    self._last_checkpoint_episode_sequence = self._episode_sequence
                transaction.commit()
            if candidate.proposal_type == ProposalType.GLOBAL_SLOW:
                self.state_machine.commit_slow()
            else:
                self.state_machine.commit_exception()
            self.metrics.record(
                "replay_memory_bytes",
                _tensor_tree_nbytes(self.replay.state_dict()),
                episode_id=self._active_episode_id,
            )
            if checkpoint_reference is not None and self.checkpoints is not None:
                checkpoint_path = (
                    self.checkpoints.state_directory / checkpoint_reference.version_file
                )
                self.metrics.record(
                    "checkpoint_bytes",
                    checkpoint_path.stat().st_size,
                    episode_id=self._active_episode_id,
                    tags={"commit_id": commit_id},
                )
            return True, {"commit_id": commit_id, "canary": dict(canary)}
        except _PeriodicCanaryFailure as exc:
            return self._rollback_periodic_canary_failure(
                exc,
                query_token_id=query_token_id,
                candidate=candidate,
                gate_report=gate_report,
            )
        except Exception as exc:
            # StateTransaction restores adapters/replay/Q/router/counters and
            # every process RNG. Query authorization remains consumed and the
            # episode is terminal: there is no alternate proposal.
            if self.state_machine.state == FDPSCState.FINAL_GATE:
                self.state_machine.reject_query(
                    f"atomic_commit_failed:{type(exc).__name__}:{exc}"
                )
            self.diagnostics.record(
                "error",
                "atomic_commit_failed",
                str(exc),
                episode_id=self._active_episode_id,
            )
            return False, {
                "commit_id": None,
                "commit_error": f"{type(exc).__name__}: {exc}",
            }

    def _commit_accumulate_baseline(self, trainer: Any) -> Dict[str, Any]:
        """Baseline 4: keep the same Pilot parameters and do not sleep."""

        return self.finish_episode_without_sleep("accumulate_continuous_adapter")

    def _commit_plain_svd_baseline(self) -> Dict[str, Any]:
        """Baseline 5: ordinary fixed-rank factor SVD, no gates or banks."""

        self._sleep_started_at = time.perf_counter()
        factors: Dict[str, LowRankFactors] = {}
        tasks: Dict[str, LowRankFactors] = {}
        ranks: Dict[str, int] = {}
        errors: Dict[str, float] = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            slow = self._canonical(adapter.get_slow_factors())
            task = self._canonical(adapter.get_episodic_factors())
            merged = concatenate_factors((slow, task))
            rank = min(
                self.config.slow_lora.initial_rank,
                self.config.slow_lora.maximum_rank,
                merged.out_features,
                merged.in_features,
            )
            factors[logical_id] = truncate_factors(
                merged,
                rank,
                dtype=slow.B.dtype,
            )
            tasks[logical_id] = task
            ranks[logical_id] = rank
            errors[logical_id] = 0.0
        candidate = _Candidate(
            ProposalType.GLOBAL_SLOW,
            factors,
            tasks,
            errors,
            ranks,
            1.0,
            1.0,
            "plain_factor_svd",
            screening_reason="baseline_no_gate_no_bank",
        )
        proposal = FinalProposal(
            f"{self._active_episode_id}:plain-svd",
            ProposalType.GLOBAL_SLOW,
            candidate,
            {"baseline": "episodic_slow_plain_svd"},
        )
        attempted_commit_sequence = self._commit_sequence + 1
        commit_base_id = f"commit-{attempted_commit_sequence:08d}"
        commit_id = (
            self.checkpoints.next_available_id(commit_base_id)
            if self.checkpoints is not None
            else commit_base_id
        )
        checkpoint_reference = None
        with StateTransaction(
            {
                "adapters": self._adapter_participant,
                "counters": _CounterParticipant(self),
                "state_machine": self.state_machine,
            },
            name=f"plain_svd:{self._active_episode_id}",
        ) as transaction:
            self.state_machine.enter_sleep()
            self._load_episode_start()
            self._apply_candidate(candidate)
            self._commit_sequence = attempted_commit_sequence
            self.state_machine.commit_baseline_slow(
                proposal,
                "explicit_plain_svd_baseline_no_gate",
            )
            self.assert_base_frozen()
            if self.checkpoints is not None:
                checkpoint_reference = self.checkpoints.save_committed(
                    self._checkpoint_state(
                        future_metric_events=self._post_checkpoint_metric_event_count(
                            checkpoint_saved=True
                        )
                    ),
                    commit_id=commit_id,
                    commit_sequence=self._commit_sequence,
                    config_identity=self.config.persistence_identity_hash(),
                    journal_metadata={
                        "event_type": "baseline_commit",
                        "episode_id": self._active_episode_id,
                        "proposal_type": candidate.proposal_type.value,
                        "baseline": "episodic_slow_plain_svd",
                    },
                )
            transaction.commit()
        if checkpoint_reference is not None:
            self._last_checkpoint_episode_sequence = self._episode_sequence
        self.metrics.record(
            "replay_memory_bytes",
            _tensor_tree_nbytes(self.replay.state_dict()),
            episode_id=self._active_episode_id,
        )
        if checkpoint_reference is not None and self.checkpoints is not None:
            checkpoint_path = (
                self.checkpoints.state_directory / checkpoint_reference.version_file
            )
            self.metrics.record(
                "checkpoint_bytes",
                checkpoint_path.stat().st_size,
                episode_id=self._active_episode_id,
                tags={"commit_id": commit_id, "baseline": "plain_svd"},
            )
        outcome = self.state_machine.last_outcome
        self._finish_terminal_episode()
        return {
            "fd_psc_outcome": outcome,
            "run_mode": self.config.run_mode,
            "commit_query_access_count": 0,
            "candidate_count": 1,
            "committed": True,
            "proposal_type": ProposalType.GLOBAL_SLOW.value,
            "commit_id": commit_id,
        }

    def end_episode_and_sleep(
        self,
        trainer: Any,
        obs_seqs: Sequence[Any],
        act_seqs: Sequence[Any],
    ) -> Dict[str, Any]:
        self.require_active_episode()
        if self._online_mode_depth:
            raise FDPSCIntegrationError("cannot enter sleep during an optimizer update")
        if self.config.run_mode == "episodic_reset":
            return self.finish_episode_without_sleep("episodic_reset_baseline")
        if self.config.run_mode == "accumulate":
            return self._commit_accumulate_baseline(trainer)
        if self.config.run_mode == "plain_svd":
            return self._commit_plain_svd_baseline()
        self._sleep_started_at = time.perf_counter()
        self.state_machine.enter_sleep()
        try:
            if not self._support_segments or self.state_machine.online_update_count == 0:
                self.state_machine.reject_no_proposal("empty_or_unadapted_support")
                outcome = self.state_machine.last_outcome
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": 0,
                    "candidate_count": 0,
                }
            if not self._eligible_replay_segments():
                self.state_machine.reject_no_proposal(
                    "support_window_shorter_than_num_hist_plus_num_pred"
                )
                outcome = self.state_machine.last_outcome
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": 0,
                    "candidate_count": 0,
                }
            if self._current_context_prototype() is None:
                self.state_machine.reject_no_proposal(
                    "unavailable_frozen_context_descriptor"
                )
                outcome = self.state_machine.last_outcome
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": 0,
                    "candidate_count": 0,
                }
            task_energy = sum(
                float(factor_frobenius_norm_sq(self._canonical(adapter.get_episodic_factors())).detach().cpu())
                for adapter in self.injection.adapters.values()
            )
            if task_energy <= self.config.gates.absolute_numerical_tolerance ** 2:
                self.state_machine.reject_no_proposal("zero_episodic_task_vector")
                outcome = self.state_machine.last_outcome
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": 0,
                    "candidate_count": 0,
                }
            if self.external is None:
                raise FDPSCIntegrationError("sleep calibration requires the fixed external registry")

            calibration = self.external.calibration(self._active_context)
            anchor = (
                self._fixed_anchor_records()
                if self.config.gates.anchor_enabled
                else ()
            )
            proposal_type = (
                ProposalType.REPLACE_EXCEPTION
                if self._route is not None and self._route.adapter_id is not None
                else ProposalType.GLOBAL_SLOW
            )
            if proposal_type == ProposalType.REPLACE_EXCEPTION:
                assert self._route is not None and self._route.adapter_id is not None
                # A routed exception is repaired/screened only against its own
                # bounded local history.  Global slow and global replay remain
                # read-only for the whole episode.
                history_windows = list(
                    self.exception_router.get(
                        self._route.adapter_id
                    ).local_replay.windows()
                )
            else:
                history_windows = list(self.replay.windows())
            calibration_before = self._evaluate_state(
                trainer, calibration, state="before"
            )
            calibration_fast = self._evaluate_state(
                trainer, calibration, state="fast"
            )
            self.metrics.record(
                "per_context_loss",
                calibration_before,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"split": "external_calibration", "state": "before"},
            )
            self.metrics.record(
                "per_context_loss",
                calibration_fast,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"split": "external_calibration", "state": "fast"},
            )
            history_before = (
                self._evaluate_state(
                    trainer,
                    self._replay_records(history_windows),
                    state="before",
                )
                if history_windows and self.config.gates.history_enabled
                else None
            )
            anchor_before = (
                self._evaluate_state(trainer, anchor, state="before")
                if anchor
                else None
            )
            plasticity_before = self._plasticity_gain(trainer, state="before")
            fast_gain_cal = calibration_before - calibration_fast
            self.metrics.record(
                "fast_adaptation_gain",
                fast_gain_cal,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"split": "external_calibration"},
            )
            self.metrics.record(
                "external_calibration_gain",
                fast_gain_cal,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"before_loss": calibration_before, "fast_loss": calibration_fast},
            )
            if history_before is not None:
                self.metrics.record(
                    "historical_replay_loss",
                    history_before,
                    episode_id=self._active_episode_id,
                    tags={"state": "before"},
                )
            if anchor_before is not None:
                self.metrics.record(
                    "anchor_loss",
                    anchor_before,
                    episode_id=self._active_episode_id,
                    tags={"state": "before"},
                )
            if plasticity_before is not None:
                self.metrics.record(
                    "plasticity_gain",
                    plasticity_before,
                    episode_id=self._active_episode_id,
                    tags={"state": "before"},
                )
            activations = self._collect_activations(
                trainer, calibration, state="fast"
            )
            gradient_pruning_enabled = (
                self.config.merge.soft_ness_enabled
                and self._merge_signal_has_threshold("gradient")
            )
            residual_pruning_enabled = (
                self.config.merge.soft_ness_enabled
                and self._merge_signal_has_threshold("residual")
            )
            gradient_pruning_data_ready = (
                gradient_pruning_enabled
                and bool(history_windows)
                and len(calibration)
                >= (
                    self.config.gradient_geometry.current_batches
                    * self.config.gradient_geometry.windows_per_batch
                )
            )
            current_gradient: Dict[str, Tensor] = {}
            history_gradient: Dict[str, Tensor] = {}
            anchor_gradient: Dict[str, Tensor] = {}
            surgery_gradients: Dict[str, Tensor] = {}
            if self.config.spectral_surgery.enabled or gradient_pruning_data_ready:
                calibration_gradient_records = self._fixed_external_gradient_records(
                    calibration,
                    batches=self.config.gradient_geometry.current_batches,
                    source="merge/spectral calibration",
                )
                current_gradient = self._collect_effective_gradients(
                    trainer,
                    calibration_gradient_records,
                    state="fast",
                    fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                )
                if history_windows:
                    sampled_history = self._balanced_history_gradient_windows(history_windows)
                    history_gradient = self._collect_effective_gradients(
                        trainer,
                        self._replay_records(sampled_history),
                        state="fast",
                        fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                    )
                if (
                    self.config.spectral_surgery.enabled
                    and self.config.spectral_surgery.anchor_weight > 0
                ):
                    anchor_gradient_records = self._fixed_external_gradient_records(
                        self._fixed_anchor_records(),
                        batches=self.config.gradient_geometry.anchor_batches,
                        source="spectral anchor",
                    )
                    anchor_gradient = self._collect_effective_gradients(
                        trainer,
                        anchor_gradient_records,
                        state="fast",
                        fixed_batch_size=self.config.gradient_geometry.windows_per_batch,
                    )
            if self.config.spectral_surgery.enabled:
                for logical_id in sorted(current_gradient):
                    value = (
                        self.config.spectral_surgery.current_weight
                        * current_gradient[logical_id]
                    )
                    if logical_id in history_gradient:
                        value = value + (
                            self.config.spectral_surgery.history_weight
                            * history_gradient[logical_id]
                        )
                    if logical_id in anchor_gradient:
                        value = value + (
                            self.config.spectral_surgery.anchor_weight
                            * anchor_gradient[logical_id]
                        )
                    surgery_gradients[logical_id] = value.detach().clone()
            task_variants = self._task_variants(surgery_gradients)

            # Similarity signals are computed entirely from fixed calibration,
            # already-committed relevant history, and accepted support.  The
            # commit-query split has not been opened and is not reachable here.
            current_replay_windows = (
                self._build_replay_windows(trainer)
                if residual_pruning_enabled and history_windows
                else ()
            )
            if self.config.merge.soft_ness_enabled:
                pruning_signals = self._merge_similarity_signals(
                    current_gradients=current_gradient,
                    history_gradients=history_gradient,
                    current_windows=current_replay_windows,
                    history_windows=history_windows,
                )
                coefficient_decision = self._pruned_coefficient_grid(pruning_signals)
            else:
                pruning_signals = _MergeSimilaritySignals()
                coefficient_decision = _CoefficientGridDecision(
                    tuple(float(value) for value in self.config.merge.shared_coefficients),
                    "soft_ness_disabled_full_task",
                    {},
                )
            for signal_name, signal_value in (
                ("gradient", pruning_signals.gradient),
                ("context", pruning_signals.context),
                ("residual", pruning_signals.residual),
            ):
                self.metrics.record(
                    f"merge_{signal_name}_similarity",
                    signal_value,
                    episode_id=self._active_episode_id,
                    context_identifier=self._active_context,
                    tags={
                        "decision": coefficient_decision.signal_decisions.get(
                            signal_name,
                            "unused",
                        )
                    },
                )
            self.metrics.record(
                "merge_shared_coefficient_count",
                len(coefficient_decision.shared_coefficients),
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
                tags={"reason": coefficient_decision.reason},
            )

            candidates = self._make_candidates(
                proposal_type,
                activations,
                task_variants,
                shared_coefficients=coefficient_decision.shared_coefficients,
            )
            selected, reports = self._screen_candidates(
                trainer,
                candidates,
                calibration=calibration,
                calibration_before=calibration_before,
                calibration_fast=calibration_fast,
                history_windows=history_windows,
                history_before=history_before,
                anchor=anchor,
                anchor_before=anchor_before,
                plasticity_before=plasticity_before,
            )
            all_candidate_count = len(candidates)
            self._calibration_candidate_count = all_candidate_count
            repair_seed = min(
                candidates,
                key=lambda item: (
                    item.calibration_loss,
                    item.alpha_shared,
                    item.alpha_safe,
                ),
                default=None,
            )
            # Path B always follows a failed path A.  A new exception (path C)
            # is not even generated until the bounded global repair has also
            # failed, so exception creation can never replace an available slow
            # consolidation merely because its quick candidate ranked worse.
            if selected is None:
                repaired = self._repair_candidate(
                    trainer,
                    repair_seed,
                    calibration=calibration,
                    calibration_before=calibration_before,
                    calibration_fast=calibration_fast,
                    history_windows=history_windows,
                    history_before=history_before,
                    anchor=anchor,
                    anchor_before=anchor_before,
                    plasticity_before=plasticity_before,
                )
                if repaired is not None:
                    selected = repaired
                    reports.append(repaired.summary())

            exception_eligible = (
                fast_gain_cal > self.config.gates.absolute_numerical_tolerance
                and fast_gain_cal
                + self.config.gates.absolute_numerical_tolerance
                >= self.config.exception.minimum_calibration_fast_gain
            )
            if (
                selected is None
                and proposal_type == ProposalType.GLOBAL_SLOW
                and self.config.exception.enabled
                and exception_eligible
            ):
                exception_candidates = self._make_candidates(
                    ProposalType.NEW_EXCEPTION,
                    activations,
                    task_variants,
                    shared_coefficients=coefficient_decision.shared_coefficients,
                )
                selected, exception_reports = self._screen_candidates(
                    trainer,
                    exception_candidates,
                    calibration=calibration,
                    calibration_before=calibration_before,
                    calibration_fast=calibration_fast,
                    history_windows=history_windows,
                    history_before=history_before,
                    anchor=anchor,
                    anchor_before=anchor_before,
                    plasticity_before=plasticity_before,
                )
                reports.extend(exception_reports)
                all_candidate_count += len(exception_candidates)
                self._calibration_candidate_count = all_candidate_count

            self.metrics.record(
                "calibration_candidate_count",
                all_candidate_count,
                episode_id=self._active_episode_id,
                context_identifier=self._active_context,
            )
            if selected is not None:
                self.metrics.record(
                    "alpha_shared",
                    selected.alpha_shared,
                    episode_id=self._active_episode_id,
                    tags={"proposal_type": selected.proposal_type.value},
                )
                self.metrics.record(
                    "alpha_safe",
                    selected.alpha_safe,
                    episode_id=self._active_episode_id,
                    tags={"proposal_type": selected.proposal_type.value},
                )
                for logical_id, rank in sorted(selected.selected_rank_by_layer.items()):
                    self.metrics.record(
                        "slow_rank",
                        rank,
                        episode_id=self._active_episode_id,
                        logical_layer_id=logical_id,
                        tags={"proposal_type": selected.proposal_type.value},
                    )
                    self.metrics.record(
                        "functional_error",
                        selected.functional_error_by_layer[logical_id],
                        episode_id=self._active_episode_id,
                        logical_layer_id=logical_id,
                    )
                    selected_factors = selected.factors_by_layer.get(logical_id)
                    rank_reference = selected.rank_reference_by_layer.get(logical_id)
                    if selected_factors is None or rank_reference is None:
                        self.metrics.record_nullable(
                            "spectral_energy",
                            None,
                            status="unavailable",
                            reason="rank_reference_missing",
                            episode_id=self._active_episode_id,
                            logical_layer_id=logical_id,
                            tags={
                                "definition": "retained_factor_spectral_energy_fraction",
                                "proposal_type": selected.proposal_type.value,
                                "state": "final_proposal",
                                "commit_status": "provisional_before_final_gate",
                            },
                        )
                    else:
                        retained_energy = float(
                            factor_frobenius_norm_sq(selected_factors).detach().cpu()
                        )
                        reference_energy = float(
                            factor_frobenius_norm_sq(rank_reference).detach().cpu()
                        )
                        zero_tolerance = (
                            self.config.gates.absolute_numerical_tolerance ** 2
                        )
                        if not (
                            math.isfinite(retained_energy)
                            and math.isfinite(reference_energy)
                        ):
                            self.metrics.record_nullable(
                                "spectral_energy",
                                None,
                                status="unavailable",
                                reason="nonfinite_factor_energy",
                                episode_id=self._active_episode_id,
                                logical_layer_id=logical_id,
                                tags={
                                    "definition": "retained_factor_spectral_energy_fraction",
                                    "proposal_type": selected.proposal_type.value,
                                    "state": "final_proposal",
                                    "commit_status": "provisional_before_final_gate",
                                },
                            )
                        elif reference_energy <= zero_tolerance:
                            self.metrics.record_nullable(
                                "spectral_energy",
                                1.0,
                                status="available",
                                episode_id=self._active_episode_id,
                                logical_layer_id=logical_id,
                                tags={
                                    "definition": "retained_factor_spectral_energy_fraction",
                                    "proposal_type": selected.proposal_type.value,
                                    "zero_reference": True,
                                    "state": "final_proposal",
                                    "commit_status": "provisional_before_final_gate",
                                },
                            )
                        else:
                            raw_fraction = retained_energy / reference_energy
                            self.metrics.record_nullable(
                                "spectral_energy",
                                max(0.0, min(1.0, raw_fraction)),
                                status="available",
                                episode_id=self._active_episode_id,
                                logical_layer_id=logical_id,
                                tags={
                                    "definition": "retained_factor_spectral_energy_fraction",
                                    "proposal_type": selected.proposal_type.value,
                                    "raw_fraction": raw_fraction,
                                    "state": "final_proposal",
                                    "commit_status": "provisional_before_final_gate",
                                },
                            )
                if selected.worst_context_regression is not None:
                    self.metrics.record(
                        "worst_context_regression",
                        selected.worst_context_regression,
                        episode_id=self._active_episode_id,
                    )

            report_path = self.runtime_output_dir / "fd_psc_candidates" / f"{self._active_episode_id}.json"
            _atomic_json(
                report_path,
                {
                    "schema_version": 1,
                    "episode_id": self._active_episode_id,
                    "context_identifier": self._active_context,
                    "calibration_before": calibration_before,
                    "calibration_fast": calibration_fast,
                    "candidate_count": all_candidate_count,
                    "candidates": reports,
                    "selected": selected.summary() if selected is not None else None,
                    "commit_query_used_for_selection": False,
                },
            )
            if selected is None:
                self.state_machine.reject_no_proposal("calibration_and_repair_rejected_all")
                outcome = self.state_machine.last_outcome
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": 0,
                    "candidate_count": all_candidate_count,
                    "calibration_before": calibration_before,
                    "calibration_fast": calibration_fast,
                }

            proposal_id = f"{self._active_episode_id}:final-proposal"
            final = FinalProposal(
                proposal_id=proposal_id,
                proposal_type=selected.proposal_type,
                payload=selected,
                calibration_metrics={
                    "before_loss": calibration_before,
                    "fast_loss": calibration_fast,
                    "candidate_loss": selected.calibration_loss,
                },
                candidate_count=all_candidate_count,
            )
            self.state_machine.set_final_proposal(final)
            token = self.external.issue_commit_query_token(
                self._active_episode_id,
                proposal_id,
            )
            # Authorize the one final gate before lookup so a missing/corrupt
            # query split can transition to the terminal REJECT_QUERY state.
            self.state_machine.begin_final_gate(proposal_id, token.token_id)
            self.metrics.record(
                "commit_query_gate_invocation_count",
                1,
                episode_id=self._active_episode_id,
                tags={"proposal_id": proposal_id},
            )
            try:
                commit_query = self.external.consume_commit_query(
                    token,
                    proposal_id=proposal_id,
                    context_identifier=self._active_context,
                )
                query_before = self._evaluate_state(
                    trainer, commit_query, state="before"
                )
                query_fast = self._evaluate_state(
                    trainer, commit_query, state="fast"
                )
                query_candidate = self._evaluate_state(
                    trainer,
                    commit_query,
                    state="candidate",
                    candidate=selected,
                )
                self.metrics.record(
                    "commit_query_gain",
                    query_before - query_candidate,
                    episode_id=self._active_episode_id,
                    tags={
                        "fast_gain": query_before - query_fast,
                        "proposal_type": selected.proposal_type.value,
                    },
                )
            except Exception as exc:
                self.state_machine.reject_query(
                    f"commit_query_failure:{type(exc).__name__}:{exc}"
                )
                outcome = self.state_machine.last_outcome
                query_count = self.external.commit_query_access_count(self._active_episode_id)
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": query_count,
                    "candidate_count": all_candidate_count,
                    "commit_error": f"{type(exc).__name__}: {exc}",
                }

            history_candidate = (
                self._evaluate_state(
                    trainer,
                    self._replay_records(history_windows),
                    state="candidate",
                    candidate=selected,
                )
                if history_windows and self.config.gates.history_enabled
                else None
            )
            before_by_context = (
                self._evaluate_by_context(trainer, history_windows, state="before")
                if history_windows and self.config.gates.history_enabled
                else {}
            )
            candidate_by_context = (
                self._evaluate_by_context(
                    trainer,
                    history_windows,
                    state="candidate",
                    candidate=selected,
                )
                if history_windows and self.config.gates.history_enabled
                else {}
            )
            anchor_candidate = (
                self._evaluate_state(
                    trainer,
                    anchor,
                    state="candidate",
                    candidate=selected,
                )
                if anchor
                else None
            )
            plasticity_candidate = self._plasticity_gain(
                trainer,
                state="candidate",
                candidate=selected,
            )
            self._emit_context_retention_metrics(
                before_by_context,
                candidate_by_context,
            )
            self._emit_plasticity_gate_ratio(
                plasticity_before,
                plasticity_candidate,
            )
            if history_candidate is not None:
                self.metrics.record(
                    "historical_replay_loss",
                    history_candidate,
                    episode_id=self._active_episode_id,
                    tags={"state": "candidate"},
                )
            if anchor_candidate is not None:
                self.metrics.record(
                    "anchor_loss",
                    anchor_candidate,
                    episode_id=self._active_episode_id,
                    tags={"state": "candidate"},
                )
            if plasticity_candidate is not None:
                self.metrics.record(
                    "plasticity_gain",
                    plasticity_candidate,
                    episode_id=self._active_episode_id,
                    tags={"state": "candidate"},
                )
            before_factors = self._factor_maps_for_state(
                proposal_type=selected.proposal_type
            )
            candidate_factors = self._factor_maps_for_state(candidate=selected)
            drift_before_by_layer = self._drift_by_layer(before_factors)
            drift_candidate_by_layer = self._drift_by_layer(candidate_factors)
            for logical_id, drift in sorted(drift_candidate_by_layer.items()):
                self.metrics.record(
                    "spectral_drift",
                    drift,
                    episode_id=self._active_episode_id,
                    logical_layer_id=logical_id,
                    tags={
                        "state": "candidate",
                        "increase": drift - drift_before_by_layer[logical_id],
                    },
                )
            gate_inputs = CommitGateInputs(
                proposal_type=selected.proposal_type.value,
                before_commit_loss=query_before,
                fast_commit_loss=query_fast,
                candidate_commit_loss=query_candidate,
                historical_replay_exists=(
                    self._successful_slow_commit_count > 0
                ),
                before_history_loss=history_before,
                candidate_history_loss=history_candidate,
                before_history_by_context=before_by_context,
                candidate_history_by_context=candidate_by_context,
                before_anchor_loss=anchor_before,
                candidate_anchor_loss=anchor_candidate,
                plasticity_before_gain=plasticity_before,
                plasticity_candidate_gain=plasticity_candidate,
                functional_error_by_layer=selected.functional_error_by_layer,
                drift_before_by_layer=drift_before_by_layer,
                drift_candidate_by_layer=drift_candidate_by_layer,
            )
            gate_report = self.gates.evaluate_once(
                episode_id=self._active_episode_id,
                proposal_id=proposal_id,
                query_token_id=token.token_id,
                inputs=gate_inputs,
            )
            gate_results = {
                result.gate: {
                    "status": result.status.value,
                    "reason": result.reason,
                    "metrics": dict(result.metrics),
                }
                for result in gate_report.results
            }
            if not gate_report.passed:
                reason = ",".join(
                    result.gate
                    for result in gate_report.results
                    if result.status.value == "fail"
                )
                self.state_machine.reject_query(f"commit_gates_failed:{reason}")
                outcome = self.state_machine.last_outcome
                query_count = self.external.commit_query_access_count(self._active_episode_id)
                self._finish_terminal_episode()
                return {
                    "fd_psc_outcome": outcome,
                    "run_mode": self.config.run_mode,
                    "commit_query_access_count": query_count,
                    "candidate_count": all_candidate_count,
                    "proposal_type": selected.proposal_type.value,
                    "gates": gate_results,
                }

            committed, commit_details = self._commit_candidate(
                trainer,
                selected,
                query_token_id=token.token_id,
                gate_report=gate_report,
            )
            outcome = self.state_machine.last_outcome
            query_count = self.external.commit_query_access_count(self._active_episode_id)
            self._finish_terminal_episode()
            return {
                "fd_psc_outcome": outcome,
                "run_mode": self.config.run_mode,
                "commit_query_access_count": query_count,
                "candidate_count": all_candidate_count,
                "proposal_type": selected.proposal_type.value,
                "selected_rank_total": sum(selected.selected_rank_by_layer.values()),
                "gates": gate_results,
                "committed": committed,
                **commit_details,
            }
        except BaseException as exc:
            self.diagnostics.record(
                "error",
                "sleep_exception",
                f"{type(exc).__name__}: {exc}",
                episode_id=self._active_episode_id,
            )
            self.abort_episode(f"sleep_exception:{type(exc).__name__}:{exc}")
            raise

    # ------------------------------------------------------------------
    # Cleanup, abort, checkpoint/resume, and final report-test evaluation
    # ------------------------------------------------------------------
    def _reset_adapter_ephemeral(self) -> None:
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            generator = _generator(
                adapter._reference().device,
                _stable_seed(
                    self.config.seed,
                    "idle-reset",
                    self._episode_sequence,
                    logical_id,
                ),
            )
            adapter.reset_episode(generator=generator)
            adapter.clear_active_exception()
            for parameter in adapter.trainable_episode_parameters():
                parameter.requires_grad_(False)

    def _clear_episode_fields(self) -> None:
        self._active_episode_id = None
        self._active_context = None
        self._episode_metadata = {}
        self._context_embedding = None
        self._route = None
        self._support_segments = []
        self._support_context_descriptors = []
        self._support_hashes = set()
        self._support_transition_cursor = 0
        self._episode_start_adapter_states = None
        self._episode_start_repair_state = None
        self._plasticity_probe_rng = None
        self._replan_index = 0
        self._online_step_count = 0
        self._online_loss = 0.0
        self._episode_early_losses = []
        self._episode_start_persistent_commit_count = 0
        self._jepa_loss_threshold = None
        self._jepa_loss_threshold_reason = "threshold_not_provided"
        self._loss_threshold_reached = False
        self._centered_reason = None
        self._calibration_candidate_count = 0
        self._latest_online_gradients = {}
        self._latest_corrected_gradients = {}
        self._latest_gradient_cosines = {}
        self._sdc_anchor_records = ()
        self._sdc_anchor_before_loss = None
        self._sdc_anchor_current_loss = None
        self._sdc_anchor_regression_value = None
        self._online_mode_depth = 0
        self._online_update_started_at = None
        self._sleep_started_at = None

    def _flush_metrics(self) -> None:
        self.metrics.write_jsonl(self.runtime_output_dir / "fd_psc_metrics.jsonl")
        self.metrics.write_csv(self.runtime_output_dir / "fd_psc_metrics.csv")

    def _emit_final_proposal_type(
        self,
        *,
        episode_id: str,
        proposal: Optional[Any],
        terminal_path: str,
    ) -> None:
        proposal_type = (
            None if proposal is None else proposal.proposal_type.value
        )
        self.metrics.record_nullable(
            "final_proposal_type",
            proposal_type,
            status="available" if proposal_type is not None else "not_applicable",
            reason=None if proposal_type is not None else "no_final_proposal",
            episode_id=episode_id,
            context_identifier=self._active_context,
            tags={
                "final_proposal_count": self.state_machine.final_proposal_count,
                "terminal_path": str(terminal_path),
            },
        )

    def _finish_terminal_episode(self) -> None:
        episode_id = self._active_episode_id
        if episode_id is None:
            raise FDPSCIntegrationError("terminal cleanup has no active episode")
        context_identifier = self._active_context
        final_proposal = self.state_machine.final_proposal
        if self.external is not None:
            self.external.end_episode()
        if self.exception_router.active_adapter_id is not None or self._route is not None:
            self.exception_router.end_episode(episode_id)
        if (
            self.repair_engine is not None
            and self._episode_start_repair_state is not None
            and self.state_machine.last_outcome
            not in {FDPSCState.COMMIT_SLOW.value, FDPSCState.COMMIT_EXCEPTION.value}
        ):
            self.repair_engine.load_state_dict(
                copy.deepcopy(self._episode_start_repair_state)
            )
        self._reset_adapter_ephemeral()
        self.state_machine.finish_episode()
        if self._sleep_started_at is not None:
            self.metrics.record(
                "sleep_latency_s",
                time.perf_counter() - self._sleep_started_at,
                episode_id=episode_id,
                context_identifier=self._active_context,
            )
            self._sleep_started_at = None
        self.metrics.record(
            "calibration_candidate_count",
            self._calibration_candidate_count,
            episode_id=episode_id,
            context_identifier=self._active_context,
            tags={"scope": "episode_final"},
        )
        self.metrics.record(
            "commit_query_gate_invocation_count",
            self.state_machine.final_gate_count,
            episode_id=episode_id,
            tags={"scope": "episode_final"},
        )
        self.metrics.record(
            "exception_count",
            len(self.exception_router),
            episode_id=episode_id,
        )
        self._emit_final_proposal_type(
            episode_id=episode_id,
            proposal=final_proposal,
            terminal_path="sleep",
        )
        self.metrics.record(
            "episode_terminal",
            self.state_machine.last_outcome,
            episode_id=episode_id,
            context_identifier=self._active_context,
            tags={"reason": self.state_machine.last_reason},
        )
        outcome = str(self.state_machine.last_outcome or "UNKNOWN")
        reason = str(self.state_machine.last_reason or "")
        self._clear_episode_fields()
        self.assert_base_frozen()
        self._save_episode_snapshot_if_due(
            episode_id=str(episode_id),
            context_identifier=context_identifier,
            outcome=outcome,
            reason=reason,
        )
        self._flush_metrics()

    def finish_episode_without_sleep(self, reason: str) -> Dict[str, Any]:
        self.require_active_episode()
        episode_id = str(self._active_episode_id)
        context_identifier = self._active_context
        if self.external is not None:
            self.external.end_episode()
        self.exception_router.end_episode(episode_id)
        if self.repair_engine is not None and self._episode_start_repair_state is not None:
            self.repair_engine.load_state_dict(
                copy.deepcopy(self._episode_start_repair_state)
            )
        if self.config.run_mode != "accumulate":
            self._reset_adapter_ephemeral()
        else:
            for adapter in self.injection.adapters.values():
                adapter.clear_active_exception()
                for parameter in adapter.trainable_episode_parameters():
                    parameter.requires_grad_(False)
        self.state_machine.finish_without_sleep(str(reason))
        self.metrics.record(
            "episode_no_sleep",
            1,
            episode_id=episode_id,
            context_identifier=self._active_context,
            tags={"reason": str(reason)},
        )
        self._emit_final_proposal_type(
            episode_id=episode_id,
            proposal=None,
            terminal_path="no_sleep",
        )
        self._clear_episode_fields()
        self.assert_base_frozen()
        self._save_episode_snapshot_if_due(
            episode_id=episode_id,
            context_identifier=context_identifier,
            outcome="NO_SLEEP",
            reason=str(reason),
        )
        self._flush_metrics()
        return {
            "fd_psc_outcome": "NO_SLEEP",
            "run_mode": self.config.run_mode,
            "commit_query_access_count": 0,
            "reason": str(reason),
        }

    def _abort_components(self, reason: str) -> None:
        episode_id = self._active_episode_id
        final_proposal = self.state_machine.final_proposal
        if self._online_hooks is not None:
            self._online_hooks.close()
            self._online_hooks = None
        if self.external is not None:
            self.external.end_episode()
        if episode_id is not None:
            try:
                self.exception_router.end_episode(episode_id)
            except Exception:
                pass
        if self._episode_start_adapter_states is not None:
            self._adapter_participant.load_state_dict(
                copy.deepcopy(self._episode_start_adapter_states)
            )
        if self.repair_engine is not None and self._episode_start_repair_state is not None:
            self.repair_engine.load_state_dict(
                copy.deepcopy(self._episode_start_repair_state)
            )
        if self.config.run_mode != "accumulate":
            self._reset_adapter_ephemeral()
        else:
            for adapter in self.injection.adapters.values():
                adapter.clear_active_exception()
                for parameter in adapter.trainable_episode_parameters():
                    parameter.requires_grad_(False)
        if self.state_machine.active:
            self.state_machine.abort(str(reason))
        if episode_id is not None:
            self._emit_final_proposal_type(
                episode_id=episode_id,
                proposal=final_proposal,
                terminal_path="abort",
            )
            self.metrics.record(
                "episode_rollback",
                1,
                episode_id=episode_id,
                context_identifier=self._active_context,
                tags={"reason": str(reason)},
            )
            self.metrics.increment("rollback_count", episode_id=episode_id)
        self._clear_episode_fields()
        self.assert_base_frozen()

    def abort_episode(self, reason: str = "planner_exception") -> None:
        if self._active_episode_id is None and not self.state_machine.active:
            return
        unwinding_exception = sys.exc_info()[1]
        episode_id = str(self._active_episode_id or self.next_episode_id)
        context_identifier = self._active_context
        self._abort_components(str(reason))
        try:
            self._save_episode_snapshot_if_due(
                episode_id=episode_id,
                context_identifier=context_identifier,
                outcome="ABORT",
                reason=str(reason),
            )
        except Exception as snapshot_error:
            self.diagnostics.record(
                "error",
                "abort_episode_snapshot_failed",
                f"{type(snapshot_error).__name__}: {snapshot_error}",
                episode_id=episode_id,
                details={"abort_reason": str(reason)},
            )
            self._flush_metrics()
            if unwinding_exception is not None:
                note = (
                    "FD-PSC abort reached IDLE but its episode snapshot failed: "
                    f"{type(snapshot_error).__name__}: {snapshot_error}"
                )
                add_note = getattr(unwinding_exception, "add_note", None)
                if callable(add_note):
                    add_note(note)
                # The caller's bare ``raise`` keeps the causal planner/online
                # exception. The diagnostic and exception note make the
                # secondary durability failure explicit without masking it.
                return
            raise
        self._flush_metrics()

    def reset_episode(self) -> None:
        """Idempotent idle cleanup; never touches slow memory or theta_0."""

        if self._active_episode_id is not None or self.state_machine.active:
            raise FDPSCIntegrationError(
                "reset_episode cannot replace finish/abort for an active episode"
            )
        if self.config.run_mode != "accumulate":
            self._reset_adapter_ephemeral()
        self.assert_base_frozen()

    def _post_checkpoint_metric_event_count(self, *, checkpoint_saved: bool) -> int:
        """Number of deterministic metric IDs emitted after sidecar capture.

        Metric payloads such as wall-clock latency are diagnostic and are not
        checkpointed, but their sequence IDs are.  Projecting this fixed tail
        makes the next episode's IDs identical after an interrupted/resumed
        run and an uninterrupted run.
        """

        return (
            1  # replay_memory_bytes
            + int(bool(checkpoint_saved))  # checkpoint_bytes
            + int(self._sleep_started_at is not None)  # sleep_latency_s
            + 5  # candidate/final-gate/exception/proposal-type/terminal
        )

    def _save_episode_snapshot_if_due(
        self,
        *,
        episode_id: str,
        context_identifier: Optional[str],
        outcome: str,
        reason: str,
    ) -> Optional[Any]:
        """Publish one auditable non-model snapshot at the episode cadence.

        Persistent commits already publish a checkpoint whose lifecycle and
        metric sequence are projected to the between-episode boundary.  Other
        terminal outcomes use this path after cleanup.  No state-machine
        commit is performed, so the persistent model-commit count is unchanged.
        """

        if self.checkpoints is None:
            return None
        if self.state_machine.state != FDPSCState.IDLE or self._active_episode_id is not None:
            raise FDPSCIntegrationError(
                "episode snapshots are legal only after IDLE cleanup"
            )
        if self._episode_sequence <= 0:
            return None
        if (
            self._episode_sequence % self.config.checkpoint.save_every_episodes
            != 0
        ):
            return None
        if self._last_checkpoint_episode_sequence == self._episode_sequence:
            return None
        snapshot_base_id = f"snapshot-episode-{self._episode_sequence:08d}"
        snapshot_id = self.checkpoints.next_available_id(snapshot_base_id)
        try:
            reference = self.checkpoints.save_committed(
                self._checkpoint_state(future_metric_events=1),
                commit_id=snapshot_id,
                commit_sequence=self._commit_sequence,
                config_identity=self.config.persistence_identity_hash(),
                journal_metadata={
                    "event_type": "episode_snapshot",
                    "episode_id": str(episode_id),
                    "context_identifier": (
                        None
                        if context_identifier is None
                        else str(context_identifier)
                    ),
                    "outcome": str(outcome),
                    "reason": str(reason),
                    "save_every_episodes": int(
                        self.config.checkpoint.save_every_episodes
                    ),
                },
            )
        except Exception as exc:
            self.diagnostics.record(
                "error",
                "episode_snapshot_failed",
                f"{type(exc).__name__}: {exc}",
                episode_id=str(episode_id),
                details={
                    "snapshot_id": snapshot_id,
                    "outcome": str(outcome),
                    "reason": str(reason),
                },
            )
            raise
        self._last_checkpoint_episode_sequence = self._episode_sequence
        checkpoint_path = self.checkpoints.state_directory / reference.version_file
        self.metrics.record(
            "checkpoint_bytes",
            checkpoint_path.stat().st_size,
            episode_id=str(episode_id),
            context_identifier=context_identifier,
            tags={"commit_id": snapshot_id, "event_type": "episode_snapshot"},
        )
        return reference

    def _projected_between_episode_lifecycle(self) -> Dict[str, int]:
        projected = copy.deepcopy(self.state_machine)
        if projected.state == FDPSCState.FINAL_GATE:
            proposal = projected.final_proposal
            if proposal is None:
                raise FDPSCIntegrationError("FINAL_GATE checkpoint has no final proposal")
            if proposal.proposal_type == ProposalType.GLOBAL_SLOW:
                projected.commit_slow()
            else:
                projected.commit_exception()
        if projected.state in {
            FDPSCState.COMMIT_SLOW,
            FDPSCState.COMMIT_EXCEPTION,
            FDPSCState.REJECT_NO_PROPOSAL,
            FDPSCState.REJECT_QUERY,
        }:
            projected.finish_episode()
        if projected.state != FDPSCState.IDLE:
            raise FDPSCIntegrationError(
                f"sidecar lifecycle projection did not reach IDLE: {projected.state.value}"
            )
        return {
            "persistent_commit_count": projected.persistent_commit_count,
            "successful_slow_commit_count": self._successful_slow_commit_count,
            "rollback_count": projected.rollback_count,
            "transition_index": projected._transition_index,
        }

    def _checkpoint_state(self, *, future_metric_events: int = 0) -> Dict[str, Any]:
        if self.state_machine.state not in {
            FDPSCState.IDLE,
            FDPSCState.FINAL_GATE,
            FDPSCState.COMMIT_SLOW,
            FDPSCState.COMMIT_EXCEPTION,
            FDPSCState.REJECT_QUERY,
            FDPSCState.REJECT_NO_PROPOSAL,
        }:
            raise FDPSCIntegrationError(
                f"checkpoint requested from illegal lifecycle state {self.state_machine.state.value}"
            )
        if int(future_metric_events) < 0:
            raise FDPSCIntegrationError("future_metric_events must be non-negative")
        router_state = self.exception_router.state_dict()
        router_state["active_episode_id"] = None
        router_state["active_route"] = None
        external_state = self.external.state_dict() if self.external is not None else None
        if external_state is not None:
            external_state["active_episode_id"] = None
            external_state["active_context"] = None
            external_state["support_sealed"] = False
            external_state["support_records"] = {}
        adapter_slow = {
            logical_id: {
                "B": adapter.get_slow_factors().B.detach().cpu().clone(),
                "A": adapter.get_slow_factors().A.detach().cpu().clone(),
            }
            for logical_id, adapter in sorted(self.injection.adapters.items())
        }
        accumulate_adapter_state = (
            self._adapter_participant.state_dict()
            if self.config.run_mode == "accumulate"
            else None
        )
        metrics_state = copy.deepcopy(self.metrics.state_dict())
        metrics_state["sequence"] = int(metrics_state["sequence"]) + int(
            future_metric_events
        )
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "config_identity": self.config.persistence_identity_hash(),
            "run_mode": self.config.run_mode,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "target_manifest": self.target_manifest.to_dict(),
            "external_manifest_hash": (
                self.external.manifest_hash if self.external is not None else None
            ),
            "canary_manifest_hash": (
                self.canary_runner.manifest.manifest_hash
                if self.canary_runner is not None
                else None
            ),
            "latent_adapter_schema": LATENT_SCHEMA_VERSION,
            "episode_sequence": self._episode_sequence,
            "commit_sequence": self._commit_sequence,
            "canary_period": self._canary_period_state(),
            "lifecycle": self._projected_between_episode_lifecycle(),
            "adapter_slow": adapter_slow,
            "accumulate_adapter_state": accumulate_adapter_state,
            "base_spectra": {
                key: value.state_dict() for key, value in sorted(self._base_spectra.items())
            },
            "activation_subspaces": self.subspaces.state_dict(),
            "replay": self.replay.state_dict(),
            "exception_router": router_state,
            "commit_gates": self.gates.state_dict(),
            "external_data": external_state,
            "repair": self.repair_engine.state_dict() if self.repair_engine is not None else None,
            "history_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._history_gradients.items())
            },
            "anchor_gradients": {
                key: value.detach().cpu().clone()
                for key, value in sorted(self._anchor_gradients.items())
            },
            "metrics": metrics_state,
            "diagnostics": self.diagnostics.state_dict(),
            "rng": RNGSnapshot.capture(),
        }

    def _validate_checkpoint_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise CheckpointValidationError("FD-PSC sidecar state must be a mapping")
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported FD-PSC sidecar schema")
        expected = {
            "config_identity": self.config.persistence_identity_hash(),
            "run_mode": self.config.run_mode,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "latent_adapter_schema": LATENT_SCHEMA_VERSION,
            "external_manifest_hash": (
                self.external.manifest_hash if self.external is not None else None
            ),
            "canary_manifest_hash": (
                self.canary_runner.manifest.manifest_hash
                if self.canary_runner is not None
                else None
            ),
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise CheckpointValidationError(
                    f"sidecar {name} mismatch: {state.get(name)!r} != {value!r}"
                )
        if set(state.get("adapter_slow", {})) != set(self.injection.adapters):
            raise CheckpointValidationError("sidecar adapter registry mismatch")
        assert_finite_tree(state["adapter_slow"], path="adapter_slow")
        accumulate_adapter_state = state.get("accumulate_adapter_state")
        if self.config.run_mode == "accumulate":
            if not isinstance(accumulate_adapter_state, Mapping):
                raise CheckpointValidationError(
                    "accumulate sidecar is missing the persistent adapter state"
                )
            if set(accumulate_adapter_state) != set(self.injection.adapters):
                raise CheckpointValidationError(
                    "accumulate sidecar adapter registry mismatch"
                )
            assert_finite_tree(
                accumulate_adapter_state,
                path="accumulate_adapter_state",
            )
        elif accumulate_adapter_state is not None:
            raise CheckpointValidationError(
                "non-accumulate sidecar contains accumulate adapter state"
            )
        if int(state.get("episode_sequence", -1)) < 0 or int(state.get("commit_sequence", -1)) < 0:
            raise CheckpointValidationError("sidecar counters must be non-negative")
        lifecycle = state.get("lifecycle")
        if not isinstance(lifecycle, Mapping):
            raise CheckpointValidationError("sidecar lifecycle state is missing")
        successful_slow_count = int(
            lifecycle.get("successful_slow_commit_count", -1)
        )
        persistent_count = int(lifecycle.get("persistent_commit_count", -1))
        if (
            successful_slow_count < 0
            or persistent_count < 0
            or successful_slow_count > persistent_count
        ):
            raise CheckpointValidationError(
                "sidecar successful slow-commit count is invalid"
            )
        canary_period = state.get("canary_period")
        if not isinstance(canary_period, Mapping):
            raise CheckpointValidationError("sidecar is missing canary-period state")
        if bool(canary_period.get("enabled", False)) != (
            self.canary_runner is not None
        ):
            raise CheckpointValidationError("sidecar canary-period enablement mismatch")

    def _load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self._validate_checkpoint_state(state)
        if self._active_episode_id is not None or self.state_machine.active:
            raise FDPSCIntegrationError("resume is allowed only between episodes")
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            factors = state["adapter_slow"][logical_id]
            adapter.replace_slow_adapter(factors["B"], factors["A"])
            adapter.clear_active_exception()
        restored_spectra = {
            key: BaseSpectrum.from_state_dict(value)
            for key, value in sorted(state.get("base_spectra", {}).items())
        }
        if set(restored_spectra) != set(self._base_spectra):
            raise CheckpointValidationError("base-spectrum registry changed")
        for logical_id, spectrum in restored_spectra.items():
            if spectrum.weight_hash != self._base_spectra[logical_id].weight_hash:
                raise CheckpointValidationError(
                    f"base spectrum hash mismatch for {logical_id}"
                )
        self._base_spectra = restored_spectra
        self.subspaces.load_state_dict(state["activation_subspaces"])
        self.replay.load_state_dict(state["replay"])
        self.exception_router.load_state_dict(state["exception_router"])
        self.gates.load_state_dict(state["commit_gates"])
        if self.external is not None:
            if state.get("external_data") is None:
                raise CheckpointValidationError("sidecar is missing external-data ledger")
            self.external.load_state_dict(state["external_data"])
        if self.repair_engine is not None and state.get("repair") is not None:
            self.repair_engine.load_state_dict(state["repair"])
        self._history_gradients = {
            key: value.to(self.injection.adapters[key]._reference().device)
            for key, value in state.get("history_gradients", {}).items()
        }
        self._anchor_gradients = {
            key: value.to(self.injection.adapters[key]._reference().device)
            for key, value in state.get("anchor_gradients", {}).items()
        }
        self.metrics.load_state_dict(state["metrics"])
        self.diagnostics.load_state_dict(state["diagnostics"])
        self._episode_sequence = int(state["episode_sequence"])
        self._commit_sequence = int(state["commit_sequence"])
        self._last_checkpoint_episode_sequence = self._episode_sequence
        self._load_canary_period_state(
            state["canary_period"],
            validate_runtime=True,
        )
        lifecycle = state.get("lifecycle", {})
        self.state_machine = FDPSCStateMachine()
        self.state_machine.persistent_commit_count = int(
            lifecycle.get("persistent_commit_count", self._commit_sequence)
        )
        self._successful_slow_commit_count = int(
            lifecycle["successful_slow_commit_count"]
        )
        self.state_machine.rollback_count = int(lifecycle.get("rollback_count", 0))
        self.state_machine._transition_index = int(lifecycle.get("transition_index", 0))
        rng = state.get("rng")
        if not isinstance(rng, RNGSnapshot):
            raise CheckpointValidationError("sidecar RNG snapshot is missing or invalid")
        rng.restore()
        if self.config.run_mode == "accumulate":
            self._adapter_participant.load_state_dict(
                copy.deepcopy(state["accumulate_adapter_state"])
            )
            for adapter in self.injection.adapters.values():
                adapter.clear_active_exception()
                for parameter in adapter.trainable_episode_parameters():
                    parameter.requires_grad_(False)
        else:
            self._reset_adapter_ephemeral()
        self.assert_base_frozen()

    def _resume(self, path: Optional[Path]) -> None:
        if self.checkpoints is None or path is None:
            return
        resolved = Path(path).expanduser().resolve()
        explicit_version = False
        if resolved == self.checkpoints.latest_pointer_path:
            state, reference = self.checkpoints.load_latest(recover_if_needed=True)
        elif resolved.parent == self.checkpoints.state_directory and resolved.suffix == ".pt":
            explicit_version = True
            state, reference = self.checkpoints.load_version(resolved)
        else:
            raise CheckpointValidationError(
                "resume_path must be the configured latest pointer or an immutable version in state_directory"
            )
        if explicit_version:
            latest_state, latest_reference = self.checkpoints.recover_latest(
                repair_pointer=False
            )
            selected_episode = int(state.get("episode_sequence", -1))
            latest_episode = int(latest_state.get("episode_sequence", -1))
            selected_key = (reference.commit_sequence, reference.commit_id)
            latest_key = (
                latest_reference.commit_sequence,
                latest_reference.commit_id,
            )
            if selected_episode < latest_episode or (
                selected_episode == latest_episode and selected_key < latest_key
            ):
                raise CheckpointValidationError(
                    "explicit resume version is stale relative to the newest valid sidecar"
                )
        self._validate_checkpoint_state(state)
        unresolved = self.checkpoints.unresolved_episode_journals(
            int(state["episode_sequence"])
        )
        if unresolved:
            raise CheckpointValidationError(
                "resume is blocked by unresolved episode durability journals: "
                + ", ".join(unresolved)
            )
        self._load_checkpoint_state(state)

    def evaluate_report_test(self, trainer: Any, context_identifier: str) -> Dict[str, Any]:
        """Evaluate the fixed report-test split without changing memory state."""

        if self.external is None:
            raise FDPSCIntegrationError("report-test requires the fixed external registry")
        if self._active_episode_id is not None:
            raise FDPSCIntegrationError("report-test is only legal between episodes")
        records = self.external.report_test(str(context_identifier))
        with self._preserve_adapter_runtime():
            self._reset_adapter_ephemeral()
            total = len(records)
            persistent_total = 0.0
            theta0_total = 0.0
            for _, descriptor, group in self._group_records_for_routing(records):
                for adapter in self.injection.adapters.values():
                    adapter.adapters_enabled = True
                decision = self.exception_router.route(descriptor, production=False)
                self._apply_exception_state(decision.adapter_id)
                self.wm.eval()
                self.injection.enforce_frozen_base_eval()
                persistent_total += len(group) * trainer.evaluate_external_records(group)
                for adapter in self.injection.adapters.values():
                    adapter.disable_all_adapters()
                self.wm.eval()
                theta0_total += len(group) * trainer.evaluate_external_records(group)
            loss = persistent_total / float(total)
            theta0_loss = theta0_total / float(total)
            self.assert_base_frozen()
        gain = theta0_loss - loss
        self.assert_base_frozen()
        return {
            "schema_version": 1,
            "context_identifier": str(context_identifier),
            "record_count": len(records),
            "jepa_loss": float(loss),
            "theta0_jepa_loss": float(theta0_loss),
            "report_test_gain": float(gain),
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "external_manifest_hash": self.external.manifest_hash,
            "target_manifest_hash": self.target_manifest.hash,
            "commit_sequence": self._commit_sequence,
            "run_mode": self.config.run_mode,
        }

    def close(self) -> None:
        if self._online_hooks is not None:
            self._online_hooks.close()
            self._online_hooks = None
        self.injection.close()


__all__ = ["FDPSCIntegrationError", "FDPSCSystem"]
