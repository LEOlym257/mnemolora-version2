"""No-external-data FSD V2 wake/sleep and capacity-recycling system.

The ordinary path contains one deterministic consolidation candidate only::

    episodic task -> fresh raw-replay geometry -> shared RTRC
                  -> fixed-rank compression -> atomic persistent commit

When consecutive compression overflow is observed, an optional branch after
compression residual-distills the frozen uncompressed teacher into dense core
memory and refits a lower-rank residual slow adapter before that same atomic
commit.

It neither imports nor constructs the legacy external-data registry.  Raw
model inputs become historical replay only after a successful persistent
commit.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..config import FDPSCConfig
from ..injector import InjectionResult, inject_fd_psc_adapters
from ..low_rank_merge import (
    LowRankFactors,
    as_factors,
    concatenate_factors,
    factor_svd,
    factors_from_svd,
)
from ..metrics import StructuredMetrics
from ..replay_memory import (
    RawReplayMemory,
    RawReplayWindow,
    deep_cpu_clone,
)
from ..transaction import RNGSnapshot
from .checkpoint import FSDV2CheckpointReference, FSDV2CheckpointStore
from .budget_controller import AdaptiveBudgetController, AdaptiveBudgetUpdate
from .deep_sleep import (
    CoreDistillationResult,
    DeepSleepController,
    DeepSleepError,
    FunctionalComparison,
    ModelTrace,
    cache_residual_targets,
    compare_functional_outputs,
    distill_core_memory,
    partition_residual_targets,
    refit_parameter_residual,
)
from .geometry import FreshReplayGeometryBuilder, GeometryBuildResult, ReplayGeometry
from .rtrc import RTRCLayerInput, RTRCResult, project_full_depth
from .state_machine import FSDV2State, FSDV2StateMachine


class FSDV2IntegrationError(RuntimeError):
    """Raised when FSD V2 cannot preserve a runtime invariant."""


@dataclass(frozen=True)
class _BaseTensor:
    name: str
    tensor: Tensor
    expected: Tensor
    parameter: bool


@dataclass(frozen=True)
class _SupportSegment:
    iteration: int
    obs: Mapping[str, Any]
    actions: Any


@dataclass(frozen=True)
class _CompressionDiagnostic:
    logical_layer_id: str
    rank_before: int
    rank_after: int
    numerical_rank_before: int
    discarded_frobenius_sq: float
    discarded_frobenius: float
    relative_discarded_frobenius: float


@dataclass(frozen=True)
class _DeepSleepDiagnostic:
    triggered: bool = False
    succeeded: bool = False
    rolled_back: bool = False
    reason: str = "not_triggered"
    steps: int = 0
    output_residual_loss: float = 0.0
    hidden_residual_loss: float = 0.0
    current_jepa_loss: float = 0.0
    final_functional_error: float = 0.0
    final_functional_absolute_error: float = 0.0
    residual_rank: int = 0
    rank_reclaimed: int = 0
    core_write_frobenius: float = 0.0
    teacher_frozen: bool = True
    source_counts: Tuple[Tuple[str, int], ...] = ()
    fit_count: int = 0
    validation_count: int = 0


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _hash_update(digest: "hashlib._Hash", value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack("!d", value))
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        digest.update(b"S" + len(raw).to_bytes(8, "big") + raw)
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _hash_update(digest, str(tensor.dtype))
        _hash_update(digest, tuple(int(item) for item in tensor.shape))
        if tensor.numel():
            # ``view(dtype)`` requires a one-dimensional final axis; scalar
            # buffers such as BatchNorm.num_batches_tracked therefore need to
            # be flattened first.  The uint8 view also keeps bfloat16 and
            # other NumPy-unsupported tensor dtypes hashable byte-for-byte.
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    elif isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, str(key))
            _hash_update(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"Q" if isinstance(value, tuple) else b"L")
        for item in value:
            _hash_update(digest, item)
    else:
        _hash_update(digest, repr(value))


def _tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(item) for item in value)
    return 0


def _factor_frobenius_sq_float64(value: LowRankFactors) -> float:
    """Evaluate ``||B A||_F^2`` through the small Gram matrices."""

    factors = as_factors(value)
    if factors.rank == 0:
        return 0.0
    b = factors.b.to(dtype=torch.float64)
    a = factors.a.to(dtype=torch.float64)
    btb = b.transpose(0, 1) @ b
    aat = a @ a.transpose(0, 1)
    energy = torch.sum(btb * aat)
    return float(torch.clamp(energy, min=0.0).detach().cpu())


def _factor_difference_frobenius_sq_float64(
    left: LowRankFactors,
    right: LowRankFactors,
) -> float:
    """Evaluate ``||left-right||_F^2`` without a dense weight matrix.

    This is intentionally computed from the *actual cast factors* returned by
    compression.  Singular-value tail energy alone misses fp16/bf16 factor
    roundoff and would make the reported final drift bound non-conservative.
    """

    lhs, rhs = as_factors(left), as_factors(right)
    if (
        lhs.out_features != rhs.out_features
        or lhs.in_features != rhs.in_features
        or lhs.b.device != rhs.b.device
    ):
        raise FSDV2IntegrationError("factor difference dimensions/device mismatch")
    if lhs.rank == 0:
        return _factor_frobenius_sq_float64(rhs)
    if rhs.rank == 0:
        return _factor_frobenius_sq_float64(lhs)
    difference = LowRankFactors(
        torch.cat(
            (lhs.b.to(dtype=torch.float64), -rhs.b.to(dtype=torch.float64)),
            dim=1,
        ),
        torch.cat(
            (lhs.a.to(dtype=torch.float64), rhs.a.to(dtype=torch.float64)),
            dim=0,
        ),
    )
    return _factor_frobenius_sq_float64(difference)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _rng_snapshot_state(snapshot: RNGSnapshot) -> Dict[str, Any]:
    """Serialize the global wake RNG streams into the schema-2 sidecar."""

    if snapshot.generator_states:
        raise FSDV2IntegrationError(
            "FSD V2 global RNG snapshot unexpectedly contains named generators"
        )
    return {
        "schema_version": 1,
        "python_state": copy.deepcopy(snapshot.python_state),
        "numpy_state": copy.deepcopy(snapshot.numpy_state),
        "torch_cpu_state": snapshot.torch_cpu_state.detach().cpu().clone(),
        "torch_cuda_states": (
            None
            if snapshot.torch_cuda_states is None
            else [state.detach().cpu().clone() for state in snapshot.torch_cuda_states]
        ),
    }


def _rng_snapshot_from_state(value: Any) -> RNGSnapshot:
    """Validate and reconstruct a persisted global RNG snapshot."""

    if not isinstance(value, Mapping) or int(value.get("schema_version", -1)) != 1:
        raise FSDV2IntegrationError("FSD V2 RNG state schema is invalid")
    cpu_state = value.get("torch_cpu_state")
    if (
        not torch.is_tensor(cpu_state)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise FSDV2IntegrationError("FSD V2 torch CPU RNG state is invalid")
    raw_cuda = value.get("torch_cuda_states")
    if raw_cuda is None:
        cuda_states = None
    elif isinstance(raw_cuda, (list, tuple)) and all(
        torch.is_tensor(state)
        and state.device.type == "cpu"
        and state.dtype == torch.uint8
        and state.ndim == 1
        and state.numel() > 0
        for state in raw_cuda
    ):
        cuda_states = [state.clone() for state in raw_cuda]
    else:
        raise FSDV2IntegrationError("FSD V2 torch CUDA RNG state is invalid")
    return RNGSnapshot(
        python_state=copy.deepcopy(value.get("python_state")),
        numpy_state=copy.deepcopy(value.get("numpy_state")),
        torch_cpu_state=cpu_state.clone(),
        torch_cuda_states=cuda_states,
        generator_states={},
    )


class FSDV2System:
    """Persistent FSD V2 memory attached to one AdaJEPA world model."""

    STATE_SCHEMA_VERSION = 2
    ALGORITHM_VERSION = "fsd_v2"

    def __init__(
        self,
        *,
        wm: nn.Module,
        config: Any,
        runtime_output_dir: Path,
        canary_evaluator: Optional[Any] = None,
        runtime_preprocess_hash: Optional[str] = None,
    ) -> None:
        del canary_evaluator  # Canaries are intentionally outside the V2 algorithm.
        self.wm = wm
        self.config = FDPSCConfig.from_mapping(config)
        if not self.config.enabled or self.config.run_mode != self.ALGORITHM_VERSION:
            raise FSDV2IntegrationError(
                "FSDV2System requires enabled run_mode='fsd_v2'"
            )
        self.runtime_output_dir = Path(runtime_output_dir).expanduser().resolve()
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)
        # V2 validation checks only V2/internal paths and never reads external data.
        self.config.validate(self.runtime_output_dir, require_files=True)
        self.paths = self.config.resolve_v2_paths(self.runtime_output_dir)
        self.external = None  # reporting layer compatibility; never an algorithm input

        try:
            self.device = next(wm.parameters()).device
        except StopIteration as exc:
            raise FSDV2IntegrationError("FSD V2 requires a parameterized model") from exc
        self._base_tensors = self._capture_base_tensors()
        computed_base_hash = self._compute_base_state_hash()
        self.runtime_base_state_hash = computed_base_hash
        supplied_base_hash = str(getattr(wm, "_base_checkpoint_hash", "")).lower()
        self.base_checkpoint_hash = supplied_base_hash or computed_base_hash
        self._require_digest("base checkpoint hash", self.base_checkpoint_hash)

        supplied_preprocess_hash = (
            runtime_preprocess_hash
            if runtime_preprocess_hash is not None
            else getattr(wm, "_fd_psc_preprocess_hash", None)
        )
        self.preprocess_hash = (
            "" if supplied_preprocess_hash is None else str(supplied_preprocess_hash).lower()
        )
        if self.preprocess_hash:
            self._require_digest("preprocess hash", self.preprocess_hash)

        schema_provider = getattr(wm, "_fd_psc_schema_sample", None)
        if callable(schema_provider):
            # A model may expose a bound, zero-argument dry-run callback.  It
            # is optional because FSD V2 construction precedes episode data.
            schema_sample: Any = lambda _model: schema_provider()
        elif isinstance(schema_provider, (Mapping, tuple, list)):
            schema_sample = schema_provider
        else:
            schema_sample = None
        self.injection: InjectionResult = inject_fd_psc_adapters(
            wm,
            self.config,
            # Without an explicit model-provided dry run, construction still
            # uses the injector's structural active-path contract; no external
            # or future episode sample is manufactured here.
            schema_sample=schema_sample,
        )
        try:
            self._finish_construction_after_injection()
        except BaseException:
            # Resume/schema/checkpoint failures must leave the caller's world
            # model exactly reusable; adapter injection is a constructor
            # transaction, not an irrevocable side effect.
            self.injection.restore_original_model(self.wm)
            raise

    def _finish_construction_after_injection(self) -> None:
        if not self.injection.adapters:
            raise FSDV2IntegrationError("enabled FSD V2 injected no logical adapters")
        self.target_manifest = self.injection.manifest
        self.target_manifest_hash = self.target_manifest.hash
        self.assert_base_frozen()

        self.state_machine = FSDV2StateMachine()
        self.metrics = StructuredMetrics()
        self._metrics_audit: list[Dict[str, Any]] = []
        self.replay = RawReplayMemory(
            self.config.raw_replay.historical_windows,
            maximum_context_clusters=self.config.raw_replay.maximum_context_clusters,
            new_cluster_similarity_threshold=0.8,
            minimum_windows_per_cluster=self.config.raw_replay.minimum_windows_per_cluster,
            seed=_stable_seed(self.config.seed, "raw-replay"),
        )
        self.geometry_builder = FreshReplayGeometryBuilder(
            self.injection.physical_modules,
            maximum_rank=self.config.rtrc.geometry_maximum_rank,
            spectral_energy_threshold=self.config.rtrc.geometry_energy_threshold,
            minimum_energy=self.config.rtrc.minimum_energy,
            # Every row from the auditable replay sample contributes to the
            # spectrum.  Silent row thinning would make lambda_(m+1) a bound
            # only for the thinned subset, not for the protected history.
            maximum_activation_rows=None,
        )
        self.budget_controller = AdaptiveBudgetController(
            self.config.adaptive_budget,
            self.config.rtrc,
        )
        self.deep_sleep_controller = DeepSleepController(
            self.config.deep_sleep,
            seed=_stable_seed(self.config.seed, "deep-sleep"),
        )

        self._episode_sequence = 0
        self._commit_sequence = 0
        self._persistent_model_version = 0
        self._replay_version = 0
        self._active_episode_id: Optional[str] = None
        self._active_context_identifier: Optional[str] = None
        self._active_metadata: Dict[str, Any] = {}
        self._support_segments: list[_SupportSegment] = []
        self._support_iterations: set[int] = set()
        self._online_optimizer_steps = 0
        self._latest_online_loss: Optional[float] = None
        self._online_mode = False
        self._latest_checkpoint: Optional[FSDV2CheckpointReference] = None

        self.checkpoint_store: Optional[FSDV2CheckpointStore] = None
        if self.config.checkpoint.enabled:
            state_directory = self.paths.get("state_directory")
            latest_pointer = self.paths.get("latest_pointer")
            if state_directory is None or latest_pointer is None:
                raise FSDV2IntegrationError("FSD V2 checkpoint paths are missing")
            self.checkpoint_store = FSDV2CheckpointStore(
                state_directory=state_directory,
                latest_pointer_path=latest_pointer,
                base_checkpoint_hash=self.base_checkpoint_hash,
                runtime_base_state_hash=self.runtime_base_state_hash,
                target_manifest_hash=self.target_manifest_hash,
                config_identity=self.config.v2_persistence_identity_hash(),
                retention_versions=self.config.checkpoint.retention_versions,
            )
            resume = self.paths.get("resume")
            if resume is not None:
                state, reference = self.checkpoint_store.load_resume(resume)
                self.load_state_dict(state)
                self._latest_checkpoint = reference

        self._write_runtime_manifest()

    @staticmethod
    def _require_digest(name: str, value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise FSDV2IntegrationError(f"{name} is not a SHA-256 digest")

    def _capture_base_tensors(self) -> Tuple[_BaseTensor, ...]:
        result: list[_BaseTensor] = []
        seen: set[int] = set()
        for name, parameter in self.wm.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            result.append(
                _BaseTensor(name, parameter, parameter.detach().clone(), True)
            )
        for name, buffer in self.wm.named_buffers():
            if id(buffer) in seen:
                continue
            seen.add(id(buffer))
            result.append(_BaseTensor(name, buffer, buffer.detach().clone(), False))
        return tuple(result)

    def _compute_base_state_hash(self) -> str:
        digest = hashlib.sha256()
        for item in self._base_tensors:
            _hash_update(digest, item.name)
            _hash_update(digest, item.expected)
        return digest.hexdigest()

    def assert_base_frozen(self) -> None:
        for item in self._base_tensors:
            if not torch.equal(item.tensor.detach(), item.expected):
                raise FSDV2IntegrationError(
                    f"theta_0 invariant failed for {item.name!r}"
                )
            if item.parameter and bool(item.tensor.requires_grad):
                raise FSDV2IntegrationError(
                    f"theta_0 parameter became trainable: {item.name!r}"
                )

    def _write_runtime_manifest(self) -> None:
        value = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "algorithm_version": self.ALGORITHM_VERSION,
            "algorithm_external_data_dependency": False,
            "report_test_used_for_algorithm": False,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "runtime_base_state_hash": self.runtime_base_state_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "config_identity": self.config.v2_persistence_identity_hash(),
            "deep_sleep": {
                "enabled": bool(self.config.deep_sleep.enabled),
                "strategy": self.config.deep_sleep.strategy,
                "core_storage": self.config.deep_sleep.core_storage.mode,
            },
        }
        path = self.runtime_output_dir / "fsd_v2_runtime_manifest.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @property
    def next_episode_id(self) -> str:
        return f"episode-{self._episode_sequence:08d}"

    @property
    def persistent_model_version(self) -> int:
        return self._persistent_model_version

    def resolve_context_identifier(self, metadata: Mapping[str, Any]) -> str:
        # This uses only current episode metadata; no external registry or split.
        for key in ("context_identifier", "context", "task_id"):
            value = metadata.get(key)
            if value is not None and str(value):
                return str(value)
        sample = metadata.get("sample_idx", self._episode_sequence)
        seed = metadata.get("seed", self.config.seed)
        return f"online-context-{sample}-{seed}"

    def begin_episode(
        self,
        *,
        episode_id: str,
        context_identifier: str,
        initial_obs: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        del initial_obs
        if self.state_machine.state is not FSDV2State.IDLE:
            raise FSDV2IntegrationError("cannot begin FSD V2 while another episode is active")
        value = str(episode_id)
        context = str(context_identifier)
        if not value or not context:
            raise FSDV2IntegrationError("episode and context identifiers must be non-empty")
        self.injection.begin_episode(self._episode_sequence)
        self._freeze_all_parameters()
        self.state_machine.begin_episode(value)
        self._active_episode_id = value
        self._active_context_identifier = context
        self._active_metadata = deep_cpu_clone(dict(metadata or {}))
        self._support_segments = []
        self._support_iterations = set()
        self._online_optimizer_steps = 0
        self._latest_online_loss = None
        self._episode_sequence += 1
        self.assert_base_frozen()
        return {
            "algorithm_version": self.ALGORITHM_VERSION,
            "episode_id": value,
            "context_identifier": context,
            "external_data_dependency": False,
        }

    def require_active_episode(self) -> None:
        if (
            self._active_episode_id is None
            or self.state_machine.state is not FSDV2State.EPISODE_WAKE
        ):
            raise FSDV2IntegrationError("FSD V2 has no active wake episode")

    def register_support_segment(
        self,
        obs: Mapping[str, Any],
        actions: Any,
        *,
        iteration: int,
    ) -> None:
        self.require_active_episode()
        index = int(iteration)
        if index < 0 or index in self._support_iterations:
            raise FSDV2IntegrationError("support iteration must be unique and non-negative")
        if not isinstance(obs, Mapping) or not obs:
            raise FSDV2IntegrationError("support obs must be a non-empty mapping")
        if not torch.is_tensor(actions) or actions.ndim < 3:
            raise FSDV2IntegrationError(
                "support actions must have shape batch x transitions x action_dim"
            )
        cloned_obs = deep_cpu_clone(obs)
        cloned_actions = deep_cpu_clone(actions)
        transition_count = int(cloned_actions.shape[1])
        if transition_count <= 0:
            raise FSDV2IntegrationError("support segment needs at least one transition")
        for key, value in cloned_obs.items():
            if torch.is_tensor(value) and value.ndim >= 2:
                if int(value.shape[0]) != int(cloned_actions.shape[0]):
                    raise FSDV2IntegrationError(
                        f"support observation {key!r} batch differs from actions"
                    )
                if int(value.shape[1]) != transition_count + 1:
                    raise FSDV2IntegrationError(
                        f"support observation {key!r} must contain actions+1 frames"
                    )
        self._support_segments.append(
            _SupportSegment(index, cloned_obs, cloned_actions)
        )
        self._support_iterations.add(index)

    def _freeze_all_parameters(self) -> None:
        for parameter in self.wm.parameters():
            parameter.requires_grad_(False)

    def _keep_base_buffer_owners_eval(self) -> None:
        """Prevent theta_0 buffer writes without disabling wake Dropout.

        ``Module.eval()`` is recursive, so applying it to the whole predictor
        would silently change the normal JEPA wake objective whenever that
        predictor contains Dropout.  Instead, only modules that directly own
        a captured base buffer are forced out of training mode.  BatchNorm and
        other running-stat modules therefore stay immutable while their
        sibling stochastic modules retain the requested training semantics.
        """

        base_buffer_ids = {
            id(item.tensor) for item in self._base_tensors if not item.parameter
        }
        for module in self.wm.modules():
            if any(
                id(buffer) in base_buffer_ids
                for buffer in module.buffers(recurse=False)
            ):
                # Assigning the local flag avoids recursively switching child
                # Dropout modules back to eval mode.
                module.training = False

    def _reset_episode_adapters_frozen(self) -> None:
        """Discard fast state and leave the between-episode model immutable."""

        self.injection.reset_episode()
        self._freeze_all_parameters()

    def prepare_online_mode(self, *, predictor_train: bool, encoder_train: bool) -> None:
        self.require_active_episode()
        self._freeze_all_parameters()
        # ``InjectionResult.*_parameters`` intentionally returns only
        # parameters that are already trainable.  Re-enable the freshly
        # created episodic branches before asking it to split optimizer
        # groups; theta_0 and persistent slow factors remain frozen/buffers.
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
        predictor = self.injection.predictor_parameters()
        encoder = self.injection.encoder_parameters() if encoder_train else []
        for parameter in predictor + encoder:
            parameter.requires_grad_(True)
        # V2 fast state consists exclusively of episodic adapter parameters.
        # Preserve the ordinary AdaJEPA train/eval request (including base
        # predictor Dropout), then locally disable modules that own immutable
        # theta_0 buffers so running statistics cannot drift.
        self.wm.eval()
        predictor_module = getattr(self.wm, "predictor", None)
        if isinstance(predictor_module, nn.Module):
            predictor_module.train(bool(predictor_train))
        encoder_module = getattr(self.wm, "encoder", None)
        if isinstance(encoder_module, nn.Module):
            encoder_module.train(bool(encoder_train))
        self.injection.enforce_frozen_base_eval()
        self._keep_base_buffer_owners_eval()
        self._online_mode = True
        self.assert_base_frozen()

    def online_parameter_groups(
        self, *, include_encoder: bool
    ) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
        self.require_active_episode()
        predictor = [
            parameter
            for parameter in self.injection.predictor_parameters()
            if parameter.requires_grad
        ]
        encoder = (
            [
                parameter
                for parameter in self.injection.encoder_parameters()
                if parameter.requires_grad
            ]
            if include_encoder
            else []
        )
        base_ids = {id(item.tensor) for item in self._base_tensors if item.parameter}
        if any(id(parameter) in base_ids for parameter in predictor + encoder):
            raise FSDV2IntegrationError("theta_0 leaked into the online optimizer")
        return predictor, encoder

    def note_optimizer_step(self, step: int, loss: float) -> None:
        self.require_active_episode()
        value = float(loss)
        if int(step) <= 0 or not math.isfinite(value):
            raise FSDV2IntegrationError("online optimizer step/loss is invalid")
        self._online_optimizer_steps += 1
        self._latest_online_loss = value
        self.assert_base_frozen()

    def finish_online_event(
        self,
        trainer: Any,
        segments: Sequence[Any],
        step_losses: Sequence[float],
    ) -> None:
        del trainer, segments
        self.require_active_episode()
        if step_losses and self._online_optimizer_steps <= 0:
            raise FSDV2IntegrationError("online event recorded losses without an optimizer step")
        self.assert_base_frozen()

    def finish_online_mode(self) -> None:
        self._freeze_all_parameters()
        self.wm.eval()
        self._online_mode = False
        self.assert_base_frozen()

    def online_metrics(self) -> Dict[str, Any]:
        return {
            "fsd_v2/episode_id": self._active_episode_id,
            "fsd_v2/state": self.state_machine.state.value,
            "fsd_v2/optimizer_steps": self._online_optimizer_steps,
            "fsd_v2/latest_online_loss": self._latest_online_loss,
            "fsd_v2/raw_replay_window_count": len(self.replay),
        }

    @contextlib.contextmanager
    def _persistent_only_model_state(self) -> Iterator[None]:
        """Temporarily remove T_t while retaining slow memory M_(t-1)."""

        training = {module: bool(module.training) for module in self.wm.modules()}
        saved: list[Tuple[Tensor, Tensor]] = []
        self.assert_base_frozen()
        with torch.no_grad():
            for adapter in self.injection.adapters.values():
                for name in ("pilot_B", "center_B", "center_B0"):
                    tensor = getattr(adapter, name, None)
                    if torch.is_tensor(tensor):
                        saved.append((tensor, tensor.detach().clone()))
                        tensor.zero_()
        self.wm.eval()
        self.injection.enforce_frozen_base_eval()
        try:
            yield
        finally:
            with torch.no_grad():
                for tensor, value in saved:
                    tensor.copy_(value)
            for module, was_training in training.items():
                module.train(was_training)
            self.injection.enforce_frozen_base_eval()
            self.assert_base_frozen()

    def _forward_raw_payloads(
        self,
        trainer: Any,
        payloads: Tuple[Mapping[str, Any], ...],
    ) -> None:
        for payload in payloads:
            obs = _move_to_device(payload["obs"], self.device)
            actions = _move_to_device(payload["actions"], self.device)
            if hasattr(trainer, "_prepare_segment"):
                obs, actions = trainer._prepare_segment(obs, actions)
            elif torch.is_tensor(actions):
                actions = torch.cat(
                    (actions, torch.zeros_like(actions[:, :1])), dim=1
                )
            encoded = self.wm.encode(obs, actions)
            if hasattr(trainer, "_prediction_loss"):
                trainer._prediction_loss(encoded, detach_src=True, detach_tgt=True)
            else:
                # Minimal toy integrations still traverse predictor targets.
                predict = getattr(self.wm, "predict", None)
                if callable(predict):
                    predict(encoded[:, :-1])

    def _episode_tasks(self) -> Dict[str, LowRankFactors]:
        tasks: Dict[str, LowRankFactors] = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            value = as_factors(adapter.get_episodic_factors())
            tasks[logical_id] = LowRankFactors(
                value.b.detach().clone(), value.a.detach().clone()
            )
        return tasks

    def _geometry_for_tasks(
        self,
        tasks: Mapping[str, LowRankFactors],
        build: GeometryBuildResult,
    ) -> Dict[str, ReplayGeometry]:
        result: Dict[str, ReplayGeometry] = {}
        by_layer = build.layers
        for logical_id, task in tasks.items():
            geometry = by_layer.get(logical_id)
            if geometry is None:
                geometry = ReplayGeometry.empty(
                    task.in_features,
                    device=task.a.device,
                )
            result[logical_id] = geometry
        return result

    def _rtrc(
        self,
        tasks: Mapping[str, LowRankFactors],
        geometries: Mapping[str, ReplayGeometry],
    ) -> RTRCResult:
        epsilon = float(self.config.rtrc.epsilon)
        layers = []
        for logical_id, task in sorted(tasks.items()):
            geometry = geometries[logical_id]
            omega = 1.0 / (float(geometry.output_energy) + epsilon)
            layers.append(RTRCLayerInput(logical_id, task, geometry, omega))
        return project_full_depth(
            layers,
            beta=float(self.budget_controller.beta),
            config=self.config.rtrc,
        )

    def _compress(
        self,
        accepted: Mapping[str, LowRankFactors],
    ) -> Tuple[Dict[str, LowRankFactors], Dict[str, _CompressionDiagnostic]]:
        configured_rank = int(self.config.slow_lora.persistent_rank or 0)
        compressed: Dict[str, LowRankFactors] = {}
        diagnostics: Dict[str, _CompressionDiagnostic] = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            old = as_factors(adapter.get_slow_factors())
            candidate = concatenate_factors((old, accepted[logical_id]))
            decomposition = factor_svd(candidate)
            rank = min(
                configured_rank,
                candidate.out_features,
                candidate.in_features,
                int(decomposition.singular_values.numel()),
            )
            result = factors_from_svd(
                decomposition,
                rank=rank,
                dtype=old.b.dtype,
            )
            # Compare against the factors that will actually be installed.
            # This covers both mathematical rank truncation and factor dtype
            # roundoff, while retaining the factor-only/small-Gram contract.
            discarded_sq = _factor_difference_frobenius_sq_float64(
                candidate,
                result,
            )
            total_sq = _factor_frobenius_sq_float64(candidate)
            discarded = math.sqrt(max(discarded_sq, 0.0))
            total = math.sqrt(max(total_sq, 0.0))
            compressed[logical_id] = result
            diagnostics[logical_id] = _CompressionDiagnostic(
                logical_layer_id=logical_id,
                rank_before=candidate.rank,
                rank_after=result.rank,
                numerical_rank_before=decomposition.numerical_rank,
                discarded_frobenius_sq=discarded_sq,
                discarded_frobenius=discarded,
                relative_discarded_frobenius=(
                    discarded / (total + float(self.config.rtrc.epsilon))
                ),
            )
        return compressed, diagnostics

    def _compression_drift_bounds(
        self,
        result: RTRCResult,
        geometry: GeometryBuildResult,
        compression: Mapping[str, _CompressionDiagnostic],
    ) -> Tuple[float, float, float]:
        """Return RTRC norm budget, compression addend, and final bound.

        The full-depth block norm obeys the triangle inequality.  For each
        logical layer, ``||E Sigma^1/2||_F`` is bounded by
        ``sqrt(lambda_max) * ||E||_F``; omega is included because RTRC uses the
        weighted full-depth norm.
        """

        additive_sq = 0.0
        epsilon = float(self.config.rtrc.epsilon)
        for logical_id, diagnostic in compression.items():
            layer_geometry = geometry.layers.get(logical_id)
            if layer_geometry is None:
                continue
            lambda_max = float(layer_geometry.tail_upper_bound)
            if layer_geometry.eigenvalues.numel():
                lambda_max = max(
                    lambda_max,
                    float(layer_geometry.eigenvalues[0].detach().cpu()),
                )
            omega = 1.0 / (float(layer_geometry.output_energy) + epsilon)
            additive_sq += (
                omega
                * max(lambda_max, 0.0)
                * max(float(diagnostic.discarded_frobenius_sq), 0.0)
            )
        budget_norm = math.sqrt(max(float(result.delta), 0.0))
        compression_additive = math.sqrt(max(additive_sq, 0.0))
        return (
            budget_norm,
            compression_additive,
            budget_norm + compression_additive,
        )

    def _context_embedding(self, context_identifier: str) -> Tensor:
        digest = hashlib.sha256(str(context_identifier).encode("utf-8")).digest()
        value = torch.tensor(list(digest), dtype=torch.float32)
        value = value - torch.mean(value)
        norm = torch.linalg.vector_norm(value)
        if float(norm) <= 0.0:
            value[0] = 1.0
            norm = torch.linalg.vector_norm(value)
        return value / norm

    def _raw_windows_from_support(self) -> list[RawReplayWindow]:
        if self._active_episode_id is None or self._active_context_identifier is None:
            raise FSDV2IntegrationError("cannot build raw replay outside an episode")
        context_embedding = self._context_embedding(self._active_context_identifier)
        result = []
        for ordinal, segment in enumerate(
            sorted(self._support_segments, key=lambda item: item.iteration)
        ):
            transitions = int(segment.actions.shape[1])
            trajectory = f"{self._active_episode_id}:segment:{segment.iteration:08d}"
            transition_ids = tuple(
                f"{trajectory}:transition:{index:08d}" for index in range(transitions)
            )
            frame_ids = tuple(
                f"{trajectory}:frame:{index:08d}" for index in range(transitions + 1)
            )
            content_hash = RawReplayWindow.compute_content_hash(
                segment.obs, segment.actions
            )
            result.append(
                RawReplayWindow(
                    window_id=(
                        f"{self._active_episode_id}:raw:{ordinal:08d}:"
                        f"{content_hash[:16]}"
                    ),
                    trajectory_id=trajectory,
                    transition_ids=transition_ids,
                    frame_ids=frame_ids,
                    timesteps=tuple(range(transitions + 1)),
                    content_hash=content_hash,
                    context_identifier=self._active_context_identifier,
                    obs=segment.obs,
                    actions=segment.actions,
                    context_embedding=context_embedding,
                    source_episode=self._active_episode_id,
                    preprocess_hash=self.preprocess_hash,
                    base_checkpoint_hash=self.base_checkpoint_hash,
                    committed=True,
                    provenance="episode_support",
                    metadata={"support_iteration": segment.iteration},
                )
            )
        return result

    def _slow_snapshot(self) -> Dict[str, LowRankFactors]:
        return {
            logical_id: LowRankFactors(
                as_factors(adapter.get_slow_factors()).b.detach().clone(),
                as_factors(adapter.get_slow_factors()).a.detach().clone(),
            )
            for logical_id, adapter in self.injection.adapters.items()
        }

    def _core_snapshot(self) -> Dict[str, Tensor]:
        return {
            logical_id: adapter.get_core_delta().detach().clone()
            for logical_id, adapter in self.injection.adapters.items()
        }

    def _restore_core(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != set(self.injection.adapters):
            raise FSDV2IntegrationError("core state does not cover all adapters")
        for logical_id, value in state.items():
            self.injection.adapters[logical_id].replace_core_delta(value)

    def _restore_slow(self, state: Mapping[str, LowRankFactors]) -> None:
        for logical_id, factors in state.items():
            self.injection.adapters[logical_id].replace_slow_adapter(
                factors.b, factors.a
            )

    def _apply_slow(self, state: Mapping[str, LowRankFactors]) -> None:
        if set(state) != set(self.injection.adapters):
            raise FSDV2IntegrationError("compressed slow state does not cover all adapters")
        for logical_id, factors in state.items():
            if not torch.isfinite(factors.b).all() or not torch.isfinite(factors.a).all():
                raise FSDV2IntegrationError("compressed slow factors are not finite")
            self.injection.adapters[logical_id].replace_slow_adapter(
                factors.b, factors.a
            )

    def _apply_uncompressed_slow(
        self, state: Mapping[str, LowRankFactors]
    ) -> None:
        """Install algebraic factors without pretending their inner rank is final.

        ``M_tilde`` may concatenate more columns than a logical matrix's
        realized rank.  It is a temporary teacher representation, never a
        committed adapter, and the factorized forward remains exact.
        """

        if set(state) != set(self.injection.adapters):
            raise FSDV2IntegrationError(
                "uncompressed slow state does not cover all adapters"
            )
        for logical_id, factors in state.items():
            adapter = self.injection.adapters[logical_id]
            value = as_factors(factors)
            if (
                value.out_features != adapter.out_features
                or value.in_features != adapter.in_features
                or not torch.isfinite(value.b).all()
                or not torch.isfinite(value.a).all()
            ):
                raise FSDV2IntegrationError(
                    "uncompressed slow factors are invalid"
                )
            reference = adapter._reference()
            adapter.slow_B = value.b.detach().to(
                device=reference.device, dtype=reference.dtype
            ).clone()
            adapter.slow_A = value.a.detach().to(
                device=reference.device, dtype=reference.dtype
            ).clone()

    def _empty_slow(self) -> Dict[str, LowRankFactors]:
        result: Dict[str, LowRankFactors] = {}
        for logical_id, adapter in self.injection.adapters.items():
            reference = adapter._reference()
            result[logical_id] = LowRankFactors(
                reference.new_empty((adapter.out_features, 0)),
                reference.new_empty((0, adapter.in_features)),
            )
        return result

    def _critical_hidden_modules(self) -> Dict[str, nn.Module]:
        """Choose a bounded set of encoder/predictor output-writing layers."""

        physical = self.injection.physical_modules
        if not physical:
            return {}
        entries = tuple(self.target_manifest.entries)
        group_by_path: Dict[str, str] = {}
        for entry in entries:
            if entry.injected:
                group_by_path.setdefault(entry.module_path, entry.module_group)
        preferred_tokens = (
            "attention_output",
            "attention",
            "attn",
            "mlp_output",
            "mlp",
            "final_projection",
            "encoder_projection",
            "projector",
            "projection",
            "proj",
        )
        preferred = [
            path
            for path in sorted(physical)
            if any(token in path.lower() for token in preferred_tokens)
        ]
        # Guarantee coverage of the output side of both persistent families,
        # even when a model uses unconventional module names.
        coverage = []
        for group_prefix in ("encoder", "predictor"):
            paths = sorted(
                path
                for path in physical
                if group_by_path.get(path, "").startswith(group_prefix)
            )
            if paths:
                coverage.append(paths[-1])
        ordered = []
        for path in preferred + coverage + sorted(physical):
            if path not in ordered:
                ordered.append(path)
        maximum = int(self.config.deep_sleep.hidden_layer_maximum)
        return {path: physical[path] for path in ordered[:maximum]}

    @staticmethod
    def _hook_tensor(value: Any) -> Tensor:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            tensors = [item.reshape(-1) for item in value if torch.is_tensor(item)]
            if tensors:
                return torch.cat(tensors)
        if isinstance(value, Mapping):
            tensors = [
                value[key].reshape(-1)
                for key in sorted(value)
                if torch.is_tensor(value[key])
            ]
            if tensors:
                return torch.cat(tensors)
        raise DeepSleepError("critical hidden module returned no tensor")

    def _trace_prepared_payload(
        self,
        payload: Mapping[str, Any],
        hidden_modules: Mapping[str, nn.Module],
    ) -> ModelTrace:
        hidden: Dict[str, list[Tensor]] = {name: [] for name in hidden_modules}
        handles = []
        for name, module in hidden_modules.items():
            def capture(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any, *, key=name) -> None:
                hidden[key].append(self._hook_tensor(output).reshape(-1))

            handles.append(module.register_forward_hook(capture))
        try:
            encoded = self.wm.encode(payload["obs"], payload["actions"])
            transitions = int(encoded.shape[1]) - 1
            if transitions < 1:
                raise DeepSleepError("deep-sleep example has no prediction transition")
            history = min(int(self.wm.num_hist), transitions)
            outputs = []
            for offset in range(transitions - history + 1):
                source = encoded[:, offset : offset + history]
                prediction = self.wm.predict(source)
                if int(self.wm.concat_dim) == 0:
                    prediction = prediction[:, :, :-1, :]
                else:
                    action_width = int(self.wm.action_dim)
                    prediction = prediction[..., :-action_width]
                if prediction.numel() == 0:
                    raise DeepSleepError("deep-sleep output mask is empty")
                outputs.append(prediction.reshape(-1))
            output = torch.cat(outputs)
            hidden_trace = {
                name: torch.cat(values)
                for name, values in hidden.items()
                if values
            }
            if set(hidden_trace) != set(hidden_modules):
                missing = sorted(set(hidden_modules) - set(hidden_trace))
                raise DeepSleepError(
                    f"critical hidden layers were not executed: {missing}"
                )
            return ModelTrace(output=output, hidden=hidden_trace)
        finally:
            for handle in handles:
                handle.remove()

    def _prepare_deep_sleep_payload(
        self, trainer: Any, window: RawReplayWindow
    ) -> Mapping[str, Any]:
        obs = _move_to_device(window.obs, self.device)
        actions = _move_to_device(window.actions, self.device)
        if hasattr(trainer, "_prepare_segment"):
            obs, actions = trainer._prepare_segment(obs, actions)
        elif torch.is_tensor(actions):
            actions = torch.cat(
                (actions, torch.zeros_like(actions[:, :1])), dim=1
            )
        return {"obs": obs, "actions": actions, "window_id": window.window_id}

    def _current_jepa_loss(
        self, trainer: Any, payload: Mapping[str, Any]
    ) -> Tensor:
        encoded = self.wm.encode(payload["obs"], payload["actions"])
        if hasattr(trainer, "_prediction_loss"):
            return trainer._prediction_loss(
                encoded,
                detach_src=False,
                detach_tgt=bool(getattr(self.wm, "stop_grad", True)),
            )
        transitions = int(encoded.shape[1]) - 1
        if transitions < 1:
            return encoded.new_zeros(())
        prediction = self.wm.predict(encoded[:, :-1])
        target = encoded[:, 1:].detach()
        return torch.mean((prediction.to(torch.float32) - target.to(torch.float32)).square())

    def _evaluate_internal_jepa_loss(
        self,
        trainer: Any,
        windows: Sequence[RawReplayWindow],
    ) -> Optional[float]:
        if not windows:
            return None
        payloads = [
            self._prepare_deep_sleep_payload(trainer, window)
            for window in windows
        ]
        with torch.no_grad():
            losses = [self._current_jepa_loss(trainer, payload) for payload in payloads]
            value = torch.stack(losses).mean()
        result = float(value.detach().cpu())
        if not math.isfinite(result):
            raise FSDV2IntegrationError("internal JEPA control loss is non-finite")
        return result

    def _run_deep_sleep(
        self,
        *,
        trainer: Any,
        slow_uncompressed: Mapping[str, LowRankFactors],
        normal_compressed: Mapping[str, LowRankFactors],
        current_windows: Sequence[RawReplayWindow],
    ) -> _DeepSleepDiagnostic:
        """Execute one rollback-safe core write and residual refit."""

        core_old = self._core_snapshot()
        hidden_modules = self._critical_hidden_modules()
        if self.config.deep_sleep.hidden_residual_weight > 0.0 and not hidden_modules:
            raise DeepSleepError("hidden residual distillation selected no critical layers")
        historical = sorted(self.replay.windows(), key=lambda item: item.window_id)
        current = sorted(current_windows, key=lambda item: item.window_id)
        sources_and_payloads = [
            ("raw_replay", self._prepare_deep_sleep_payload(trainer, window))
            for window in historical
        ] + [
            ("current_support", self._prepare_deep_sleep_payload(trainer, window))
            for window in current
        ]
        if not historical:
            raise DeepSleepError("deep sleep requires historical raw replay")
        if len(historical) < int(
            self.config.deep_sleep.minimum_validation_windows
        ):
            raise DeepSleepError(
                "deep sleep lacks the minimum held-out validation windows"
            )

        with self._persistent_only_model_state():
            self._freeze_all_parameters()
            self.wm.eval()
            self.injection.enforce_frozen_base_eval()
            self._apply_uncompressed_slow(self._empty_slow())
            with torch.no_grad():
                core_traces = tuple(
                    self._trace_prepared_payload(payload, hidden_modules)
                    for _source, payload in sources_and_payloads
                )
            self._apply_uncompressed_slow(slow_uncompressed)
            with torch.no_grad():
                teacher_traces = tuple(
                    self._trace_prepared_payload(payload, hidden_modules)
                    for _source, payload in sources_and_payloads
                )
            targets = cache_residual_targets(
                sources_and_payloads=sources_and_payloads,
                core_traces=core_traces,
                teacher_traces=teacher_traces,
            )
            split = self.deep_sleep_controller.split_fit_validation_indices(
                len(historical)
            )
            fit_targets, validation_targets = partition_residual_targets(
                targets,
                historical_count=len(historical),
                split=split,
            )
            self._apply_uncompressed_slow(self._empty_slow())
            distilled: CoreDistillationResult = distill_core_memory(
                adapters=self.injection.adapters,
                targets=fit_targets,
                trace_student=lambda payload: self._trace_prepared_payload(
                    payload, hidden_modules
                ),
                current_jepa_loss=lambda payload: self._current_jepa_loss(
                    trainer, payload
                ),
                controller=self.deep_sleep_controller,
                config=self.config.deep_sleep,
            )
            core_new = self._core_snapshot()
            refit = refit_parameter_residual(
                core_old=core_old,
                slow_uncompressed=slow_uncompressed,
                core_new=core_new,
                residual_rank=int(self.config.deep_sleep.residual_rank),
                epsilon=float(self.config.deep_sleep.epsilon),
            )
            self._apply_slow(refit.factors)
            with torch.no_grad():
                final_traces = tuple(
                    self._trace_prepared_payload(target.payload, hidden_modules)
                    for target in validation_targets
                )
            teacher_outputs = tuple(
                target.core_output + target.teacher_output_residual
                for target in validation_targets
            )
            comparison: FunctionalComparison = compare_functional_outputs(
                teacher_outputs,
                tuple(trace.output for trace in final_traces),
                epsilon=float(self.config.deep_sleep.epsilon),
            )
            residual_rank = max(
                (factors.rank for factors in refit.factors.values()), default=0
            )
            reclaimed = sum(
                max(0, normal_compressed[key].rank - refit.factors[key].rank)
                for key in sorted(normal_compressed)
            )
            success = comparison.passes(
                float(self.config.deep_sleep.functional_error_threshold),
                float(self.config.deep_sleep.functional_error_absolute_tolerance),
            )
            if success:
                self.deep_sleep_controller.mark_success()
            else:
                # Expected functional rejection is a Deep Sleep subtransaction
                # rollback.  The surrounding episode can still atomically
                # commit the already-computed normal compressed slow state.
                self._restore_core(core_old)
                self._apply_slow(normal_compressed)
            source_counts = (
                ("fit_current_support", len(current)),
                ("fit_raw_replay", len(split.fit_indices)),
                ("validation_raw_replay", len(split.validation_indices)),
            )
            return _DeepSleepDiagnostic(
                triggered=True,
                succeeded=success,
                rolled_back=not success,
                reason="committed" if success else "functional_error_threshold",
                steps=distilled.steps,
                output_residual_loss=distilled.output_residual_loss,
                hidden_residual_loss=distilled.hidden_residual_loss,
                current_jepa_loss=distilled.current_jepa_loss,
                final_functional_error=comparison.relative_error,
                final_functional_absolute_error=comparison.absolute_error,
                residual_rank=residual_rank if success else max(
                    (value.rank for value in normal_compressed.values()), default=0
                ),
                rank_reclaimed=reclaimed if success else 0,
                core_write_frobenius=(
                    distilled.core_write_frobenius if success else 0.0
                ),
                teacher_frozen=distilled.teacher_frozen,
                source_counts=source_counts,
                fit_count=len(fit_targets),
                validation_count=len(validation_targets),
            )

    def _record_commit_metrics(
        self,
        result: RTRCResult,
        geometry: GeometryBuildResult,
        compression: Mapping[str, _CompressionDiagnostic],
        budget_update: AdaptiveBudgetUpdate,
        deep_sleep: _DeepSleepDiagnostic,
        replay_bytes: int,
        *,
        target: Optional[StructuredMetrics] = None,
    ) -> None:
        metrics = self.metrics if target is None else target
        episode = self._active_episode_id
        beta = float(result.beta)
        utilization = (
            float(result.accepted_drift) / max(float(result.delta), self.config.rtrc.epsilon)
            if result.delta > 0
            else 0.0
        )
        budget_norm, compression_additive, final_commit_bound = (
            self._compression_drift_bounds(result, geometry, compression)
        )
        global_rank_error = math.sqrt(
            math.fsum(
                item.discarded_frobenius_sq for item in compression.values()
            )
        )
        values = {
            "rtrc_beta_before": beta,
            "rtrc_beta_after": budget_update.beta_after,
            "rtrc_eta": float(result.eta),
            "rtrc_delta": float(result.delta),
            "rtrc_raw_drift": float(result.raw_drift),
            "rtrc_accepted_drift": float(result.accepted_drift),
            "rtrc_budget_utilization": utilization,
            "rtrc_task_distortion_frobenius": result.distortion_frobenius,
            "rtrc_geometry_rank": sum(item.rank for item in geometry.layers.values()),
            "rtrc_tail_upper_bound": max(
                (item.tail_upper_bound for item in geometry.layers.values()),
                default=0.0,
            ),
            "raw_replay_window_count": len(self.replay),
            "raw_replay_bytes": replay_bytes,
            "geometry_replay_window_count": geometry.replay_window_count,
            "geometry_reembed_latency_s": geometry.reembed_latency_s,
            "slow_rank_before": max(
                (item.rank_before for item in compression.values()), default=0
            ),
            "slow_rank_after": max(
                (item.rank_after for item in compression.values()), default=0
            ),
            "slow_numerical_rank_before_compression": max(
                (item.numerical_rank_before for item in compression.values()),
                default=0,
            ),
            "rank_compression_error": global_rank_error,
            "rank_compression_relative_error": max(
                (
                    item.relative_discarded_frobenius
                    for item in compression.values()
                ),
                default=0.0,
            ),
            "rtrc_budget_norm": budget_norm,
            "compression_additive_bound": compression_additive,
            "final_commit_drift_upper_bound": final_commit_bound,
            "wake_gain": budget_update.wake_gain,
            "rtrc_current_loss_cost": budget_update.current_loss_cost,
            "operational_plasticity_loss_ratio": (
                budget_update.operational_plasticity_loss_ratio
            ),
            "historical_replay_regression": (
                budget_update.historical_replay_regression
            ),
            "budget_controller_error": budget_update.controller_error,
            "budget_controller_update_applied": budget_update.update_applied,
            "deep_sleep_triggered": deep_sleep.triggered,
            "deep_sleep_steps": deep_sleep.steps,
            "deep_sleep_output_residual_loss": deep_sleep.output_residual_loss,
            "deep_sleep_hidden_residual_loss": deep_sleep.hidden_residual_loss,
            "deep_sleep_current_jepa_loss": deep_sleep.current_jepa_loss,
            "deep_sleep_final_functional_error": deep_sleep.final_functional_error,
            "deep_sleep_residual_rank": deep_sleep.residual_rank,
            "deep_sleep_rank_reclaimed": deep_sleep.rank_reclaimed,
            "deep_sleep_core_write_frobenius": deep_sleep.core_write_frobenius,
            "deep_sleep_rolled_back": deep_sleep.rolled_back,
            "deep_sleep_fit_count": deep_sleep.fit_count,
            "deep_sleep_validation_count": deep_sleep.validation_count,
        }
        for name, value in values.items():
            metrics.record(name, value, episode_id=episode)
        for logical_id, layer in result.layers.items():
            layer_geometry = geometry.layers[logical_id]
            metrics.record(
                "rtrc_layer_weight",
                1.0 / (float(layer_geometry.output_energy) + self.config.rtrc.epsilon),
                episode_id=episode,
                logical_layer_id=logical_id,
            )

            metrics.record(
                "rtrc_raw_drift_layer",
                layer.raw_drift,
                episode_id=episode,
                logical_layer_id=logical_id,
            )
            diagnostic = compression[logical_id]
            for name, value in (
                ("slow_rank_before", diagnostic.rank_before),
                ("slow_rank_after", diagnostic.rank_after),
                (
                    "slow_numerical_rank_before_compression",
                    diagnostic.numerical_rank_before,
                ),
                ("rank_compression_error", diagnostic.discarded_frobenius),
                (
                    "rank_compression_relative_error",
                    diagnostic.relative_discarded_frobenius,
                ),
            ):
                metrics.record(
                    name,
                    value,
                    episode_id=episode,
                    logical_layer_id=logical_id,
                )
            metrics.record(
                "rtrc_accepted_drift_layer",
                layer.accepted_drift,
                episode_id=episode,
                logical_layer_id=logical_id,
            )
            metrics.record(
                "rtrc_distortion_layer",
                layer.distortion_frobenius,
                episode_id=episode,
                logical_layer_id=logical_id,
            )

    @staticmethod
    def _validate_metrics_audit(
        value: Any,
        *,
        expected_sequence: int,
    ) -> list[Dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            raise FSDV2IntegrationError("FSD V2 metrics audit must be a sequence")
        result: list[Dict[str, Any]] = []
        for index, raw_event in enumerate(value):
            if not isinstance(raw_event, Mapping):
                raise FSDV2IntegrationError(
                    "FSD V2 metrics audit event must be a mapping"
                )
            event = copy.deepcopy(dict(raw_event))
            if int(event.get("sequence", -1)) != index:
                raise FSDV2IntegrationError(
                    "FSD V2 metrics audit sequence is not contiguous"
                )
            try:
                json.dumps(event, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise FSDV2IntegrationError(
                    "FSD V2 metrics audit event is not finite JSON"
                ) from exc
            result.append(event)
        if len(result) != int(expected_sequence):
            raise FSDV2IntegrationError(
                "FSD V2 metrics audit does not match its next sequence"
            )
        return result

    def _prospective_metrics_audit(
        self,
        pending_metrics: StructuredMetrics,
    ) -> list[Dict[str, Any]]:
        result = copy.deepcopy(self._metrics_audit)
        for event_value in pending_metrics.events():
            event = asdict(event_value)
            if int(event["sequence"]) != len(result):
                raise FSDV2IntegrationError(
                    "pending FSD V2 metric event would create an audit gap"
                )
            result.append(event)
        return self._validate_metrics_audit(
            result,
            expected_sequence=int(pending_metrics.state_dict()["sequence"]),
        )

    def _export_metrics(self) -> None:
        """Atomically merge this process's events with the published audit log.

        ``StructuredMetrics`` intentionally checkpoints only its next sequence
        and counters.  After resume, its in-memory event list therefore starts
        at the restored sequence.  Rewriting from that list alone would erase
        prior committed diagnostics, so JSONL is treated as the canonical
        append history and CSV is regenerated from the merged sequence.
        """

        jsonl_path = self.runtime_output_dir / "fsd_v2_metrics.jsonl"
        csv_path = self.runtime_output_dir / "fsd_v2_metrics.csv"
        merged: list[Dict[str, Any]] = []
        if jsonl_path.exists():
            for line_number, raw_line in enumerate(
                jsonl_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise FSDV2IntegrationError(
                        f"existing FSD V2 metrics JSONL is corrupt at line {line_number}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise FSDV2IntegrationError(
                        "existing FSD V2 metrics JSONL event is not a mapping"
                    )
                event = dict(payload)
                if int(event.get("sequence", -1)) != len(merged):
                    raise FSDV2IntegrationError(
                        "existing FSD V2 metrics JSONL sequence is not contiguous"
                    )
                merged.append(event)

        for raw_event in self._metrics_audit:
            event = copy.deepcopy(raw_event)
            sequence = int(event["sequence"])
            if sequence < len(merged):
                if merged[sequence] != event:
                    raise FSDV2IntegrationError(
                        "FSD V2 metrics sequence conflicts with existing audit history"
                    )
                continue
            if sequence != len(merged):
                raise FSDV2IntegrationError(
                    "FSD V2 metrics export would create a sequence gap"
                )
            merged.append(event)

        expected_sequence = int(self.metrics.state_dict()["sequence"])
        if len(merged) != expected_sequence:
            raise FSDV2IntegrationError(
                "FSD V2 metrics audit history does not reach the checkpointed sequence"
            )

        jsonl_temporary = jsonl_path.with_name(f".{jsonl_path.name}.tmp")
        csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp")
        try:
            with jsonl_temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for event in merged:
                    handle.write(
                        json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(jsonl_temporary, jsonl_path)

            fields = [
                "sequence",
                "timestamp_ns",
                "name",
                "value",
                "episode_id",
                "replan_index",
                "logical_layer_id",
                "context_identifier",
                "tags_json",
            ]
            with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for event in merged:
                    row = dict(event)
                    row["tags_json"] = json.dumps(
                        row.pop("tags", {}),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(csv_temporary, csv_path)
        finally:
            for temporary in (jsonl_temporary, csv_temporary):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def sleep(self, trainer: Any) -> Dict[str, Any]:
        self.require_active_episode()
        if not self._support_segments or self._online_optimizer_steps <= 0:
            return self.finish_episode_without_sleep(
                "no_support_or_no_online_update"
            )

        rng_before = RNGSnapshot.capture()
        replay_before = copy.deepcopy(self.replay.state_dict())
        slow_before = self._slow_snapshot()
        core_before = self._core_snapshot()
        budget_controller_before = copy.deepcopy(
            self.budget_controller.state_dict()
        )
        deep_sleep_controller_before = copy.deepcopy(
            self.deep_sleep_controller.state_dict()
        )
        state_machine_before = copy.deepcopy(self.state_machine)
        commit_before = self._commit_sequence
        persistent_version_before = self._persistent_model_version
        replay_version_before = self._replay_version
        try:
            self.state_machine.enter_sleep_geometry()
            tasks = self._episode_tasks()
            history_windows = self.replay.sample_balanced(
                min(int(self.config.rtrc.geometry_windows), len(self.replay)),
                allow_replacement=False,
            )
            geometry_build = self.geometry_builder.build(
                history_windows,
                forward_callback=lambda payloads: self._forward_raw_payloads(
                    trainer, payloads
                ),
                persistent_model_version=self._persistent_model_version,
                model_state_context=self._persistent_only_model_state,
            )
            geometries = self._geometry_for_tasks(tasks, geometry_build)

            self.state_machine.enter_sleep_rtrc()
            rtrc_result = self._rtrc(tasks, geometries)
            tolerance = max(
                float(self.config.rtrc.epsilon),
                float(self.config.rtrc.bisection_relative_tolerance)
                * max(float(rtrc_result.delta), float(self.config.rtrc.epsilon)),
            )
            if rtrc_result.accepted_drift > rtrc_result.delta + tolerance:
                raise FSDV2IntegrationError("RTRC accepted drift exceeds its budget")

            self.state_machine.enter_sleep_compress()
            accepted = {
                logical_id: layer.accepted
                for logical_id, layer in rtrc_result.layers.items()
            }
            compressed, compression = self._compress(accepted)
            raw_windows = self._raw_windows_from_support()

            slow_uncompressed = {
                logical_id: concatenate_factors(
                    (slow_before[logical_id], accepted[logical_id])
                )
                for logical_id in sorted(self.injection.adapters)
            }
            # Operational controller signals use only the current episode and
            # internal replay, each evaluated at the exact three states from
            # the guide: before, fast, and before+RTRC (uncompressed).
            current_fast = self._evaluate_internal_jepa_loss(
                trainer, raw_windows
            )
            assert current_fast is not None
            with self._persistent_only_model_state():
                current_before = self._evaluate_internal_jepa_loss(
                    trainer, raw_windows
                )
                history_before = self._evaluate_internal_jepa_loss(
                    trainer, history_windows
                )
                try:
                    self._apply_uncompressed_slow(slow_uncompressed)
                    current_rtrc = self._evaluate_internal_jepa_loss(
                        trainer, raw_windows
                    )
                    history_rtrc = self._evaluate_internal_jepa_loss(
                        trainer, history_windows
                    )
                finally:
                    self._restore_slow(slow_before)
            assert current_before is not None and current_rtrc is not None
            budget_update = self.budget_controller.update(
                current_before=current_before,
                current_fast=current_fast,
                current_rtrc=current_rtrc,
                history_before=history_before,
                history_rtrc=history_rtrc,
                epsilon=float(self.config.rtrc.epsilon),
            )
            rank_error = max(
                (
                    diagnostic.relative_discarded_frobenius
                    for diagnostic in compression.values()
                ),
                default=0.0,
            )
            deep_sleep_attempt_before = copy.deepcopy(
                self.deep_sleep_controller.state_dict()
            )
            deep_sleep_global_rng_before = RNGSnapshot.capture()
            deep_sleep_triggered = self.deep_sleep_controller.should_trigger(
                rank_error,
                # The minimum guards historical coverage specifically; the
                # current support is always added separately to the actual
                # distillation corpus.
                available_windows=len(self.replay),
            )
            if deep_sleep_triggered:
                self.state_machine.enter_deep_sleep()
                deep_sleep = self._run_deep_sleep(
                    trainer=trainer,
                    slow_uncompressed=slow_uncompressed,
                    normal_compressed=compressed,
                    current_windows=raw_windows,
                )
                if deep_sleep.rolled_back:
                    # Functional rejection rolls back core, residual slow,
                    # controller counters, and its sampler RNG as one Deep
                    # Sleep subtransaction.
                    self.deep_sleep_controller.load_state_dict(
                        deep_sleep_attempt_before
                    )
                    deep_sleep_global_rng_before.restore()
            else:
                self._apply_slow(compressed)
                deep_sleep = _DeepSleepDiagnostic(
                    triggered=False,
                    succeeded=False,
                    rolled_back=False,
                    reason=(
                        "disabled"
                        if not self.config.deep_sleep.enabled
                        else "capacity_healthy_or_not_consecutive"
                    ),
                )

            self.state_machine.enter_commit()
            admissions = self.replay.add_committed_windows(
                raw_windows, commit_kind="slow"
            )
            self._commit_sequence += 1
            self._persistent_model_version += 1
            # A reservoir rejection still advances seen IDs/counts, cluster
            # prototypes, and its RNG.  Any successful admission attempt is
            # therefore a replay-state mutation even when no stored window is
            # replaced, and must invalidate versioned geometry/cache state.
            if raw_windows:
                self._replay_version += 1
            self.state_machine.finish_commit()

            replay_bytes = sum(
                _tree_nbytes(window.obs) + _tree_nbytes(window.actions)
                for window in self.replay.windows()
            )
            # Pre-validate this commit's metric events in an isolated ledger.
            # The prospective sequence is placed in the published sidecar, so a
            # crash immediately after the pointer swap cannot cause metric IDs
            # to be reused on resume.
            pending_metrics = StructuredMetrics()
            pending_metrics.load_state_dict(
                copy.deepcopy(self.metrics.state_dict())
            )
            self._record_commit_metrics(
                rtrc_result,
                geometry_build,
                compression,
                budget_update,
                deep_sleep,
                replay_bytes,
                target=pending_metrics,
            )
            prospective_metrics_audit = self._prospective_metrics_audit(
                pending_metrics
            )
            self.assert_base_frozen()

            if (
                self.checkpoint_store is not None
                and self._commit_sequence
                % int(self.config.checkpoint.save_every_episodes)
                == 0
            ):
                checkpoint_state = self.state_dict()
                checkpoint_state["metrics"] = copy.deepcopy(
                    pending_metrics.state_dict()
                )
                checkpoint_state["metrics_audit"] = copy.deepcopy(
                    prospective_metrics_audit
                )
                self._latest_checkpoint = self.checkpoint_store.save(
                    checkpoint_state,
                    commit_sequence=self._commit_sequence,
                )
        except Exception:
            self._restore_slow(slow_before)
            self._restore_core(core_before)
            self.budget_controller.load_state_dict(budget_controller_before)
            self.deep_sleep_controller.load_state_dict(
                deep_sleep_controller_before
            )
            self.replay.load_state_dict(replay_before)
            self._commit_sequence = commit_before
            self._persistent_model_version = persistent_version_before
            self._replay_version = replay_version_before
            # A checkpoint failure occurs after finish_commit() has returned
            # the machine to idle.  Restore the pre-sleep wake state first so
            # rollback is recorded and no failed commit transition survives.
            if self.state_machine.state is FSDV2State.IDLE:
                self.state_machine = state_machine_before
            self.state_machine.rollback("sleep_failed")
            self._reset_episode_adapters_frozen()
            self._clear_active_episode()
            rng_before.restore()
            self.assert_base_frozen()
            raise

        # Install the exact pre-validated events whose sequence/audit payload
        # was placed in the published sidecar.  This closes the process-level
        # window
        # between pointer publication and diagnostic-file export.
        self.metrics = pending_metrics
        self._metrics_audit = prospective_metrics_audit
        metric_export_error: Optional[str] = None
        try:
            self._export_metrics()
        except Exception as exc:
            # Metrics files are reporting artifacts, never an algorithmic
            # gate.  The sidecar already contains the correct next sequence.
            metric_export_error = f"{type(exc).__name__}: {exc}"
        budget_norm, compression_additive, final_commit_bound = (
            self._compression_drift_bounds(
                rtrc_result,
                geometry_build,
                compression,
            )
        )
        report = {
            "algorithm_version": self.ALGORITHM_VERSION,
            "status": "committed",
            "episode_id": self._active_episode_id,
            "commit_sequence": self._commit_sequence,
            "persistent_model_version": self._persistent_model_version,
            "rtrc_beta": float(rtrc_result.beta),
            "rtrc_beta_after": budget_update.beta_after,
            "rtrc_eta": float(rtrc_result.eta),
            "rtrc_delta": float(rtrc_result.delta),
            "rtrc_raw_drift": float(rtrc_result.raw_drift),
            "rtrc_accepted_drift": float(rtrc_result.accepted_drift),
            "rtrc_geometry_rank": sum(
                geometry.rank for geometry in geometry_build.layers.values()
            ),
            "rtrc_tail_upper_bound": max(
                (geometry.tail_upper_bound for geometry in geometry_build.layers.values()),
                default=0.0,
            ),
            "geometry_replay_window_count": geometry_build.replay_window_count,
            "raw_replay_window_count": len(self.replay),
            "raw_replay_bytes": replay_bytes,
            "rank_compression_error": math.sqrt(
                math.fsum(
                    item.discarded_frobenius_sq
                    for item in compression.values()
                )
            ),
            "rank_compression_relative_error": max(
                (
                    item.relative_discarded_frobenius
                    for item in compression.values()
                ),
                default=0.0,
            ),
            "rtrc_budget_norm": budget_norm,
            "compression_additive_bound": compression_additive,
            "final_commit_drift_upper_bound": final_commit_bound,
            "wake_gain": budget_update.wake_gain,
            "rtrc_current_loss_cost": budget_update.current_loss_cost,
            "operational_plasticity_loss_ratio": (
                budget_update.operational_plasticity_loss_ratio
            ),
            "historical_replay_regression": (
                budget_update.historical_replay_regression
            ),
            "budget_controller_error": budget_update.controller_error,
            "budget_controller_update_applied": budget_update.update_applied,
            "deep_sleep_triggered": deep_sleep.triggered,
            "deep_sleep_status": deep_sleep.reason,
            "deep_sleep_steps": deep_sleep.steps,
            "deep_sleep_output_residual_loss": deep_sleep.output_residual_loss,
            "deep_sleep_hidden_residual_loss": deep_sleep.hidden_residual_loss,
            "deep_sleep_current_jepa_loss": deep_sleep.current_jepa_loss,
            "deep_sleep_final_functional_error": deep_sleep.final_functional_error,
            "deep_sleep_final_functional_absolute_error": (
                deep_sleep.final_functional_absolute_error
            ),
            "deep_sleep_residual_rank": deep_sleep.residual_rank,
            "deep_sleep_rank_reclaimed": deep_sleep.rank_reclaimed,
            "deep_sleep_core_write_frobenius": deep_sleep.core_write_frobenius,
            "deep_sleep_teacher_frozen": deep_sleep.teacher_frozen,
            "deep_sleep_source_counts": dict(deep_sleep.source_counts),
            "deep_sleep_rolled_back": deep_sleep.rolled_back,
            "deep_sleep_fit_count": deep_sleep.fit_count,
            "deep_sleep_validation_count": deep_sleep.validation_count,
            "external_data_dependency": False,
        }
        if metric_export_error is not None:
            report["metrics_export_error"] = metric_export_error
        self._reset_episode_adapters_frozen()
        self._clear_active_episode()
        self.assert_base_frozen()
        return report

    def end_episode_and_sleep(
        self,
        trainer: Any,
        obs_seqs: Sequence[Any],
        act_seqs: Sequence[Any],
    ) -> Dict[str, Any]:
        self.require_active_episode()
        if not self._support_segments:
            for index, (obs, actions) in enumerate(zip(obs_seqs, act_seqs)):
                self.register_support_segment(obs, actions, iteration=index)
        return self.sleep(trainer)

    def finish_episode_without_sleep(self, reason: str) -> Dict[str, Any]:
        self.require_active_episode()
        episode = self._active_episode_id
        self.state_machine.finish_without_sleep(str(reason))
        self._reset_episode_adapters_frozen()
        self._clear_active_episode()
        self.assert_base_frozen()
        return {
            "algorithm_version": self.ALGORITHM_VERSION,
            "status": "not_committed",
            "reason": str(reason),
            "episode_id": episode,
            "raw_replay_window_count": len(self.replay),
            "external_data_dependency": False,
        }

    def abort_episode(self, reason: str = "planner_exception") -> None:
        if self.state_machine.state is FSDV2State.IDLE:
            return
        self.state_machine.rollback(str(reason))
        self._reset_episode_adapters_frozen()
        self._clear_active_episode()
        self.assert_base_frozen()

    def _clear_active_episode(self) -> None:
        self._active_episode_id = None
        self._active_context_identifier = None
        self._active_metadata = {}
        self._support_segments = []
        self._support_iterations = set()
        self._online_optimizer_steps = 0
        self._latest_online_loss = None
        self._online_mode = False

    def reset_episode(self) -> None:
        if self.state_machine.state is not FSDV2State.IDLE:
            self.abort_episode("reset_episode")
            return
        self._reset_episode_adapters_frozen()
        self._clear_active_episode()
        self.assert_base_frozen()

    def state_dict(self) -> Dict[str, Any]:
        if self.state_machine.state is not FSDV2State.IDLE:
            raise FSDV2IntegrationError("FSD V2 may only checkpoint between episodes")
        slow = {}
        core = {}
        for logical_id, adapter in sorted(self.injection.adapters.items()):
            value = as_factors(adapter.get_slow_factors())
            slow[logical_id] = {
                "b": value.b.detach().cpu().clone(),
                "a": value.a.detach().cpu().clone(),
            }
            core[logical_id] = adapter.get_core_delta().detach().cpu().clone()
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "algorithm_version": self.ALGORITHM_VERSION,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "runtime_base_state_hash": self.runtime_base_state_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "config_identity": self.config.v2_persistence_identity_hash(),
            "preprocess_hash": self.preprocess_hash,
            "core_delta": core,
            "slow_factors": slow,
            "deep_sleep_controller": copy.deepcopy(
                self.deep_sleep_controller.state_dict()
            ),
            "raw_replay": copy.deepcopy(self.replay.state_dict()),
            "budget_controller": copy.deepcopy(
                self.budget_controller.state_dict()
            ),
            "episode_sequence": self._episode_sequence,
            "commit_sequence": self._commit_sequence,
            "persistent_model_version": self._persistent_model_version,
            "replay_version": self._replay_version,
            "metrics": copy.deepcopy(self.metrics.state_dict()),
            "metrics_audit": copy.deepcopy(self._metrics_audit),
            "state_machine": self.state_machine.state_dict(),
            "global_rng": _rng_snapshot_state(RNGSnapshot.capture()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.state_machine.state is not FSDV2State.IDLE:
            raise FSDV2IntegrationError(
                "FSD V2 state may only be loaded between episodes"
            )
        version = int(state.get("schema_version", -1))
        if version == 1:
            raise FSDV2IntegrationError(
                "V1 sidecar cannot be loaded as FSD V2 without explicit migration."
            )
        if version != self.STATE_SCHEMA_VERSION:
            raise FSDV2IntegrationError(
                f"unsupported FSD V2 algorithm-state schema {version}"
            )
        expected = {
            "algorithm_version": self.ALGORITHM_VERSION,
            "base_checkpoint_hash": self.base_checkpoint_hash,
            "runtime_base_state_hash": self.runtime_base_state_hash,
            "target_manifest_hash": self.target_manifest_hash,
            "config_identity": self.config.v2_persistence_identity_hash(),
            "preprocess_hash": self.preprocess_hash,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise FSDV2IntegrationError(f"FSD V2 state {key} mismatch")
        raw_slow = state.get("slow_factors")
        if not isinstance(raw_slow, Mapping) or set(raw_slow) != set(self.injection.adapters):
            raise FSDV2IntegrationError("FSD V2 state slow-factor manifest mismatch")
        candidates: Dict[str, LowRankFactors] = {}
        for logical_id, payload in raw_slow.items():
            if not isinstance(payload, Mapping):
                raise FSDV2IntegrationError("FSD V2 slow-factor payload is invalid")
            b, a = payload.get("b"), payload.get("a")
            if not torch.is_tensor(b) or not torch.is_tensor(a):
                raise FSDV2IntegrationError("FSD V2 slow factors must be tensors")
            factors = LowRankFactors(b, a)
            adapter = self.injection.adapters[logical_id]
            if (
                factors.out_features != adapter.out_features
                or factors.in_features != adapter.in_features
                or factors.rank > int(self.config.slow_lora.persistent_rank or 0)
            ):
                raise FSDV2IntegrationError("FSD V2 slow-factor dimensions/rank mismatch")
            if not torch.isfinite(b).all() or not torch.isfinite(a).all():
                raise FSDV2IntegrationError("FSD V2 slow factors are not finite")
            candidates[logical_id] = factors

        raw_core = state.get("core_delta")
        if not isinstance(raw_core, Mapping) or set(raw_core) != set(
            self.injection.adapters
        ):
            raise FSDV2IntegrationError("FSD V2 core-delta manifest mismatch")
        candidate_core: Dict[str, Tensor] = {}
        for logical_id, value in raw_core.items():
            adapter = self.injection.adapters[logical_id]
            if (
                not torch.is_tensor(value)
                or tuple(value.shape)
                != (adapter.out_features, adapter.in_features)
                or not torch.isfinite(value).all()
            ):
                raise FSDV2IntegrationError(
                    "FSD V2 core_delta dimensions/value are invalid"
                )
            candidate_core[logical_id] = value.detach().clone()

        episode_sequence = int(state.get("episode_sequence", -1))
        commit_sequence = int(state.get("commit_sequence", -1))
        persistent_version = int(state.get("persistent_model_version", -1))
        replay_version = int(state.get("replay_version", -1))
        if (
            episode_sequence < 0
            or commit_sequence < 0
            or persistent_version != commit_sequence
            or replay_version < 0
            or replay_version > commit_sequence
            or commit_sequence > episode_sequence
        ):
            raise FSDV2IntegrationError("FSD V2 sequence/version state is invalid")

        # Validate every independently stateful participant in isolation before
        # touching the live system.  In particular, both metrics and the state
        # machine mutate during their load methods, so a failed candidate must
        # simply be discarded rather than partially installed.
        candidate_replay = RawReplayMemory(
            self.replay.capacity,
            maximum_context_clusters=self.replay.maximum_context_clusters,
            new_cluster_similarity_threshold=(
                self.replay.new_cluster_similarity_threshold
            ),
            minimum_windows_per_cluster=self.replay.minimum_windows_per_cluster,
            seed=self.replay.seed,
        )
        candidate_replay.load_state_dict(copy.deepcopy(state["raw_replay"]))
        for window in candidate_replay.windows():
            if window.base_checkpoint_hash != self.base_checkpoint_hash:
                raise FSDV2IntegrationError("raw replay base checkpoint hash mismatch")
            if window.preprocess_hash != self.preprocess_hash:
                raise FSDV2IntegrationError("raw replay preprocess hash mismatch")
        candidate_state_machine = FSDV2StateMachine()
        candidate_state_machine.load_state_dict(state["state_machine"])
        candidate_metrics = StructuredMetrics()
        candidate_metrics.load_state_dict(state["metrics"])
        candidate_metrics_audit = self._validate_metrics_audit(
            state.get("metrics_audit"),
            expected_sequence=int(candidate_metrics.state_dict()["sequence"]),
        )
        candidate_rng = _rng_snapshot_from_state(state.get("global_rng"))
        candidate_deep_sleep_controller = DeepSleepController(
            self.config.deep_sleep,
            seed=self.deep_sleep_controller.seed,
        )
        candidate_deep_sleep_controller.load_state_dict(
            copy.deepcopy(state.get("deep_sleep_controller"))
        )
        candidate_budget_controller = AdaptiveBudgetController(
            self.config.adaptive_budget,
            self.config.rtrc,
        )
        candidate_budget_controller.load_state_dict(
            copy.deepcopy(state.get("budget_controller"))
        )
        if int(candidate_metrics.state_dict()["sequence"]) < 0:
            raise FSDV2IntegrationError("FSD V2 metrics sequence is invalid")

        replay_before = copy.deepcopy(self.replay.state_dict())
        slow_before = self._slow_snapshot()
        core_before = self._core_snapshot()
        deep_sleep_controller_before = self.deep_sleep_controller
        budget_controller_before = self.budget_controller
        state_machine_before = self.state_machine
        metrics_before = self.metrics
        metrics_audit_before = self._metrics_audit
        rng_before = RNGSnapshot.capture()
        sequences_before = (
            self._episode_sequence,
            self._commit_sequence,
            self._persistent_model_version,
            self._replay_version,
        )
        try:
            self.replay.load_state_dict(candidate_replay.state_dict())
            self._apply_slow(candidates)
            self._restore_core(candidate_core)
            self.deep_sleep_controller = candidate_deep_sleep_controller
            self.budget_controller = candidate_budget_controller
            self.state_machine = candidate_state_machine
            self.metrics = candidate_metrics
            self._metrics_audit = candidate_metrics_audit
            self._episode_sequence = episode_sequence
            self._commit_sequence = commit_sequence
            self._persistent_model_version = persistent_version
            self._replay_version = replay_version
            self._clear_active_episode()
            self._reset_episode_adapters_frozen()
            candidate_rng.restore()
            self.assert_base_frozen()
        except Exception:
            self.replay.load_state_dict(replay_before)
            self._restore_slow(slow_before)
            self._restore_core(core_before)
            self.deep_sleep_controller = deep_sleep_controller_before
            self.budget_controller = budget_controller_before
            self.state_machine = state_machine_before
            self.metrics = metrics_before
            self._metrics_audit = metrics_audit_before
            (
                self._episode_sequence,
                self._commit_sequence,
                self._persistent_model_version,
                self._replay_version,
            ) = sequences_before
            rng_before.restore()
            raise

    def close(self) -> None:
        self.injection.close()


__all__ = ["FSDV2IntegrationError", "FSDV2System"]
