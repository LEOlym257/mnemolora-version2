"""Fresh replay-induced activation geometry for FSD V2.

The builder never reads a latent cache.  Its caller supplies a raw-forward
callback and, optionally, a context manager that installs the episode-start
persistent model state (``W_before``) and restores the live fast state.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..activation_subspace import (
    ActivationSubspace,
    conv2d_group_activation_matrices,
)
from ..replay_memory import RawReplayWindow


class GeometryError(RuntimeError):
    """Raised when raw replay cannot produce auditable layer geometry."""


@dataclass(frozen=True)
class ReplayGeometry:
    """Top activation spectrum plus a conservative discarded-tail bound."""

    q: Tensor
    eigenvalues: Tensor
    tail_upper_bound: float
    sample_count: int
    input_dim: int
    output_energy: float

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.q) or self.q.ndim != 2:
            raise ValueError("ReplayGeometry.q must be a matrix")
        if not torch.is_tensor(self.eigenvalues) or self.eigenvalues.ndim != 1:
            raise ValueError("ReplayGeometry.eigenvalues must be a vector")
        if self.q.shape[1] != self.eigenvalues.shape[0]:
            raise ValueError("ReplayGeometry Q rank and eigenvalue count differ")
        if self.q.device != self.eigenvalues.device:
            raise ValueError("ReplayGeometry Q/eigenvalues must share a device")
        if int(self.q.shape[0]) != int(self.input_dim):
            raise ValueError("ReplayGeometry Q input dimension is inconsistent")
        if int(self.sample_count) < 0 or int(self.input_dim) < 0:
            raise ValueError("ReplayGeometry dimensions/counts must be non-negative")
        if not torch.isfinite(self.q).all() or not torch.isfinite(
            self.eigenvalues
        ).all():
            raise ValueError("ReplayGeometry tensors must be finite")
        if torch.any(self.eigenvalues < 0):
            raise ValueError("ReplayGeometry eigenvalues must be non-negative")
        if self.eigenvalues.numel() > 1 and torch.any(
            self.eigenvalues[1:] > self.eigenvalues[:-1]
        ):
            raise ValueError("ReplayGeometry eigenvalues must be non-increasing")
        tail = float(self.tail_upper_bound)
        output_energy = float(self.output_energy)
        if not math.isfinite(tail) or tail < 0:
            raise ValueError("ReplayGeometry tail_upper_bound must be finite and non-negative")
        if not math.isfinite(output_energy) or output_energy < 0:
            raise ValueError("ReplayGeometry output_energy must be finite and non-negative")
        if self.eigenvalues.numel() and tail > float(self.eigenvalues[-1]) + max(
            1.0e-12,
            1.0e-6 * abs(float(self.eigenvalues[-1])),
        ):
            raise ValueError("ReplayGeometry tail exceeds the last retained eigenvalue")
        if int(self.sample_count) == 0:
            if self.q.shape[1] != 0 or tail != 0.0 or output_energy != 0.0:
                raise ValueError("empty ReplayGeometry must have zero rank, tail, and energy")
        elif int(self.input_dim) <= 0:
            raise ValueError("non-empty ReplayGeometry requires a positive input dimension")

    @property
    def rank(self) -> int:
        return int(self.q.shape[1])

    @classmethod
    def empty(
        cls,
        input_dim: int,
        *,
        device: Optional[torch.device] = None,
        q_dtype: torch.dtype = torch.float32,
    ) -> "ReplayGeometry":
        return cls(
            q=torch.empty(int(input_dim), 0, device=device, dtype=q_dtype),
            eigenvalues=torch.empty(0, device=device, dtype=torch.float64),
            tail_upper_bound=0.0,
            sample_count=0,
            input_dim=int(input_dim),
            output_energy=0.0,
        )


@dataclass(frozen=True)
class GeometryBuildResult:
    """All logical-layer geometries from one fresh raw replay forward."""

    by_layer: Mapping[str, ReplayGeometry]
    window_count: int
    reembed_latency_s: float
    persistent_model_version: int = 0

    def __post_init__(self) -> None:
        stable = {str(key): value for key, value in sorted(self.by_layer.items())}
        if any(not key for key in stable):
            raise ValueError("geometry logical layer ids must be non-empty")
        if any(not isinstance(value, ReplayGeometry) for value in stable.values()):
            raise TypeError("GeometryBuildResult values must be ReplayGeometry")
        if int(self.window_count) < 0:
            raise ValueError("window_count must be non-negative")
        if int(self.persistent_model_version) < 0:
            raise ValueError("persistent_model_version must be non-negative")
        latency = float(self.reembed_latency_s)
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("reembed_latency_s must be finite and non-negative")
        object.__setattr__(self, "by_layer", stable)

    @property
    def geometries(self) -> Mapping[str, ReplayGeometry]:
        return self.by_layer

    @property
    def replay_window_count(self) -> int:
        return int(self.window_count)

    @property
    def layers(self) -> Mapping[str, ReplayGeometry]:
        """Compatibility alias for consumers that call the mapping layers."""

        return self.by_layer

    def __getitem__(self, logical_layer_id: str) -> ReplayGeometry:
        return self.by_layer[logical_layer_id]


@dataclass(frozen=True)
class _CaptureSpec:
    path: str
    module: nn.Module
    base_layer: nn.Module
    logical_layer_ids: Tuple[str, ...]
    input_dims: Tuple[int, ...]


RawForwardCallback = Callable[[Tuple[Mapping[str, Any], ...]], Any]
RawWindowForwardCallback = Callable[[RawReplayWindow], Any]
ModelStateContextFactory = Callable[[], ContextManager[Any]]


class FreshReplayGeometryBuilder:
    """Re-embed raw replay and capture every adapted layer's input geometry.

    ``physical_modules`` normally comes from ``InjectionResult.physical_modules``.
    Linear wrappers expose one ``logical_id``; grouped Conv2d wrappers expose
    one logical group per convolution group.  Bare Linear/Conv2d modules are
    also accepted for small integrations and use the mapping path as their
    stable logical id.
    """

    def __init__(
        self,
        physical_modules: Mapping[str, nn.Module],
        manifest_entries: Optional[Sequence[Any]] = None,
        maximum_rank: int = 64,
        spectral_energy_threshold: float = 0.999,
        minimum_energy: float = 1.0e-8,
        epsilon: float = 1.0e-8,
        maximum_activation_rows: Optional[int] = None,
    ) -> None:
        if not physical_modules:
            raise ValueError("fresh replay geometry requires adapted physical modules")
        if int(maximum_rank) < 0:
            raise ValueError("maximum_rank must be non-negative")
        if not 0.0 < float(spectral_energy_threshold) <= 1.0:
            raise ValueError("spectral_energy_threshold must be in (0, 1]")
        if not math.isfinite(float(minimum_energy)) or float(minimum_energy) < 0:
            raise ValueError("minimum_energy must be finite and non-negative")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
            raise ValueError("epsilon must be finite and positive")
        if maximum_activation_rows is not None and int(maximum_activation_rows) <= 0:
            raise ValueError("maximum_activation_rows must be positive when provided")
        self.maximum_rank = int(maximum_rank)
        self.spectral_energy_threshold = float(spectral_energy_threshold)
        self.minimum_energy = float(minimum_energy)
        self.epsilon = float(epsilon)
        self.maximum_activation_rows = (
            None
            if maximum_activation_rows is None
            else int(maximum_activation_rows)
        )
        self._specs = self._build_specs(physical_modules, manifest_entries)

    @staticmethod
    def _build_specs(
        physical_modules: Mapping[str, nn.Module],
        manifest_entries: Optional[Sequence[Any]],
    ) -> Tuple[_CaptureSpec, ...]:
        entries_by_path: Optional[Dict[str, list[Any]]] = None
        if manifest_entries is not None:
            raw_entries = getattr(manifest_entries, "entries", manifest_entries)
            entries_by_path = {}
            for entry in raw_entries:
                if not bool(getattr(entry, "injected", True)):
                    continue
                entry_path = str(getattr(entry, "module_path", ""))
                logical_id = str(getattr(entry, "logical_layer_id", ""))
                if not entry_path or not logical_id:
                    raise ValueError("manifest geometry entries require path and logical id")
                entries_by_path.setdefault(entry_path, []).append(entry)
        specs = []
        seen_modules: set[int] = set()
        seen_logical_ids: set[str] = set()
        for raw_path, module in sorted(physical_modules.items(), key=lambda item: str(item[0])):
            path = str(raw_path)
            if not path or not isinstance(module, nn.Module):
                raise ValueError("physical module paths/modules must be valid")
            if id(module) in seen_modules:
                raise ValueError("one physical module cannot be registered under multiple paths")
            seen_modules.add(id(module))
            base = getattr(module, "base_layer", module)
            path_entries = None if entries_by_path is None else entries_by_path.get(path)
            if entries_by_path is not None and not path_entries:
                raise ValueError(
                    f"physical module {path!r} has no injected manifest entry"
                )
            if isinstance(base, nn.Linear):
                if path_entries is not None:
                    if len(path_entries) != 1:
                        raise ValueError(
                            f"Linear module {path!r} must have one manifest entry"
                        )
                    logical_id = str(path_entries[0].logical_layer_id)
                else:
                    logical_id = str(getattr(module, "logical_id", None) or path)
                logical_ids = (logical_id,)
                input_dims = (int(base.in_features),)
            elif isinstance(base, nn.Conv2d):
                groups = int(base.groups)
                logical_groups = tuple(getattr(module, "logical_groups", ()))
                if logical_groups and len(logical_groups) != groups:
                    raise ValueError("Conv2d wrapper logical group count differs from base layer")
                if path_entries is not None:
                    by_group = {
                        int(getattr(entry, "logical_group", 0) or 0): entry
                        for entry in path_entries
                    }
                    if set(by_group) != set(range(groups)) or len(path_entries) != groups:
                        raise ValueError(
                            f"Conv2d module {path!r} manifest groups are incomplete"
                        )
                    logical_ids = tuple(
                        str(by_group[index].logical_layer_id)
                        for index in range(groups)
                    )
                else:
                    logical_ids = tuple(
                        str(
                            getattr(logical_groups[index], "logical_id", None)
                            or (
                                path
                                if groups == 1
                                else f"{path}::group={index}"
                            )
                        )
                        for index in range(groups)
                    )
                features = (
                    int(base.in_channels)
                    // groups
                    * int(base.kernel_size[0])
                    * int(base.kernel_size[1])
                )
                input_dims = tuple(features for _ in range(groups))
            else:
                raise TypeError(
                    f"geometry module {path!r} must wrap Linear or Conv2d"
                )
            for logical_id in logical_ids:
                if not logical_id or logical_id in seen_logical_ids:
                    raise ValueError(f"duplicate/empty logical layer id {logical_id!r}")
                seen_logical_ids.add(logical_id)
            specs.append(
                _CaptureSpec(
                    path=path,
                    module=module,
                    base_layer=base,
                    logical_layer_ids=logical_ids,
                    input_dims=input_dims,
                )
            )
        return tuple(specs)

    @property
    def logical_layer_ids(self) -> Tuple[str, ...]:
        return tuple(
            logical_id
            for spec in self._specs
            for logical_id in spec.logical_layer_ids
        )

    def _empty_result(
        self,
        *,
        persistent_model_version: int,
    ) -> GeometryBuildResult:
        geometries: Dict[str, ReplayGeometry] = {}
        for spec in self._specs:
            try:
                reference = next(spec.module.parameters())
                device = reference.device
            except StopIteration:
                device = torch.device("cpu")
            for logical_id, input_dim in zip(
                spec.logical_layer_ids,
                spec.input_dims,
            ):
                geometries[logical_id] = ReplayGeometry.empty(
                    input_dim,
                    device=device,
                    q_dtype=torch.float32,
                )
        return GeometryBuildResult(
            by_layer=geometries,
            window_count=0,
            reembed_latency_s=0.0,
            persistent_model_version=int(persistent_model_version),
        )

    def build(
        self,
        raw_replay: Sequence[RawReplayWindow],
        forward_window: Optional[RawWindowForwardCallback] = None,
        *,
        forward_callback: Optional[RawForwardCallback] = None,
        persistent_model_version: int = 0,
        model_state_context: Optional[ModelStateContextFactory] = None,
    ) -> GeometryBuildResult:
        """Build fresh geometry from raw inputs under caller-installed state.

        The primary ``forward_window`` callback receives one sanitized raw
        window at a time.  Its optional latent cache is removed before the
        call.  The batch-oriented ``forward_callback`` compatibility form
        instead receives a tuple of ``{"obs", "actions"}`` payloads.  Exactly
        one callback form is required.  A supplied ``model_state_context``
        must install ``W_before`` and restore the live fast state in its
        ``finally`` path.
        """

        version = int(persistent_model_version)
        if version < 0:
            raise ValueError("persistent_model_version must be non-negative")
        windows = tuple(raw_replay)
        if not windows:
            return self._empty_result(persistent_model_version=version)
        if (forward_window is None) == (forward_callback is None):
            raise TypeError("provide exactly one of forward_window or forward_callback")
        if forward_window is not None and not callable(forward_window):
            raise TypeError("forward_window must be callable")
        if forward_callback is not None and not callable(forward_callback):
            raise TypeError("forward_callback must be callable")
        if len({window.window_id for window in windows}) != len(windows):
            raise GeometryError("fresh geometry requires unique raw replay windows")
        for window in windows:
            if not isinstance(window, RawReplayWindow):
                raise GeometryError("fresh geometry accepts only RawReplayWindow values")
            window.validate()

        captured: Dict[str, list[Tensor]] = {
            logical_id: [] for logical_id in self.logical_layer_ids
        }
        output_energy_sum: Dict[str, float] = {
            logical_id: 0.0 for logical_id in self.logical_layer_ids
        }
        output_row_count: Dict[str, int] = {
            logical_id: 0 for logical_id in self.logical_layer_ids
        }
        handles = []

        def make_hook(spec: _CaptureSpec):
            def hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
                if not args or not torch.is_tensor(args[0]):
                    raise GeometryError(
                        f"adapted module {spec.path!r} received no tensor input"
                    )
                if not torch.is_tensor(output):
                    raise GeometryError(
                        f"adapted module {spec.path!r} produced a non-tensor output"
                    )
                input_tensor = args[0].detach()
                output_tensor = output.detach().to(dtype=torch.float32)
                if not torch.isfinite(output_tensor).all():
                    raise GeometryError(
                        f"adapted module {spec.path!r} produced non-finite output"
                    )
                if isinstance(spec.base_layer, nn.Linear):
                    if input_tensor.shape[-1] != spec.base_layer.in_features:
                        raise GeometryError(
                            f"Linear input dimension changed for {spec.path!r}"
                        )
                    if output_tensor.shape[-1] != spec.base_layer.out_features:
                        raise GeometryError(
                            f"Linear output dimension changed for {spec.path!r}"
                        )
                    logical_id = spec.logical_layer_ids[0]
                    matrix = input_tensor.reshape(
                        -1, input_tensor.shape[-1]
                    ).to(dtype=torch.float32)
                    output_matrix = output_tensor.reshape(
                        -1, output_tensor.shape[-1]
                    )
                    captured[logical_id].append(matrix)
                    output_energy_sum[logical_id] += float(
                        output_matrix.to(dtype=torch.float64).square().sum().cpu()
                    )
                    output_row_count[logical_id] += int(output_matrix.shape[0])
                    return

                assert isinstance(spec.base_layer, nn.Conv2d)
                matrices = conv2d_group_activation_matrices(
                    spec.base_layer,
                    input_tensor,
                )
                if output_tensor.ndim != 4:
                    raise GeometryError(
                        f"Conv2d output for {spec.path!r} must be N x C x H x W"
                    )
                out_per_group = spec.base_layer.out_channels // spec.base_layer.groups
                for group_index, (logical_id, matrix) in enumerate(
                    zip(spec.logical_layer_ids, matrices)
                ):
                    start = group_index * out_per_group
                    group_output = output_tensor[
                        :, start : start + out_per_group
                    ].movedim(1, -1).reshape(-1, out_per_group)
                    captured[logical_id].append(matrix)
                    output_energy_sum[logical_id] += float(
                        group_output.to(dtype=torch.float64).square().sum().cpu()
                    )
                    output_row_count[logical_id] += int(group_output.shape[0])

            return hook

        for spec in self._specs:
            handles.append(spec.module.register_forward_hook(make_hook(spec)))

        context_factory = model_state_context or contextlib.nullcontext
        started = time.perf_counter()
        try:
            with context_factory():
                with torch.no_grad():
                    if forward_window is not None:
                        for window in windows:
                            sanitized = window.clone()
                            sanitized.optional_latent_cache = None
                            sanitized.optional_latent_cache_model_version = -1
                            sanitized.validate()
                            forward_window(sanitized)
                    else:
                        assert forward_callback is not None
                        payloads = tuple(
                            window.to_model_payload() for window in windows
                        )
                        forward_callback(payloads)
        finally:
            for handle in handles:
                handle.remove()
        reembed_latency = time.perf_counter() - started

        geometries: Dict[str, ReplayGeometry] = {}
        input_dims = {
            logical_id: input_dim
            for spec in self._specs
            for logical_id, input_dim in zip(
                spec.logical_layer_ids,
                spec.input_dims,
            )
        }
        for logical_id in self.logical_layer_ids:
            values = captured[logical_id]
            if not values or output_row_count[logical_id] <= 0:
                raise GeometryError(
                    f"raw replay forward did not traverse adapted layer {logical_id!r}"
                )
            device = values[0].device
            matrix = torch.cat(
                [value.to(device=device, dtype=torch.float32) for value in values],
                dim=0,
            )
            if matrix.shape[1] != input_dims[logical_id]:
                raise GeometryError(
                    f"captured input dimension changed for {logical_id!r}"
                )
            if self.maximum_activation_rows is not None and (
                matrix.shape[0] > self.maximum_activation_rows
            ):
                indices = torch.linspace(
                    0,
                    matrix.shape[0] - 1,
                    self.maximum_activation_rows,
                    device=matrix.device,
                ).round().long()
                matrix = matrix.index_select(0, indices)
            subspace, tail = ActivationSubspace.from_activations_with_tail(
                matrix,
                maximum_rank=self.maximum_rank,
                spectral_energy_threshold=self.spectral_energy_threshold,
                minimum_energy=self.minimum_energy,
            )
            geometries[logical_id] = ReplayGeometry(
                q=subspace.q,
                eigenvalues=subspace.energies.to(dtype=torch.float64),
                tail_upper_bound=tail,
                sample_count=int(matrix.shape[0]),
                input_dim=int(matrix.shape[1]),
                output_energy=(
                    output_energy_sum[logical_id]
                    / float(output_row_count[logical_id])
                ),
            )
        return GeometryBuildResult(
            by_layer=geometries,
            window_count=len(windows),
            reembed_latency_s=reembed_latency,
            persistent_model_version=version,
        )


__all__ = [
    "FreshReplayGeometryBuilder",
    "GeometryBuildResult",
    "GeometryError",
    "ReplayGeometry",
]
