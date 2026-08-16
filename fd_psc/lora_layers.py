"""Full-depth FD-PSC LoRA layers.

The classes in this module deliberately keep the original layer as an
unmodified, frozen child module.  Persistent adapters consume/produce
*canonical* factors: ``B @ A`` is the exact effective delta and no caller has
to apply LoRA scaling a second time.

Large dense deltas are never materialised by production forward paths.  The
small materialisation helpers are guarded and exist for tests/diagnostics.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DEFAULT_MATERIALIZE_LIMIT = 1_000_000


@dataclass(frozen=True)
class CanonicalFactors:
    """A canonical low-rank matrix represented as ``B @ A``.

    Scaling is already absorbed in ``B``.  The object is iterable so existing
    code can naturally use ``B, A = adapter.get_*_factors()``.
    """

    B: Tensor
    A: Tensor

    def __post_init__(self) -> None:
        if self.B.ndim != 2 or self.A.ndim != 2:
            raise ValueError("canonical factors must both be rank-2 tensors")
        if self.B.shape[1] != self.A.shape[0]:
            raise ValueError(
                f"factor inner dimensions differ: B={tuple(self.B.shape)}, A={tuple(self.A.shape)}"
            )

    def __iter__(self) -> Iterator[Tensor]:
        yield self.B
        yield self.A

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Tensor:
        if index in (0, -2):
            return self.B
        if index in (1, -1):
            return self.A
        raise IndexError(index)

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.B.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.A.shape[1])

    def detach(self, clone: bool = False) -> "CanonicalFactors":
        b = self.B.detach()
        a = self.A.detach()
        if clone:
            b, a = b.clone(), a.clone()
        return CanonicalFactors(b, a)

    def materialize(self, maximum_elements: int = _DEFAULT_MATERIALIZE_LIMIT) -> Tensor:
        n = self.out_features * self.in_features
        if n > maximum_elements:
            raise RuntimeError(
                f"refusing to materialize {n} elements (limit={maximum_elements})"
            )
        return self.B @ self.A


FactorInput = Union[CanonicalFactors, Tuple[Tensor, Tensor], List[Tensor], Tensor]


def _factor_pair(
    value: FactorInput,
    a: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if isinstance(value, CanonicalFactors):
        if a is not None:
            raise ValueError("A must be omitted when CanonicalFactors is supplied")
        return value.B, value.A
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or a is not None:
            raise ValueError("factor sequence must be exactly (B, A)")
        return value[0], value[1]
    if not isinstance(value, Tensor) or a is None:
        raise TypeError("expected CanonicalFactors, (B, A), or B and A tensors")
    return value, a


def _concat_factors(
    factors: Iterable[CanonicalFactors],
    out_features: int,
    in_features: int,
    reference: Tensor,
) -> CanonicalFactors:
    nonempty = [f for f in factors if f.rank > 0]
    if not nonempty:
        return CanonicalFactors(
            reference.new_empty((out_features, 0)),
            reference.new_empty((0, in_features)),
        )
    return CanonicalFactors(
        torch.cat([f.B for f in nonempty], dim=1),
        torch.cat([f.A for f in nonempty], dim=0),
    )


class LogicalLoRAAdapter(nn.Module):
    """State and lifecycle shared by Linear and logical Conv2d groups."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        logical_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.requested_rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.logical_id = logical_id
        self.adapters_enabled = True
        self.pilot_frozen = False
        self.centered_active = False
        self.centered_alpha = float(alpha)
        self.active_exception_id: Optional[str] = None

        factory_kwargs = {"device": device, "dtype": dtype}
        empty_b = torch.empty((self.out_features, 0), **factory_kwargs)
        empty_a = torch.empty((0, self.in_features), **factory_kwargs)
        # FSD V2 core memory is a persistent full-depth delta.  It is kept
        # separate from theta_0 and from every low-rank branch so the official
        # checkpoint remains bitwise immutable and rank recycling has an
        # auditable destination.
        self.register_buffer(
            "core_delta",
            torch.zeros((self.out_features, self.in_features), **factory_kwargs),
            persistent=True,
        )
        self.register_buffer("slow_B", empty_b.clone(), persistent=True)
        self.register_buffer("slow_A", empty_a.clone(), persistent=True)
        self.register_buffer("exception_B", empty_b.clone(), persistent=True)
        self.register_buffer("exception_A", empty_a.clone(), persistent=True)
        self.register_buffer("center_B0", empty_b.clone(), persistent=True)
        self.register_buffer("center_A0", empty_a.clone(), persistent=True)

        self.register_parameter("pilot_B", None)
        self.register_parameter("pilot_A", None)
        self.register_parameter("center_B", None)
        self.register_parameter("center_A", None)
        self.episode_dropout = nn.Dropout(dropout)
        self.begin_episode(generator=generator, clear_exception=False)

    @property
    def actual_rank(self) -> int:
        return min(self.requested_rank, self.in_features, self.out_features)

    @property
    def pilot_actual_rank(self) -> int:
        return 0 if self.pilot_A is None else int(self.pilot_A.shape[0])

    @property
    def centered_actual_rank(self) -> int:
        return 0 if self.center_A is None else int(self.center_A.shape[0])

    @property
    def pilot_scaling(self) -> float:
        r = self.pilot_actual_rank
        return 0.0 if r == 0 else self.alpha / r

    @property
    def centered_scaling(self) -> float:
        r = self.centered_actual_rank
        return 0.0 if r == 0 else self.centered_alpha / r

    # Common aliases used by LoRA tooling.
    @property
    def lora_A(self) -> Optional[nn.Parameter]:
        return self.pilot_A

    @property
    def lora_B(self) -> Optional[nn.Parameter]:
        return self.pilot_B

    def _reference(self) -> Tensor:
        if self.pilot_A is not None:
            return self.pilot_A
        return self.slow_A

    def _empty_factors(self) -> CanonicalFactors:
        ref = self._reference()
        return CanonicalFactors(
            ref.new_empty((self.out_features, 0)),
            ref.new_empty((0, self.in_features)),
        )

    def _new_pilot(
        self,
        rank: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[nn.Parameter, nn.Parameter]:
        ref = self._reference()
        a = torch.empty((rank, self.in_features), device=ref.device, dtype=ref.dtype)
        # This is exactly Kaiming-uniform with a=sqrt(5): +/-1/sqrt(fan_in).
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            a.uniform_(-bound, bound, generator=generator)
        b = torch.zeros((self.out_features, rank), device=ref.device, dtype=ref.dtype)
        return nn.Parameter(b), nn.Parameter(a)

    def begin_episode(
        self,
        *,
        generator: Optional[torch.Generator] = None,
        clear_exception: bool = True,
    ) -> None:
        """Start a zero-function Pilot branch and discard all episodic state."""

        rank = self.actual_rank
        pilot_b, pilot_a = self._new_pilot(rank, generator)
        self.pilot_B = pilot_b
        self.pilot_A = pilot_a
        self.center_B = None
        self.center_A = None
        self.center_B0 = self.center_B0.new_empty((self.out_features, 0))
        self.center_A0 = self.center_A0.new_empty((0, self.in_features))
        self.centered_active = False
        self.centered_alpha = self.alpha
        self.pilot_frozen = False
        if clear_exception:
            self.clear_active_exception()

    def reset_episode(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.begin_episode(generator=generator, clear_exception=True)

    def freeze_pilot(self) -> None:
        if self.pilot_A is None or self.pilot_B is None:
            raise RuntimeError("cannot freeze a missing Pilot branch")
        self.pilot_A.requires_grad_(False)
        self.pilot_B.requires_grad_(False)
        self.pilot_frozen = True

    def activate_centered_branch(
        self,
        B0: Optional[FactorInput] = None,
        A0: Optional[Tensor] = None,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Freeze Pilot and activate ``s * (B A - B0 A0)``.

        ``B0/A0`` are the SLICE initialisation.  When omitted a standard
        zero-function LoRA initialisation is used.  In both cases the branch is
        exactly zero at the switching instant.
        """

        if self.centered_active:
            raise RuntimeError("the Centered branch may only be activated once per episode")
        self.freeze_pilot()
        center_alpha = self.alpha if alpha is None else float(alpha)
        if not math.isfinite(center_alpha) or center_alpha <= 0:
            raise ValueError("Centered alpha must be finite and positive")

        if B0 is None:
            requested = self.requested_rank if rank is None else int(rank)
            if requested <= 0:
                raise ValueError("Centered rank must be positive")
            actual = min(requested, self.in_features, self.out_features)
            b0, a0 = self._new_pilot(actual, generator)
            b = b0.detach()
            a = a0.detach()
        else:
            b, a = _factor_pair(B0, A0)
            if b.ndim != 2 or a.ndim != 2:
                raise ValueError("Centered factors must be matrices")
            if b.shape[0] != self.out_features or a.shape[1] != self.in_features:
                raise ValueError(
                    f"Centered factor shape mismatch for ({self.out_features}, {self.in_features})"
                )
            if b.shape[1] != a.shape[0] or b.shape[1] <= 0:
                raise ValueError("Centered factors need the same positive inner rank")
            actual = int(b.shape[1])
            if actual > min(self.in_features, self.out_features):
                raise ValueError("Centered factor rank exceeds logical matrix dimensions")
            if rank is not None and actual > int(rank):
                raise ValueError("provided Centered factors exceed requested rank")

        ref = self._reference()
        b = b.detach().to(device=ref.device, dtype=ref.dtype).clone()
        a = a.detach().to(device=ref.device, dtype=ref.dtype).clone()
        self.center_B0 = b.clone()
        self.center_A0 = a.clone()
        self.center_B = nn.Parameter(b.clone())
        self.center_A = nn.Parameter(a.clone())
        self.centered_alpha = center_alpha
        self.centered_active = True

    def _validated_persistent_factors(
        self,
        B: Tensor,
        A: Tensor,
        name: str,
    ) -> Tuple[Tensor, Tensor]:
        if B.ndim != 2 or A.ndim != 2:
            raise ValueError(f"{name} factors must be matrices")
        if B.shape[0] != self.out_features or A.shape[1] != self.in_features:
            raise ValueError(
                f"{name} factor shape mismatch: B={tuple(B.shape)}, A={tuple(A.shape)}, "
                f"expected outer shape ({self.out_features}, {self.in_features})"
            )
        if B.shape[1] != A.shape[0]:
            raise ValueError(f"{name} factor inner dimensions differ")
        if B.shape[1] > min(self.in_features, self.out_features):
            raise ValueError(f"{name} factor rank exceeds logical matrix dimensions")
        ref = self._reference()
        return (
            B.detach().to(device=ref.device, dtype=ref.dtype).clone(),
            A.detach().to(device=ref.device, dtype=ref.dtype).clone(),
        )

    def replace_slow_adapter(
        self,
        B: FactorInput,
        A: Optional[Tensor] = None,
    ) -> None:
        b, a = _factor_pair(B, A)
        b, a = self._validated_persistent_factors(b, a, "slow")
        self.slow_B, self.slow_A = b, a

    def get_core_delta(self) -> Tensor:
        return self.core_delta

    def replace_core_delta(self, value: Tensor) -> None:
        if not torch.is_tensor(value) or value.ndim != 2:
            raise ValueError("core_delta must be a matrix")
        if tuple(value.shape) != (self.out_features, self.in_features):
            raise ValueError(
                "core_delta shape mismatch: "
                f"got {tuple(value.shape)}, expected "
                f"({self.out_features}, {self.in_features})"
            )
        if not torch.isfinite(value).all():
            raise ValueError("core_delta must be finite")
        ref = self._reference()
        self.core_delta = value.detach().to(
            device=ref.device, dtype=ref.dtype
        ).clone()

    def set_active_exception(
        self,
        B: FactorInput,
        A: Optional[Tensor] = None,
        *,
        adapter_id: Optional[str] = None,
    ) -> None:
        b, a = _factor_pair(B, A)
        b, a = self._validated_persistent_factors(b, a, "exception")
        if a.shape[0] == 0:
            raise ValueError("active exception adapter cannot have rank zero")
        self.exception_B, self.exception_A = b, a
        self.active_exception_id = None if adapter_id is None else str(adapter_id)

    def clear_active_exception(self) -> None:
        self.exception_B = self.exception_B.new_empty((self.out_features, 0))
        self.exception_A = self.exception_A.new_empty((0, self.in_features))
        self.active_exception_id = None

    def get_slow_factors(self) -> CanonicalFactors:
        return CanonicalFactors(self.slow_B, self.slow_A)

    def get_exception_factors(self) -> CanonicalFactors:
        return CanonicalFactors(self.exception_B, self.exception_A)

    def get_episodic_factors(self) -> CanonicalFactors:
        if self.pilot_A is None or self.pilot_B is None:
            return self._empty_factors()
        parts = [
            CanonicalFactors(self.pilot_B * self.pilot_scaling, self.pilot_A),
        ]
        if self.centered_active:
            assert self.center_B is not None and self.center_A is not None
            parts.extend(
                [
                    CanonicalFactors(self.center_B * self.centered_scaling, self.center_A),
                    CanonicalFactors(-self.center_B0 * self.centered_scaling, self.center_A0),
                ]
            )
        return _concat_factors(
            parts, self.out_features, self.in_features, self._reference()
        )

    def get_effective_factors(self) -> CanonicalFactors:
        if not self.adapters_enabled:
            return self._empty_factors()
        return _concat_factors(
            [
                self.get_slow_factors(),
                self.get_exception_factors(),
                self.get_episodic_factors(),
            ],
            self.out_features,
            self.in_features,
            self._reference(),
        )

    def trainable_episode_parameters(self) -> List[nn.Parameter]:
        result: List[nn.Parameter] = []
        for p in (self.pilot_A, self.pilot_B, self.center_A, self.center_B):
            if p is not None and p.requires_grad:
                result.append(p)
        return result

    def disable_all_adapters(self) -> None:
        self.adapters_enabled = False

    def enable_all_adapters(self) -> None:
        self.adapters_enabled = True

    @contextlib.contextmanager
    def adapters_disabled(self) -> Iterator[None]:
        old = self.adapters_enabled
        self.adapters_enabled = False
        try:
            yield
        finally:
            self.adapters_enabled = old

    def adapter_state_dict(self) -> Dict[str, Any]:
        def copy(t: Optional[Tensor]) -> Optional[Tensor]:
            return None if t is None else t.detach().clone()

        return {
            "schema_version": 2,
            "logical_id": self.logical_id,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "requested_rank": self.requested_rank,
            "alpha": self.alpha,
            "dropout": self.dropout_p,
            "adapters_enabled": self.adapters_enabled,
            "core_delta": copy(self.core_delta),
            "slow_B": copy(self.slow_B),
            "slow_A": copy(self.slow_A),
            "exception_B": copy(self.exception_B),
            "exception_A": copy(self.exception_A),
            "active_exception_id": self.active_exception_id,
            "pilot_B": copy(self.pilot_B),
            "pilot_A": copy(self.pilot_A),
            "pilot_frozen": self.pilot_frozen,
            "centered_active": self.centered_active,
            "centered_alpha": self.centered_alpha,
            "center_B": copy(self.center_B),
            "center_A": copy(self.center_A),
            "center_B0": copy(self.center_B0),
            "center_A0": copy(self.center_A0),
        }

    def get_extra_state(self) -> Dict[str, Any]:
        """Lifecycle metadata needed by ordinary ``nn.Module.state_dict``."""

        return {
            "schema_version": 1,
            "logical_id": self.logical_id,
            "adapters_enabled": self.adapters_enabled,
            "pilot_frozen": self.pilot_frozen,
            "centered_active": self.centered_active,
            "centered_alpha": self.centered_alpha,
            "active_exception_id": self.active_exception_id,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported LoRA module extra-state schema")
        self.adapters_enabled = bool(state.get("adapters_enabled", True))
        self.pilot_frozen = bool(state.get("pilot_frozen", False))
        self.centered_active = bool(state.get("centered_active", False))
        self.centered_alpha = float(state.get("centered_alpha", self.alpha))
        self.active_exception_id = state.get("active_exception_id")
        if self.pilot_A is not None:
            self.pilot_A.requires_grad_(not self.pilot_frozen)
        if self.pilot_B is not None:
            self.pilot_B.requires_grad_(not self.pilot_frozen)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Resize dynamic-rank tensors before PyTorch's normal tensor copy."""

        ref = self._reference()

        def prepare_buffer(name: str) -> None:
            key = prefix + name
            value = state_dict.get(key)
            if isinstance(value, Tensor):
                setattr(
                    self,
                    name,
                    torch.empty(value.shape, device=ref.device, dtype=ref.dtype),
                )

        for name in (
            "slow_B",
            "slow_A",
            "exception_B",
            "exception_A",
            "center_B0",
            "center_A0",
        ):
            prepare_buffer(name)

        for name in ("pilot_B", "pilot_A", "center_B", "center_A"):
            value = state_dict.get(prefix + name)
            if isinstance(value, Tensor):
                setattr(
                    self,
                    name,
                    nn.Parameter(
                        torch.empty(value.shape, device=ref.device, dtype=ref.dtype)
                    ),
                )
            elif name.startswith("center_"):
                setattr(self, name, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_adapter_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("schema_version", -1))
        if version not in {1, 2}:
            raise ValueError(f"unsupported adapter state schema {version}")
        if int(state["in_features"]) != self.in_features or int(state["out_features"]) != self.out_features:
            raise ValueError("adapter state dimensions do not match logical layer")
        if int(state["requested_rank"]) != self.requested_rank:
            raise ValueError("adapter state requested rank does not match runtime")
        if float(state["alpha"]) != self.alpha:
            raise ValueError("adapter state alpha does not match runtime")

        ref = self._reference()

        def tensor(name: str) -> Tensor:
            value = state[name]
            if not isinstance(value, Tensor):
                raise ValueError(f"adapter state '{name}' must be a tensor")
            return value.detach().to(device=ref.device, dtype=ref.dtype).clone()

        if version >= 2:
            self.replace_core_delta(tensor("core_delta"))
        else:
            self.replace_core_delta(torch.zeros_like(self.core_delta))
        self.replace_slow_adapter(tensor("slow_B"), tensor("slow_A"))
        exc_b, exc_a = tensor("exception_B"), tensor("exception_A")
        if exc_a.shape[0]:
            self.set_active_exception(
                exc_b, exc_a, adapter_id=state.get("active_exception_id")
            )
        else:
            self.clear_active_exception()

        pilot_b, pilot_a = tensor("pilot_B"), tensor("pilot_A")
        self._validated_persistent_factors(pilot_b, pilot_a, "pilot")
        if pilot_a.shape[0] != self.actual_rank:
            raise ValueError(
                f"loaded Pilot rank {pilot_a.shape[0]} != runtime actual rank {self.actual_rank}"
            )
        self.pilot_B = nn.Parameter(pilot_b)
        self.pilot_A = nn.Parameter(pilot_a)
        self.pilot_frozen = bool(state.get("pilot_frozen", False))
        self.pilot_B.requires_grad_(not self.pilot_frozen)
        self.pilot_A.requires_grad_(not self.pilot_frozen)

        self.centered_active = bool(state.get("centered_active", False))
        self.centered_alpha = float(state.get("centered_alpha", self.alpha))
        if self.centered_active:
            center_b, center_a = tensor("center_B"), tensor("center_A")
            b0, a0 = tensor("center_B0"), tensor("center_A0")
            self._validated_persistent_factors(center_b, center_a, "center")
            self._validated_persistent_factors(b0, a0, "center initial")
            if center_b.shape != b0.shape or center_a.shape != a0.shape:
                raise ValueError("Centered current and initial factor shapes differ")
            self.center_B = nn.Parameter(center_b)
            self.center_A = nn.Parameter(center_a)
            self.center_B0, self.center_A0 = b0, a0
        else:
            self.center_B = None
            self.center_A = None
            self.center_B0 = self.center_B0.new_empty((self.out_features, 0))
            self.center_A0 = self.center_A0.new_empty((0, self.in_features))
        self.adapters_enabled = bool(state.get("adapters_enabled", True))

    def materialize_effective_delta(
        self, maximum_elements: int = _DEFAULT_MATERIALIZE_LIMIT
    ) -> Tensor:
        if self.core_delta.numel() > maximum_elements:
            raise RuntimeError(
                f"refusing to materialize {self.core_delta.numel()} adapter elements "
                f"(limit={maximum_elements})"
            )
        if not self.adapters_enabled:
            return torch.zeros_like(self.core_delta)
        low_rank = self.get_effective_factors().materialize(maximum_elements)
        return self.core_delta + low_rank

    def _apply_factors(self, x: Tensor, B: Tensor, A: Tensor) -> Tensor:
        raise NotImplementedError

    def _apply_core(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _adapter_delta(self, x: Tensor) -> Optional[Tensor]:
        if not self.adapters_enabled:
            return None
        result: Optional[Tensor] = None

        def add(value: Tensor) -> None:
            nonlocal result
            result = value if result is None else result + value

        # Keep the dense branch live even when it is numerically zero: during
        # Deep Sleep it is a leaf buffer with gradients enabled and must
        # receive the first optimization step from the zero state.
        add(self._apply_core(x))
        if self.slow_A.shape[0]:
            add(self._apply_factors(x, self.slow_B, self.slow_A))
        if self.exception_A.shape[0]:
            add(self._apply_factors(x, self.exception_B, self.exception_A))
        if self.pilot_A is not None and self.pilot_B is not None:
            pilot_x = self.episode_dropout(x)
            add(
                self._apply_factors(pilot_x, self.pilot_B, self.pilot_A)
                * self.pilot_scaling
            )
        if self.centered_active:
            assert self.center_A is not None and self.center_B is not None
            center_x = self.episode_dropout(x)
            current = self._apply_factors(center_x, self.center_B, self.center_A)
            initial = self._apply_factors(center_x, self.center_B0, self.center_A0)
            add((current - initial) * self.centered_scaling)
        return result


class DualLoRALinear(LogicalLoRAAdapter):
    """Frozen ``nn.Linear`` plus slow, routed exception, and episodic LoRA."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        generator: Optional[torch.Generator] = None,
        logical_id: Optional[str] = None,
    ) -> None:
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("DualLoRALinear requires an nn.Linear base layer")
        super().__init__(
            base_layer.in_features,
            base_layer.out_features,
            rank,
            alpha,
            dropout,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
            generator=generator,
            logical_id=logical_id,
        )
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_linear(cls, layer: nn.Linear, **kwargs: Any) -> "DualLoRALinear":
        return cls(layer, **kwargs)

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.base_layer.bias

    def _apply_factors(self, x: Tensor, B: Tensor, A: Tensor) -> Tensor:
        return F.linear(F.linear(x, A, bias=None), B, bias=None)

    def _apply_core(self, x: Tensor) -> Tensor:
        return F.linear(x, self.core_delta, bias=None)

    def forward(self, x: Tensor) -> Tensor:
        output = self.base_layer(x)
        delta = self._adapter_delta(x)
        return output if delta is None else output + delta


class ConvLoRAGroup(LogicalLoRAAdapter):
    """One independently routed logical group of a grouped Conv2d."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Union[str, Tuple[int, int]],
        dilation: Tuple[int, int],
        padding_mode: str,
        reversed_padding_repeated_twice: Sequence[int],
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        logical_id: Optional[str] = None,
    ) -> None:
        self.in_channels_per_group = int(in_channels)
        self.out_channels_per_group = int(out_channels)
        self.kernel_size = tuple(int(v) for v in kernel_size)
        self.stride = tuple(int(v) for v in stride)
        self.padding = padding if isinstance(padding, str) else tuple(int(v) for v in padding)
        self.dilation = tuple(int(v) for v in dilation)
        self.padding_mode = str(padding_mode)
        self.reversed_padding_repeated_twice = tuple(
            int(v) for v in reversed_padding_repeated_twice
        )
        flat_in = self.in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        super().__init__(
            flat_in,
            self.out_channels_per_group,
            rank,
            alpha,
            dropout,
            device=device,
            dtype=dtype,
            generator=generator,
            logical_id=logical_id,
        )

    def _prepare_input(self, x: Tensor) -> Tuple[Tensor, Union[str, Tuple[int, int]]]:
        if self.padding_mode != "zeros":
            x = F.pad(
                x,
                self.reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            return x, (0, 0)
        return x, self.padding

    def _apply_factors(self, x: Tensor, B: Tensor, A: Tensor) -> Tensor:
        rank = int(A.shape[0])
        if rank == 0:
            raise ValueError("cannot apply empty ConvLoRA factors")
        a_kernel = A.reshape(
            rank,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        padded, padding = self._prepare_input(x)
        hidden = F.conv2d(
            padded,
            a_kernel,
            bias=None,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=1,
        )
        b_kernel = B.reshape(self.out_channels_per_group, rank, 1, 1)
        return F.conv2d(hidden, b_kernel, bias=None, stride=1, padding=0)

    def _apply_core(self, x: Tensor) -> Tensor:
        kernel = self.core_delta.reshape(
            self.out_channels_per_group,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        padded, padding = self._prepare_input(x)
        return F.conv2d(
            padded,
            kernel,
            bias=None,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=1,
        )

    def materialize_effective_delta(
        self, maximum_elements: int = _DEFAULT_MATERIALIZE_LIMIT
    ) -> Tensor:
        flat = super().materialize_effective_delta(maximum_elements)
        return flat.reshape(
            self.out_channels_per_group,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def forward(self, x: Tensor) -> Tensor:
        delta = self._adapter_delta(x)
        if delta is None:
            # A logical group is only a delta branch; callers normally skip it
            # while disabled.  Returning an exact zero is convenient and safe.
            return x.new_zeros(
                (x.shape[0], self.out_channels_per_group, 0, 0)
            )
        return delta


class DualLoRAConv2d(nn.Module):
    """Frozen Conv2d with one independent :class:`ConvLoRAGroup` per group."""

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        generators: Optional[Sequence[Optional[torch.Generator]]] = None,
        logical_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError("DualLoRAConv2d requires an nn.Conv2d base layer")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        self.logical_id = logical_id
        self.groups = int(base_layer.groups)
        self.in_channels_per_group = base_layer.in_channels // self.groups
        self.out_channels_per_group = base_layer.out_channels // self.groups
        if generators is not None and len(generators) != self.groups:
            raise ValueError("one generator is required per Conv2d group")
        group_modules: List[ConvLoRAGroup] = []
        for group_index in range(self.groups):
            group_id = (
                f"{logical_id}::group={group_index}"
                if logical_id is not None
                else f"group={group_index}"
            )
            group_modules.append(
                ConvLoRAGroup(
                    in_channels=self.in_channels_per_group,
                    out_channels=self.out_channels_per_group,
                    kernel_size=base_layer.kernel_size,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    dilation=base_layer.dilation,
                    padding_mode=base_layer.padding_mode,
                    reversed_padding_repeated_twice=base_layer._reversed_padding_repeated_twice,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    device=base_layer.weight.device,
                    dtype=base_layer.weight.dtype,
                    generator=None if generators is None else generators[group_index],
                    logical_id=group_id,
                )
            )
        self.logical_groups = nn.ModuleList(group_modules)

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, **kwargs: Any) -> "DualLoRAConv2d":
        return cls(layer, **kwargs)

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.base_layer.bias

    @property
    def in_channels(self) -> int:
        return self.base_layer.in_channels

    @property
    def out_channels(self) -> int:
        return self.base_layer.out_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self.base_layer.kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        return self.base_layer.stride

    @property
    def padding(self) -> Union[str, Tuple[int, int]]:
        return self.base_layer.padding

    @property
    def dilation(self) -> Tuple[int, int]:
        return self.base_layer.dilation

    @property
    def padding_mode(self) -> str:
        return self.base_layer.padding_mode

    def iter_logical_groups(self) -> Iterator[ConvLoRAGroup]:
        yield from self.logical_groups

    def begin_episode(
        self,
        *,
        generators: Optional[Sequence[Optional[torch.Generator]]] = None,
        clear_exception: bool = True,
    ) -> None:
        if generators is not None and len(generators) != self.groups:
            raise ValueError("one generator is required per Conv2d group")
        for i, group in enumerate(self.logical_groups):
            group.begin_episode(
                generator=None if generators is None else generators[i],
                clear_exception=clear_exception,
            )

    def reset_episode(
        self,
        *,
        generators: Optional[Sequence[Optional[torch.Generator]]] = None,
    ) -> None:
        self.begin_episode(generators=generators, clear_exception=True)

    def freeze_pilot(self) -> None:
        for group in self.logical_groups:
            group.freeze_pilot()

    def activate_centered_branch(
        self,
        factors: Union[
            Mapping[Union[int, str], Union[CanonicalFactors, Tuple[Tensor, Tensor]]],
            Sequence[Union[CanonicalFactors, Tuple[Tensor, Tensor]]],
        ],
        *,
        alpha: Optional[float] = None,
    ) -> None:
        for i, group in enumerate(self.logical_groups):
            if isinstance(factors, Mapping):
                value = factors.get(i, factors.get(group.logical_id))
                if value is None:
                    raise KeyError(f"missing Centered factors for {group.logical_id}")
            else:
                value = factors[i]
            b, a = _factor_pair(value, None)
            group.activate_centered_branch(b, a, alpha=alpha)

    def get_slow_factors(self) -> Dict[str, CanonicalFactors]:
        return {g.logical_id or str(i): g.get_slow_factors() for i, g in enumerate(self.logical_groups)}

    def get_episodic_factors(self) -> Dict[str, CanonicalFactors]:
        return {g.logical_id or str(i): g.get_episodic_factors() for i, g in enumerate(self.logical_groups)}

    def get_effective_factors(self) -> Dict[str, CanonicalFactors]:
        return {g.logical_id or str(i): g.get_effective_factors() for i, g in enumerate(self.logical_groups)}

    def replace_slow_adapter(
        self,
        factors: Mapping[Union[int, str], Union[CanonicalFactors, Tuple[Tensor, Tensor]]],
    ) -> None:
        for i, group in enumerate(self.logical_groups):
            value = factors.get(i, factors.get(group.logical_id))
            if value is None:
                raise KeyError(f"missing slow factors for {group.logical_id}")
            group.replace_slow_adapter(value)

    def set_active_exception(
        self,
        factors: Mapping[Union[int, str], Union[CanonicalFactors, Tuple[Tensor, Tensor]]],
        *,
        adapter_id: Optional[str] = None,
    ) -> None:
        for i, group in enumerate(self.logical_groups):
            value = factors.get(i, factors.get(group.logical_id))
            if value is None:
                raise KeyError(f"missing exception factors for {group.logical_id}")
            group.set_active_exception(value, adapter_id=adapter_id)

    def clear_active_exception(self) -> None:
        for group in self.logical_groups:
            group.clear_active_exception()

    def disable_all_adapters(self) -> None:
        for group in self.logical_groups:
            group.disable_all_adapters()

    def enable_all_adapters(self) -> None:
        for group in self.logical_groups:
            group.enable_all_adapters()

    def trainable_episode_parameters(self) -> List[nn.Parameter]:
        return [p for g in self.logical_groups for p in g.trainable_episode_parameters()]

    def adapter_state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "logical_id": self.logical_id,
            "groups": [g.adapter_state_dict() for g in self.logical_groups],
        }

    def load_adapter_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) not in {1, 2}:
            raise ValueError("unsupported ConvLoRA adapter state schema")
        groups = state.get("groups")
        if not isinstance(groups, Sequence) or len(groups) != self.groups:
            raise ValueError("ConvLoRA checkpoint group count differs from base Conv2d")
        for group, payload in zip(self.logical_groups, groups):
            group.load_adapter_state_dict(payload)

    def materialize_effective_delta(
        self, maximum_elements: int = _DEFAULT_MATERIALIZE_LIMIT
    ) -> Tensor:
        total = self.base_layer.weight.numel()
        if total > maximum_elements:
            raise RuntimeError(
                f"refusing to materialize {total} Conv2d elements (limit={maximum_elements})"
            )
        return torch.cat(
            [g.materialize_effective_delta(maximum_elements) for g in self.logical_groups],
            dim=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.base_layer(x)
        deltas: List[Tensor] = []
        for group_index, group in enumerate(self.logical_groups):
            if not group.adapters_enabled:
                continue
            start = group_index * self.in_channels_per_group
            x_group = x[:, start : start + self.in_channels_per_group]
            delta = group._adapter_delta(x_group)
            if delta is None:
                # A zero Pilot still returns a tensor, so this occurs only if a
                # caller explicitly removed all branches.
                delta = output[:, : self.out_channels_per_group].new_zeros(
                    output.shape[0],
                    self.out_channels_per_group,
                    *output.shape[2:],
                )
            deltas.append(delta)
        if not deltas:
            return output
        if len(deltas) != self.groups:
            # Disabled groups contribute zero without introducing cross-group
            # parameters or computations.
            complete: List[Tensor] = []
            delta_iter = iter(deltas)
            for group in self.logical_groups:
                if group.adapters_enabled:
                    complete.append(next(delta_iter))
                else:
                    complete.append(
                        output[:, : self.out_channels_per_group].new_zeros(
                            output.shape[0],
                            self.out_channels_per_group,
                            *output.shape[2:],
                        )
                    )
            deltas = complete
        return output + torch.cat(deltas, dim=1)


# Backward-compatible spellings used in design notes and external experiments.
DualLoRAConv = DualLoRAConv2d


__all__ = [
    "CanonicalFactors",
    "LogicalLoRAAdapter",
    "DualLoRALinear",
    "ConvLoRAGroup",
    "DualLoRAConv2d",
    "DualLoRAConv",
]
