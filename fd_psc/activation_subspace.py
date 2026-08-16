"""Historical input-activation subspaces and factor-only soft-NESS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .low_rank_merge import FactorLike, LowRankFactors, as_factors


def conv2d_group_activation_matrices(
    module: nn.Conv2d,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Unfold Conv2d inputs into one ``N*L x group_din*kh*kw`` matrix per group.

    Padding is prepared exactly as ``nn.Conv2d._conv_forward`` does, including
    asymmetric string ``'same'`` padding and non-zero padding modes.  Keeping
    groups separate prevents cross-group activation-subspace contamination.
    """

    if not isinstance(module, nn.Conv2d):
        raise TypeError("module must be nn.Conv2d")
    if input_tensor.ndim != 4 or input_tensor.shape[1] != module.in_channels:
        raise ValueError("Conv2d activation input must have shape N x Cin x H x W")
    value = input_tensor.detach().to(dtype=torch.float32)
    if module.padding_mode != "zeros":
        value = F.pad(value, module._reversed_padding_repeated_twice, mode=module.padding_mode)
        padding = (0, 0)
    elif isinstance(module.padding, str):
        if module.padding not in ("same", "valid"):
            raise ValueError(f"unsupported Conv2d padding string: {module.padding}")
        value = F.pad(
            value,
            module._reversed_padding_repeated_twice,
            mode="constant",
            value=0.0,
        )
        padding = (0, 0)
    else:
        padding = tuple(int(item) for item in module.padding)
    if not torch.isfinite(value).all():
        raise ValueError("non-finite Conv2d inputs cannot update Q/lambda")
    unfolded = F.unfold(
        value,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=padding,
        stride=module.stride,
    )
    batch, _, locations = unfolded.shape
    features_per_group = (
        module.in_channels // module.groups
        * module.kernel_size[0]
        * module.kernel_size[1]
    )
    grouped = unfolded.reshape(batch, module.groups, features_per_group, locations)
    return tuple(
        grouped[:, group_index]
        .permute(0, 2, 1)
        .reshape(batch * locations, features_per_group)
        .contiguous()
        for group_index in range(module.groups)
    )


def _select_energy_rank(
    energies: torch.Tensor,
    threshold: float,
    maximum_rank: int,
    minimum_energy: float,
) -> int:
    """Select a prefix rank from covariance-scale activation energies.

    ``minimum_energy`` and the stored lambda values both have units
    of activation covariance.  Callers must therefore normalize raw
    activation singular values by the sample count before entering here;
    singular values from the incremental normalized sketch are already in
    these units after squaring.
    """

    if energies.numel() == 0 or maximum_rank <= 0:
        return 0
    covariance_energies = energies.to(dtype=torch.float64)
    valid = torch.isfinite(covariance_energies) & (
        covariance_energies > float(minimum_energy)
    )
    if not torch.any(valid):
        return 0
    covariance_energies = covariance_energies[valid]
    total = torch.sum(covariance_energies)
    if float(total.detach().cpu()) <= minimum_energy:
        return 0
    cumulative = torch.cumsum(covariance_energies, dim=0) / total
    reached = torch.nonzero(cumulative >= float(threshold), as_tuple=False)
    rank = (
        int(reached[0, 0].item()) + 1
        if reached.numel()
        else int(covariance_energies.numel())
    )
    return min(rank, int(maximum_rank), int(covariance_energies.numel()))


@dataclass(frozen=True)
class SoftNESSWeights:
    weights: torch.Tensor
    tau: float
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class ActivationSubspace:
    """Input-space basis ``Q`` (din x q) and matching activation energies."""

    q: torch.Tensor
    energies: torch.Tensor

    def __post_init__(self) -> None:
        if self.q.ndim != 2 or self.energies.ndim != 1:
            raise ValueError("Q must be a matrix and energies must be a vector")
        if self.q.shape[1] != self.energies.shape[0]:
            raise ValueError("Q rank and energy count differ")
        if self.q.device != self.energies.device:
            raise ValueError("Q and energies must share a device")
        if not torch.isfinite(self.q).all() or not torch.isfinite(self.energies).all():
            raise ValueError("Q and activation energies must be finite")
        if torch.any(self.energies < 0):
            raise ValueError("activation energies must be non-negative")

    @property
    def input_dim(self) -> int:
        return int(self.q.shape[0])

    @property
    def rank(self) -> int:
        return int(self.q.shape[1])

    @classmethod
    def empty(
        cls,
        input_dim: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ActivationSubspace":
        return cls(
            torch.empty(input_dim, 0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.float32),
        )

    @classmethod
    def from_activations(
        cls,
        activations: torch.Tensor,
        maximum_rank: int = 64,
        spectral_energy_threshold: float = 0.99,
        minimum_energy: float = 1.0e-8,
    ) -> "ActivationSubspace":
        """Build ``Q=Vh.T`` from ``H`` shaped ``N x din``."""

        state, _ = cls.from_activations_with_tail(
            activations,
            maximum_rank=maximum_rank,
            spectral_energy_threshold=spectral_energy_threshold,
            minimum_energy=minimum_energy,
        )
        return state

    @classmethod
    def from_activations_with_tail(
        cls,
        activations: torch.Tensor,
        maximum_rank: int = 64,
        spectral_energy_threshold: float = 0.99,
        minimum_energy: float = 1.0e-8,
    ) -> Tuple["ActivationSubspace", float]:
        """Build the retained subspace and return ``lambda_(m+1)``.

        This is the single activation-SVD implementation used by both legacy
        :meth:`from_activations` and FSD V2 fresh replay geometry.  The tail is
        the first discarded eigenvalue of ``H.T @ H / N``.  Consequently a
        non-empty activation matrix may have retained rank zero while still
        carrying a positive conservative tail bound.
        """

        if activations.ndim != 2:
            raise ValueError("activation matrix must have shape N x din")
        if not 0.0 < spectral_energy_threshold <= 1.0:
            raise ValueError("spectral_energy_threshold must be in (0, 1]")
        input_dim = int(activations.shape[1])
        if activations.shape[0] == 0 or input_dim == 0:
            return cls.empty(input_dim, device=activations.device), 0.0
        h = activations.detach().to(dtype=torch.float32)
        if not torch.isfinite(h).all():
            raise ValueError("non-finite activations cannot update Q/lambda")
        _, singular_values, vh = torch.linalg.svd(h, full_matrices=False)
        covariance_energies = singular_values.square() / float(h.shape[0])
        rank = _select_energy_rank(
            covariance_energies,
            spectral_energy_threshold,
            maximum_rank,
            minimum_energy,
        )
        tail_upper_bound = (
            float(covariance_energies[rank].detach().to(dtype=torch.float64).cpu())
            if rank < int(covariance_energies.numel())
            else 0.0
        )
        if rank == 0:
            return cls.empty(input_dim, device=h.device), tail_upper_bound
        # The input directions are right singular vectors, never U.
        q = vh[:rank].transpose(0, 1).contiguous()
        energies = covariance_energies[:rank]
        return cls(q, energies.to(dtype=torch.float32)), tail_upper_bound

    def soft_ness_weights(
        self,
        mode: str = "median",
        fixed_tau: Optional[float] = None,
        quantile: float = 0.5,
        minimum_energy: float = 1.0e-8,
    ) -> SoftNESSWeights:
        if self.rank == 0:
            return SoftNESSWeights(
                torch.empty(0, device=self.q.device, dtype=torch.float32),
                float(minimum_energy),
                False,
                "empty_history_safe_fallback",
            )
        positive = self.energies[
            torch.isfinite(self.energies) & (self.energies > float(minimum_energy))
        ]
        if positive.numel() == 0:
            return SoftNESSWeights(
                torch.zeros_like(self.energies, dtype=torch.float32),
                float(minimum_energy),
                False,
                "no_positive_historical_energy",
            )
        if mode == "median":
            tau_t = torch.quantile(positive, 0.5)
        elif mode == "quantile":
            if not 0.0 <= float(quantile) <= 1.0:
                raise ValueError("soft-NESS quantile must be in [0, 1]")
            tau_t = torch.quantile(positive, float(quantile))
        elif mode == "fixed":
            if fixed_tau is None or not torch.isfinite(torch.tensor(float(fixed_tau))) or fixed_tau <= 0:
                raise ValueError("fixed soft-NESS tau must be finite and positive")
            tau_t = torch.tensor(float(fixed_tau), device=self.q.device, dtype=torch.float32)
        else:
            raise ValueError(f"unknown soft-NESS tau mode: {mode}")
        tau = max(float(tau_t.detach().cpu()), float(minimum_energy))
        weights = self.energies.to(dtype=torch.float32) / (
            self.energies.to(dtype=torch.float32) + tau
        )
        weights = torch.clamp(weights, 0.0, 1.0)
        return SoftNESSWeights(weights, tau, True)

    def update(
        self,
        new_activations: torch.Tensor,
        forgetting_factor: float = 0.99,
        maximum_rank: int = 64,
        spectral_energy_threshold: float = 0.99,
        minimum_energy: float = 1.0e-8,
    ) -> "ActivationSubspace":
        """Apply the V2 incremental row-space sketch and return a new state."""

        if new_activations.ndim != 2 or new_activations.shape[1] != self.input_dim:
            raise ValueError("new activation matrix must have shape N x din")
        if not 0.0 <= float(forgetting_factor) <= 1.0:
            raise ValueError("forgetting_factor must be in [0, 1]")
        h = new_activations.detach().to(device=self.q.device, dtype=torch.float32)
        if not torch.isfinite(h).all():
            raise ValueError("non-finite activations cannot update Q/lambda")
        rows = []
        if self.rank:
            old = (
                float(forgetting_factor) ** 0.5
                * torch.sqrt(torch.clamp(self.energies, min=0.0)).unsqueeze(1)
                * self.q.to(dtype=torch.float32).transpose(0, 1)
            )
            rows.append(old)
        if h.shape[0]:
            new = (
                max(0.0, 1.0 - float(forgetting_factor)) ** 0.5
                * h
                / float(h.shape[0]) ** 0.5
            )
            rows.append(new)
        if not rows:
            return ActivationSubspace.empty(self.input_dim, device=self.q.device)
        sketch = torch.cat(rows, dim=0)
        _, singular_values, vh = torch.linalg.svd(sketch, full_matrices=False)
        covariance_energies = singular_values.square()
        rank = _select_energy_rank(
            covariance_energies,
            spectral_energy_threshold,
            maximum_rank,
            minimum_energy,
        )
        if rank == 0:
            return ActivationSubspace.empty(self.input_dim, device=self.q.device)
        return ActivationSubspace(
            vh[:rank].transpose(0, 1).contiguous(),
            covariance_energies[:rank].to(dtype=torch.float32),
        )

    def transform_right_factors(
        self,
        factors: FactorLike,
        alpha_shared: float,
        alpha_safe: float,
        weights: Optional[torch.Tensor] = None,
    ) -> LowRankFactors:
        """Apply ``alpha_safe I + (alpha_shared-alpha_safe) Q diag(p) Q.T``.

        Only the small ``rank x din`` right factor is produced; neither ``I``
        nor the dense ``din x din`` projector is ever constructed.
        """

        value = as_factors(factors)
        if value.in_features != self.input_dim:
            raise ValueError("factor input dimension does not match activation subspace")
        a = value.a.to(dtype=torch.float32)
        if self.rank == 0:
            transformed = float(alpha_safe) * a
        else:
            p = weights
            if p is None:
                p = self.soft_ness_weights().weights
            if p.ndim != 1 or p.shape[0] != self.rank:
                raise ValueError("soft-NESS weights must have one value per Q direction")
            q = self.q.to(device=a.device, dtype=torch.float32)
            protected = ((a @ q) * p.to(device=a.device, dtype=torch.float32).unsqueeze(0)) @ q.transpose(0, 1)
            transformed = float(alpha_safe) * a + (
                float(alpha_shared) - float(alpha_safe)
            ) * protected
        return LowRankFactors(value.b, transformed.to(dtype=value.a.dtype))

    def shared_safe_factors(
        self,
        factors: FactorLike,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[LowRankFactors, LowRankFactors]:
        value = as_factors(factors)
        if self.rank == 0:
            zero = LowRankFactors.zeros(
                value.out_features,
                value.in_features,
                device=value.b.device,
                dtype=value.b.dtype,
            )
            return zero, value
        p = weights if weights is not None else self.soft_ness_weights().weights
        q = self.q.to(device=value.a.device, dtype=torch.float32)
        a = value.a.to(dtype=torch.float32)
        shared_a = ((a @ q) * p.to(device=a.device, dtype=torch.float32).unsqueeze(0)) @ q.transpose(0, 1)
        return (
            LowRankFactors(value.b, shared_a.to(dtype=value.a.dtype)),
            LowRankFactors(value.b, (a - shared_a).to(dtype=value.a.dtype)),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"q": self.q.detach().clone(), "energies": self.energies.detach().clone()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "ActivationSubspace":
        return cls(state["q"].detach().clone(), state["energies"].detach().clone())


class ActivationSubspaceBank:
    """Checkpointable logical-layer map; Conv groups use separate stable ids."""

    def __init__(self) -> None:
        self._states: Dict[str, ActivationSubspace] = {}

    def get(self, logical_layer_id: str, input_dim: int, device: torch.device) -> ActivationSubspace:
        return self._states.get(
            logical_layer_id,
            ActivationSubspace.empty(input_dim, device=device),
        )

    def set(self, logical_layer_id: str, state: ActivationSubspace) -> None:
        self._states[str(logical_layer_id)] = state

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            key: value.state_dict()
            for key, value in sorted(self._states.items(), key=lambda item: item[0])
        }

    def load_state_dict(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self._states = {
            key: ActivationSubspace.from_state_dict(value)
            for key, value in sorted(state.items(), key=lambda item: item[0])
        }


def apply_soft_ness(
    factors: FactorLike,
    subspace: ActivationSubspace,
    alpha_shared: float,
    alpha_safe: float,
    weights: Optional[torch.Tensor] = None,
) -> LowRankFactors:
    return subspace.transform_right_factors(factors, alpha_shared, alpha_safe, weights)


__all__ = [
    "ActivationSubspace",
    "ActivationSubspaceBank",
    "SoftNESSWeights",
    "apply_soft_ness",
    "conv2d_group_activation_matrices",
]
