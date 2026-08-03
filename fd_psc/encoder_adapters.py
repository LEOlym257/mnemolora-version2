"""Explicit frozen-visual-latent protocols for AdaJEPA encoders.

Replay is cut after the permanently frozen visual backbone and before any
trainable projection head.  The protocol is explicit rather than hook-only so
stored layouts can be validated and replayed without rerunning the backbone.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


LATENT_SCHEMA_VERSION = "fd-psc-frozen-visual-latent-v1"


class UnsupportedEncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenVisualLatent:
    """Serializable tensor plus enough layout metadata to replay projection."""

    tensor: Tensor
    layout: str
    encoder_type: str
    cut_path: str
    schema_version: str = LATENT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, Tensor):
            raise TypeError("FrozenVisualLatent.tensor must be a torch.Tensor")
        if self.layout not in {"tokens", "feature_map", "vector"}:
            raise ValueError(f"unsupported frozen latent layout: {self.layout!r}")
        if self.schema_version != LATENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported frozen latent schema: {self.schema_version!r}")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def to(self, *args: Any, **kwargs: Any) -> "FrozenVisualLatent":
        return dataclasses.replace(self, tensor=self.tensor.to(*args, **kwargs))

    def detach(self, clone: bool = False) -> "FrozenVisualLatent":
        tensor = self.tensor.detach()
        if clone:
            tensor = tensor.clone()
        return dataclasses.replace(self, tensor=tensor)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "tensor": self.tensor,
            "layout": self.layout,
            "encoder_type": self.encoder_type,
            "cut_path": self.cut_path,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenVisualLatent":
        return cls(
            tensor=payload["tensor"],
            layout=str(payload["layout"]),
            encoder_type=str(payload["encoder_type"]),
            cut_path=str(payload["cut_path"]),
            schema_version=str(payload["schema_version"]),
            metadata=dict(payload.get("metadata", {})),
        )


class VisualLatentAdapter:
    """Non-Module protocol object; attaching it never alters model state_dict."""

    projection_module_paths: Tuple[str, ...] = ()
    frozen_module_paths: Tuple[str, ...] = ()
    has_projection_head: bool = False

    def __init__(self, encoder: nn.Module, wm: Optional[nn.Module] = None) -> None:
        self.encoder = encoder
        self.wm = wm
        self.encoder_type = f"{type(encoder).__module__}.{type(encoder).__qualname__}"

    def extract_frozen_visual_latent(
        self, obs: Union[Tensor, Mapping[str, Tensor]]
    ) -> FrozenVisualLatent:
        raise NotImplementedError

    def project_visual_latent(self, latent: FrozenVisualLatent) -> Tensor:
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        """Freeze all encoder base parameters; adapters are injected afterwards."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for module in self._frozen_modules():
            module.eval()
        # All base BatchNorm buffers are theta_0 and must never update.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def enforce_frozen_eval(self) -> None:
        for module in self._frozen_modules():
            module.eval()
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _frozen_modules(self) -> Sequence[nn.Module]:
        if not self.frozen_module_paths:
            return (self.encoder,)
        result = []
        for path in self.frozen_module_paths:
            result.append(self.encoder.get_submodule(path))
        return result

    def _visual_input(
        self, obs: Union[Tensor, Mapping[str, Tensor]]
    ) -> Tuple[Tensor, Tuple[int, ...]]:
        visual = obs["visual"] if isinstance(obs, Mapping) else obs
        if not isinstance(visual, Tensor):
            raise TypeError("obs visual input must be a tensor")
        if visual.ndim < 3:
            raise ValueError("visual input must have at least C,H,W dimensions")
        leading = tuple(int(v) for v in visual.shape[:-3])
        flat = visual.reshape(-1, *visual.shape[-3:])
        transform = getattr(self.wm, "encoder_transform", None)
        if transform is not None:
            flat = transform(flat)
        return flat, leading

    @staticmethod
    def _restore_leading(tensor: Tensor, leading: Tuple[int, ...]) -> Tensor:
        if not leading:
            return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor
        return tensor.reshape(*leading, *tensor.shape[1:])

    def _validate(self, latent: FrozenVisualLatent) -> None:
        if latent.schema_version != LATENT_SCHEMA_VERSION:
            raise ValueError(
                f"latent schema mismatch: {latent.schema_version!r} != {LATENT_SCHEMA_VERSION!r}"
            )
        if latent.encoder_type != self.encoder_type:
            raise ValueError(
                f"latent encoder mismatch: {latent.encoder_type!r} != {self.encoder_type!r}"
            )


class DinoVisualLatentAdapter(VisualLatentAdapter):
    """DINOv2 cut: ``base_model.forward_features`` -> optional projector."""

    frozen_module_paths = ("base_model",)

    def __init__(self, encoder: nn.Module, wm: Optional[nn.Module] = None) -> None:
        super().__init__(encoder, wm)
        if not hasattr(encoder, "base_model") or not hasattr(encoder, "feature_key"):
            raise UnsupportedEncoderError("DINO adapter requires base_model and feature_key")
        projector_name = getattr(encoder, "projector_name", "none")
        self.has_projection_head = bool(
            hasattr(encoder, "projector") and projector_name in {"channel", "global"}
        )
        self.projection_module_paths = ("projector",) if self.has_projection_head else ()
        self.cut_path = f"encoder.base_model.forward_features[{encoder.feature_key!r}]"

    def extract_frozen_visual_latent(
        self, obs: Union[Tensor, Mapping[str, Tensor]]
    ) -> FrozenVisualLatent:
        visual, leading = self._visual_input(obs)
        self.encoder.base_model.eval()
        with torch.no_grad():
            features = self.encoder.base_model.forward_features(visual)
            if self.encoder.feature_key not in features:
                raise KeyError(
                    f"DINO feature key {self.encoder.feature_key!r} absent; "
                    f"available={sorted(features)}"
                )
            tensor = features[self.encoder.feature_key].detach()
        if tensor.ndim == 3:
            n = int(tensor.shape[1])
            side = int(math.isqrt(n))
            grid = (side, side) if side * side == n else None
            layout = "tokens"
        elif tensor.ndim == 2:
            grid = None
            layout = "vector"
        else:
            raise ValueError(f"unexpected DINO feature shape {tuple(tensor.shape)}")
        return FrozenVisualLatent(
            tensor=tensor,
            layout=layout,
            encoder_type=self.encoder_type,
            cut_path=self.cut_path,
            metadata={
                "leading_shape": leading,
                "feature_key": str(self.encoder.feature_key),
                "projector_name": str(getattr(self.encoder, "projector_name", "none")),
                "token_grid": grid,
            },
        )

    def project_visual_latent(self, latent: FrozenVisualLatent) -> Tensor:
        self._validate(latent)
        self.enforce_frozen_eval()
        if latent.cut_path != self.cut_path:
            raise ValueError("DINO latent cut path differs from runtime adapter")
        leading = tuple(int(v) for v in latent.metadata.get("leading_shape", ()))
        tensor = latent.tensor
        if self.has_projection_head:
            if latent.layout != "tokens" or tensor.ndim != 3:
                raise ValueError("DINO channel/global projector requires token latent layout")
            grid = latent.metadata.get("token_grid")
            if grid is None or len(grid) != 2 or int(grid[0]) * int(grid[1]) != tensor.shape[1]:
                raise ValueError(
                    "DINO replay is missing a valid token_grid for token->feature-map restoration"
                )
            h, w = int(grid[0]), int(grid[1])
            reference = next(self.encoder.projector.parameters(), None)
            if reference is not None:
                tensor = tensor.to(device=reference.device, dtype=reference.dtype)
            tensor = tensor.reshape(tensor.shape[0], h, w, tensor.shape[2])
            feature_map = tensor.permute(0, 3, 1, 2).contiguous()
            projected = self.encoder.projector(feature_map)
            if projected.ndim == 2:
                projected = projected.unsqueeze(1)
            elif projected.ndim == 4:
                projected = projected.flatten(2).transpose(1, 2).contiguous()
            elif projected.ndim != 3:
                raise ValueError(
                    f"DINO projector returned unsupported shape {tuple(projected.shape)}"
                )
        else:
            projected = tensor.unsqueeze(1) if latent.layout == "vector" else tensor
        return self._restore_leading(projected, leading)


class ProjectionVisualLatentAdapter(VisualLatentAdapter):
    """Known local encoders with a named post-backbone Linear head."""

    def __init__(
        self,
        encoder: nn.Module,
        wm: Optional[nn.Module],
        *,
        kind: str,
    ) -> None:
        super().__init__(encoder, wm)
        self.kind = kind
        if kind in {"small_resnet", "small_resnet_gem"}:
            self.projection_module_paths = ("projection",)
            self.frozen_module_paths = tuple(
                name for name, _ in encoder.named_children() if name != "projection"
            )
            self.cut_path = "encoder.projection:input"
            self.cut_layout = "vector"
        elif kind == "vit_encoder":
            self.projection_module_paths = ("to_out",)
            self.frozen_module_paths = tuple(
                name for name, _ in encoder.named_children() if name != "to_out"
            )
            self.cut_path = "encoder.to_out:input"
            self.cut_layout = "tokens"
        else:
            raise UnsupportedEncoderError(f"unknown projection adapter kind {kind!r}")
        self.has_projection_head = True

    def extract_frozen_visual_latent(
        self, obs: Union[Tensor, Mapping[str, Tensor]]
    ) -> FrozenVisualLatent:
        x, leading = self._visual_input(obs)
        self.enforce_frozen_eval()
        with torch.no_grad():
            if self.kind in {"small_resnet", "small_resnet_gem"}:
                for index in range(1, 6):
                    x = getattr(self.encoder, f"rb{index}")(x)
                if self.kind == "small_resnet_gem":
                    x = self.encoder.gem(x)
                x = self.encoder.flat(x)
            else:
                x = self.encoder.patch_to_embedding(x)
                x = x + self.encoder.pos_embedding[:, : x.shape[1]]
                x = self.encoder.dropout(x)
                x = self.encoder.transformer(x)
            x = x.detach()
        return FrozenVisualLatent(
            tensor=x,
            layout=self.cut_layout,
            encoder_type=self.encoder_type,
            cut_path=self.cut_path,
            metadata={"leading_shape": leading, "adapter_kind": self.kind},
        )

    def project_visual_latent(self, latent: FrozenVisualLatent) -> Tensor:
        self._validate(latent)
        self.enforce_frozen_eval()
        if latent.cut_path != self.cut_path or latent.layout != self.cut_layout:
            raise ValueError("projection latent layout/cut path mismatch")
        leading = tuple(int(v) for v in latent.metadata.get("leading_shape", ()))
        head = self.encoder.get_submodule(self.projection_module_paths[0])
        reference = next(head.parameters(), None)
        tensor = latent.tensor
        if reference is not None:
            tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        projected = head(tensor)
        if self.kind in {"small_resnet", "small_resnet_gem"}:
            projected = projected.unsqueeze(1)
        return self._restore_leading(projected, leading)


class IdentityVisualLatentAdapter(VisualLatentAdapter):
    """An entirely frozen encoder with no post-backbone projection head."""

    has_projection_head = False
    projection_module_paths = ()
    frozen_module_paths = ()

    def __init__(self, encoder: nn.Module, wm: Optional[nn.Module] = None) -> None:
        super().__init__(encoder, wm)
        self.cut_path = "encoder:output"

    def extract_frozen_visual_latent(
        self, obs: Union[Tensor, Mapping[str, Tensor]]
    ) -> FrozenVisualLatent:
        x, leading = self._visual_input(obs)
        self.encoder.eval()
        with torch.no_grad():
            tensor = self.encoder(x).detach()
        if tensor.ndim == 4:
            layout = "feature_map"
        elif tensor.ndim == 3:
            layout = "tokens"
        elif tensor.ndim == 2:
            layout = "vector"
        else:
            raise ValueError(
                f"identity encoder returned unsupported latent shape {tuple(tensor.shape)}"
            )
        return FrozenVisualLatent(
            tensor=tensor,
            layout=layout,
            encoder_type=self.encoder_type,
            cut_path=self.cut_path,
            metadata={"leading_shape": leading},
        )

    def project_visual_latent(self, latent: FrozenVisualLatent) -> Tensor:
        self._validate(latent)
        if latent.cut_path != self.cut_path:
            raise ValueError("identity latent cut path mismatch")
        leading = tuple(int(v) for v in latent.metadata.get("leading_shape", ()))
        tensor = latent.tensor
        if latent.layout == "vector":
            tensor = tensor.unsqueeze(1)
        return self._restore_leading(tensor, leading)


_IDENTITY_CLASS_NAMES = {
    "resnet18",
    "ResNetSpatial",
    "R3M",
    "DummyModel",
}


def _resolve_wm_and_encoder(value: nn.Module) -> Tuple[Optional[nn.Module], nn.Module]:
    if hasattr(value, "encoder") and isinstance(getattr(value, "encoder"), nn.Module):
        return value, value.encoder
    if not isinstance(value, nn.Module):
        raise TypeError("get_encoder_adapter expects a VWorldModel or nn.Module encoder")
    return None, value


def get_encoder_adapter(value: nn.Module) -> VisualLatentAdapter:
    """Resolve an explicit supported adapter or fail with an actionable error."""

    wm, encoder = _resolve_wm_and_encoder(value)
    cached = getattr(value, "_fd_psc_encoder_adapter", None)
    if isinstance(cached, VisualLatentAdapter) and cached.encoder is encoder:
        return cached

    class_name = type(encoder).__name__
    if hasattr(encoder, "base_model") and hasattr(encoder, "feature_key"):
        adapter: VisualLatentAdapter = DinoVisualLatentAdapter(encoder, wm)
    elif class_name == "SmallResNet" and hasattr(encoder, "projection"):
        adapter = ProjectionVisualLatentAdapter(encoder, wm, kind="small_resnet")
    elif class_name == "SmallResNetGeM" and hasattr(encoder, "projection"):
        adapter = ProjectionVisualLatentAdapter(encoder, wm, kind="small_resnet_gem")
    elif class_name == "ViTEncoder" and hasattr(encoder, "to_out"):
        adapter = ProjectionVisualLatentAdapter(encoder, wm, kind="vit_encoder")
    elif class_name in _IDENTITY_CLASS_NAMES:
        adapter = IdentityVisualLatentAdapter(encoder, wm)
    else:
        raise UnsupportedEncoderError(
            f"FD-PSC has no frozen-latent adapter for {type(encoder).__module__}.{class_name}. "
            "Implement an explicit extract_frozen_visual_latent/project_visual_latent "
            "adapter and declare its projection_module_paths; hook-only replay is not supported."
        )
    return adapter


def install_encoder_adapter_protocol(value: nn.Module) -> VisualLatentAdapter:
    """Resolve and cache a non-Module protocol without changing state_dict keys."""

    adapter = get_encoder_adapter(value)
    adapter.freeze_backbone()
    # VisualLatentAdapter is intentionally not an nn.Module, so this does not
    # register a child or alter checkpoint keys.
    setattr(value, "_fd_psc_encoder_adapter", adapter)
    return adapter


# Short alias used by integration code.
resolve_encoder_adapter = get_encoder_adapter


__all__ = [
    "LATENT_SCHEMA_VERSION",
    "UnsupportedEncoderError",
    "FrozenVisualLatent",
    "VisualLatentAdapter",
    "DinoVisualLatentAdapter",
    "ProjectionVisualLatentAdapter",
    "IdentityVisualLatentAdapter",
    "get_encoder_adapter",
    "resolve_encoder_adapter",
    "install_encoder_adapter_protocol",
]
