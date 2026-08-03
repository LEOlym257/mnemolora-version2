"""FD-PSC target discovery, reachability manifest, and adapter injection."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

import torch
from torch import Tensor, nn

from .encoder_adapters import (
    UnsupportedEncoderError,
    VisualLatentAdapter,
    get_encoder_adapter,
    install_encoder_adapter_protocol,
)
from .lora_layers import ConvLoRAGroup, DualLoRAConv2d, DualLoRALinear, LogicalLoRAAdapter


MANIFEST_SCHEMA_VERSION = "fd-psc-target-manifest-v1"


@dataclass(frozen=True)
class ManifestEntry:
    module_path: str
    logical_layer_id: str
    layer_type: str
    in_features: int
    out_features: int
    module_group: str
    role: str
    active_in_forward: bool
    default_inject: bool
    injected: bool
    actual_rank: int
    attention_output: bool = False
    mlp_output: bool = False
    final_projection: bool = False
    kernel_size: Optional[Tuple[int, int]] = None
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[Union[str, Tuple[int, int]]] = None
    dilation: Optional[Tuple[int, int]] = None
    groups: Optional[int] = None
    logical_group: Optional[int] = None
    bias: Optional[bool] = None
    active_detection: str = "structural"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # JSON has no tuples; normalise explicitly for cross-process hashing.
        for key in ("kernel_size", "stride", "dilation"):
            if result[key] is not None:
                result[key] = list(result[key])
        if isinstance(result["padding"], tuple):
            result["padding"] = list(result["padding"])
        return result


@dataclass(frozen=True)
class TargetManifest:
    entries: Tuple[ManifestEntry, ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def manifest_hash(self) -> str:
        return self.hash

    def __iter__(self) -> Iterator[ManifestEntry]:
        yield from self.entries

    def by_logical_id(self) -> Dict[str, ManifestEntry]:
        return {entry.logical_layer_id: entry for entry in self.entries}


@dataclass
class InjectionResult:
    adapters: Dict[str, LogicalLoRAAdapter]
    manifest: TargetManifest
    encoder_adapter: Optional[VisualLatentAdapter] = None
    _physical_modules: Dict[str, nn.Module] = field(default_factory=dict, repr=False)
    _base_eval_hook: Optional[Any] = field(default=None, repr=False)
    _seed: int = field(default=0, repr=False)
    _episode_counter: int = field(default=0, repr=False)

    def _parameters_for_group(self, group: str) -> List[nn.Parameter]:
        entries = self.manifest.by_logical_id()
        result: List[nn.Parameter] = []
        seen: Set[int] = set()
        for logical_id in sorted(self.adapters):
            entry = entries.get(logical_id)
            if entry is None or entry.module_group != group:
                continue
            for parameter in self.adapters[logical_id].trainable_episode_parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    result.append(parameter)
        return result

    def predictor_parameters(self) -> List[nn.Parameter]:
        return self._parameters_for_group("predictor")

    def encoder_parameters(self) -> List[nn.Parameter]:
        result: List[nn.Parameter] = []
        for group in ("encoder_projection", "action_encoder", "proprio_encoder"):
            result.extend(self._parameters_for_group(group))
        return result

    def gradient_modules(self) -> Dict[str, nn.Module]:
        """Physical wrappers for effective-weight hooks.

        A grouped Conv2d appears once under its module path; the hook collector
        subsequently splits its exact weight gradient into ``::group=<g>``
        logical matrices.  ``adapters`` remains the per-logical-layer registry.
        """

        result: Dict[str, nn.Module] = {}
        for path, module in self._physical_modules.items():
            # The existing hook collector preserves its supplied ID for a
            # groups=1 Conv and appends ::group=g only for grouped Conv.  Feed
            # it the canonical group-0 ID in the former case so hook keys and
            # the manifest registry are identical.
            if isinstance(module, DualLoRAConv2d) and module.groups == 1:
                result[f"{path}::group=0"] = module
            else:
                result[path] = module
        return result

    @property
    def physical_modules(self) -> Dict[str, nn.Module]:
        return dict(self._physical_modules)

    def begin_episode(self, episode_id: Optional[int] = None) -> None:
        if episode_id is None:
            episode_id = self._episode_counter
        episode_id = int(episode_id)
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        for logical_id in sorted(self.adapters):
            adapter = self.adapters[logical_id]
            generator = _generator(
                adapter._reference().device,
                _stable_seed(self._seed, f"episode={episode_id}\0{logical_id}"),
            )
            adapter.begin_episode(generator=generator, clear_exception=True)
        self._episode_counter = max(self._episode_counter, episode_id + 1)

    def reset_episode(self) -> None:
        for logical_id in sorted(self.adapters):
            adapter = self.adapters[logical_id]
            generator = _generator(
                adapter._reference().device,
                _stable_seed(self._seed, f"reset={self._episode_counter}\0{logical_id}"),
            )
            adapter.reset_episode(generator=generator)

    def enforce_frozen_base_eval(self) -> None:
        if self.encoder_adapter is not None:
            self.encoder_adapter.enforce_frozen_eval()

    def close(self) -> None:
        if self._base_eval_hook is not None:
            self._base_eval_hook.remove()
            self._base_eval_hook = None


@dataclass
class _Candidate:
    module_path: str
    module: Union[nn.Linear, nn.Conv2d]
    module_group: str
    role: str
    default_enabled: bool
    active_structural: bool


def _cfg_get(config: Any, dotted: str, default: Any) -> Any:
    value = config
    for part in dotted.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        else:
            if not hasattr(value, part):
                return default
            value = getattr(value, part)
    return value


def _stable_seed(seed: int, logical_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{logical_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    kind = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=kind)
    generator.manual_seed(seed)
    return generator


def _contains_injected_adapter(module: nn.Module) -> bool:
    return any(isinstance(child, (DualLoRALinear, DualLoRAConv2d)) for child in module.modules())


def _iter_linear_conv(module: nn.Module, prefix: str) -> Iterator[Tuple[str, Union[nn.Linear, nn.Conv2d]]]:
    for name, child in module.named_modules():
        if isinstance(child, (DualLoRALinear, DualLoRAConv2d)):
            raise RuntimeError(f"duplicate FD-PSC injection attempted under {prefix}.{name}".rstrip("."))
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            path = prefix if not name else f"{prefix}.{name}"
            if path == "base_model" or ".base_model." in f".{path}.":
                raise RuntimeError(f"frozen-backbone target escaped exclusion: {path}")
            yield path, child


def _predictor_role(path: str) -> str:
    if path.endswith(".to_qkv"):
        return "attention_qkv"
    if ".to_out." in path or path.endswith(".to_out"):
        return "attention_output"
    if path.endswith(".net.1"):
        return "mlp_input"
    if path.endswith(".net.4"):
        return "mlp_output"
    return "other_linear"


def _collect_candidates(
    wm: nn.Module,
    config: Any,
    encoder_adapter: VisualLatentAdapter,
) -> List[_Candidate]:
    result: List[_Candidate] = []
    predictor_enabled = bool(_cfg_get(config, "target_modules.predictor_linear", True))
    predictor = getattr(wm, "predictor", None)
    if predictor is not None:
        for path, module in _iter_linear_conv(predictor, "predictor"):
            if isinstance(module, nn.Linear):
                result.append(
                    _Candidate(path, module, "predictor", _predictor_role(path), predictor_enabled, True)
                )

    projection_linear = bool(
        _cfg_get(config, "target_modules.post_backbone_projection_linear", True)
    )
    conv_enabled = bool(_cfg_get(config, "conv_lora.enabled", True))
    encoder = wm.encoder
    for head_path in encoder_adapter.projection_module_paths:
        head = encoder.get_submodule(head_path)
        prefix = f"encoder.{head_path}"
        for path, module in _iter_linear_conv(head, prefix):
            enabled = projection_linear if isinstance(module, nn.Linear) else conv_enabled
            result.append(
                _Candidate(
                    path,
                    module,
                    "encoder_projection",
                    "encoder_projection" if isinstance(module, nn.Linear) else "encoder_projection_conv",
                    enabled,
                    True,
                )
            )

    for attr, flag_name, group in (
        ("action_encoder", "target_modules.action_encoder_linear", "action_encoder"),
        ("proprio_encoder", "target_modules.proprio_encoder_linear", "proprio_encoder"),
    ):
        enabled = bool(_cfg_get(config, flag_name, False))
        module = getattr(wm, attr, None)
        if module is None:
            continue
        dummy = type(module).__name__ in {"DummyModel", "DummyRepeatActionEncoder"}
        for path, child in _iter_linear_conv(module, attr):
            if isinstance(child, nn.Linear):
                result.append(
                    _Candidate(path, child, group, "other_linear", enabled, not dummy)
                )
    return result


def _capture_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_schema_dry_run(
    wm: nn.Module,
    candidates: Sequence[_Candidate],
    schema_sample: Any,
) -> Set[str]:
    reached: Set[str] = set()
    handles = []
    for candidate in candidates:
        handles.append(
            candidate.module.register_forward_hook(
                lambda _module, _args, _output, path=candidate.module_path: reached.add(path)
            )
        )
    rng = _capture_rng()
    training = {module: module.training for module in wm.modules()}
    buffers = {name: tensor.detach().clone() for name, tensor in wm.named_buffers()}
    mutable_attrs: Dict[Tuple[nn.Module, str], Any] = {}
    for module in wm.modules():
        if hasattr(module, "latent_ndim"):
            mutable_attrs[(module, "latent_ndim")] = getattr(module, "latent_ndim")
    try:
        wm.eval()
        with torch.no_grad():
            if callable(schema_sample):
                schema_sample(wm)
            elif isinstance(schema_sample, Mapping):
                if "call" in schema_sample and callable(schema_sample["call"]):
                    schema_sample["call"](wm)
                elif "obs" in schema_sample and "act" in schema_sample:
                    wm(schema_sample["obs"], schema_sample["act"])
                elif "args" in schema_sample:
                    wm(*schema_sample["args"], **schema_sample.get("kwargs", {}))
                else:
                    raise ValueError("schema sample mapping needs call, obs+act, or args")
            elif isinstance(schema_sample, (tuple, list)):
                wm(*schema_sample)
            else:
                raise TypeError("unsupported schema-only dry-run sample")
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training.items():
            module.train(was_training)
        current_buffers = dict(wm.named_buffers())
        with torch.no_grad():
            for name, saved in buffers.items():
                if name in current_buffers:
                    current_buffers[name].copy_(saved)
        for (module, name), value in mutable_attrs.items():
            setattr(module, name, value)
        _restore_rng(rng)
    return reached


def _entries_from_candidates(
    candidates: Sequence[_Candidate],
    rank: int,
    reached: Optional[Set[str]],
    require_active: bool,
) -> List[ManifestEntry]:
    # The last projection leaf is the projection output layer.  Official
    # ViTPredictor has no separate final projection, so MLP outputs keep their
    # own role rather than being relabelled.
    projection_paths = [c.module_path for c in candidates if c.module_group == "encoder_projection"]
    final_projection_path = projection_paths[-1] if projection_paths else None
    predictor_paths = [c.module_path for c in candidates if c.module_group == "predictor"]
    predictor_final_path = predictor_paths[-1] if predictor_paths else None
    entries: List[ManifestEntry] = []
    for candidate in candidates:
        active = (
            candidate.module_path in reached
            if reached is not None
            else candidate.active_structural
        )
        detection = "schema_dry_run" if reached is not None else "structural"
        module = candidate.module
        if isinstance(module, nn.Linear):
            logical_id = candidate.module_path
            actual_rank = min(rank, module.in_features, module.out_features)
            final = candidate.module_path == final_projection_path or (
                candidate.module_path == predictor_final_path
                and candidate.role == "other_linear"
            )
            entries.append(
                ManifestEntry(
                    module_path=candidate.module_path,
                    logical_layer_id=logical_id,
                    layer_type="Linear",
                    in_features=module.in_features,
                    out_features=module.out_features,
                    module_group=candidate.module_group,
                    role="final_projection" if final else candidate.role,
                    active_in_forward=active,
                    default_inject=candidate.default_enabled and (active or not require_active),
                    injected=False,
                    actual_rank=actual_rank,
                    attention_output=candidate.role == "attention_output",
                    mlp_output=candidate.role == "mlp_output",
                    final_projection=final,
                    bias=module.bias is not None,
                    active_detection=detection,
                )
            )
        else:
            kh, kw = module.kernel_size
            flat_in = (module.in_channels // module.groups) * kh * kw
            out_group = module.out_channels // module.groups
            actual_rank = min(rank, flat_in, out_group)
            final = candidate.module_path == final_projection_path
            for group_index in range(module.groups):
                entries.append(
                    ManifestEntry(
                        module_path=candidate.module_path,
                        logical_layer_id=f"{candidate.module_path}::group={group_index}",
                        layer_type="Conv2d",
                        in_features=flat_in,
                        out_features=out_group,
                        module_group=candidate.module_group,
                        role="final_projection" if final else candidate.role,
                        active_in_forward=active,
                        default_inject=candidate.default_enabled and (active or not require_active),
                        injected=False,
                        actual_rank=actual_rank,
                        final_projection=final,
                        kernel_size=tuple(module.kernel_size),
                        stride=tuple(module.stride),
                        padding=module.padding if isinstance(module.padding, str) else tuple(module.padding),
                        dilation=tuple(module.dilation),
                        groups=module.groups,
                        logical_group=group_index,
                        bias=module.bias is not None,
                        active_detection=detection,
                    )
                )
    return entries


def _validate_entries(
    entries: Sequence[ManifestEntry],
    adapter: VisualLatentAdapter,
    config: Any,
) -> None:
    ids = [entry.logical_layer_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("target manifest contains duplicate logical layer IDs")
    if any(path == "encoder.base_model" or ".base_model." in f".{path}." for path in (e.module_path for e in entries)):
        raise RuntimeError("target manifest illegally includes frozen base_model")
    predictor_targets = [
        e for e in entries if e.module_group == "predictor" and e.default_inject
    ]
    if bool(_cfg_get(config, "target_modules.fail_on_empty_predictor_targets", True)) and not predictor_targets:
        raise RuntimeError("FD-PSC found no active predictor Linear targets")

    projection_candidates = [e for e in entries if e.module_group == "encoder_projection"]
    active_projection = [e for e in projection_candidates if e.default_inject]
    require_if_head = bool(
        _cfg_get(config, "target_modules.require_projection_targets_if_head_exists", True)
    )
    if adapter.has_projection_head and require_if_head and not active_projection:
        raise RuntimeError(
            "encoder has a post-backbone projection head but no active enabled Linear/Conv2d targets"
        )
    if (
        not adapter.has_projection_head
        and bool(_cfg_get(config, "target_modules.fail_on_empty_projection_targets", False))
    ):
        raise RuntimeError("encoder has no projection head (projection targets are not applicable)")

    for group, flag in (
        ("action_encoder", "target_modules.action_encoder_linear"),
        ("proprio_encoder", "target_modules.proprio_encoder_linear"),
    ):
        if bool(_cfg_get(config, flag, False)) and not any(
            e.module_group == group and e.default_inject for e in entries
        ):
            raise RuntimeError(f"{flag} is enabled but no active Linear target exists")


def enumerate_fd_psc_targets(
    wm: nn.Module,
    config: Any,
    *,
    schema_sample: Any = None,
) -> Tuple[TargetManifest, VisualLatentAdapter]:
    """Enumerate targets after base checkpoint load and before injection."""

    if _contains_injected_adapter(wm):
        raise RuntimeError("target enumeration must run before adapter injection")
    adapter = get_encoder_adapter(wm)
    candidates = _collect_candidates(wm, config, adapter)
    reached = (
        _run_schema_dry_run(wm, candidates, schema_sample)
        if schema_sample is not None
        else None
    )
    rank = int(_cfg_get(config, "episodic_lora.rank", 8))
    if rank <= 0:
        raise ValueError("episodic_lora.rank must be positive")
    entries = _entries_from_candidates(
        candidates,
        rank,
        reached,
        bool(_cfg_get(config, "target_modules.require_active_forward_path", True)),
    )
    _validate_entries(entries, adapter, config)
    manifest = TargetManifest(
        entries=tuple(sorted(entries, key=lambda item: item.logical_layer_id)),
        enabled=True,
        metadata={
            "encoder_type": adapter.encoder_type,
            "projection_head_exists": adapter.has_projection_head,
            "projection_module_paths": list(adapter.projection_module_paths),
            "zero_projection_targets": not any(
                e.module_group == "encoder_projection" and e.default_inject for e in entries
            ),
            "projection_status": (
                "applicable"
                if adapter.has_projection_head
                else "not_applicable_no_projection_head"
            ),
            "excluded_subtrees": ["encoder.base_model", "decoder"],
        },
    )
    return manifest, adapter


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root.get_submodule(".".join(parts[:-1]))
    return parent, parts[-1]


def _replace_child(parent: nn.Module, name: str, module: nn.Module) -> None:
    if name not in parent._modules:
        raise KeyError(f"{type(parent).__name__} has no child module {name!r}")
    parent._modules[name] = module


def inject_fd_psc_adapters(
    wm: nn.Module,
    config: Any,
    *,
    schema_sample: Any = None,
) -> InjectionResult:
    """Inject all enabled, active full-depth adapters and return stable registries.

    When ``fd_psc.enabled`` is false this function is a strict no-op: no module
    replacement, protocol attachment, hooks, or state_dict-key changes.
    """

    if not bool(_cfg_get(config, "enabled", False)):
        return InjectionResult(
            adapters={},
            manifest=TargetManifest(
                entries=(), enabled=False, metadata={"status": "disabled_no_injection"}
            ),
            encoder_adapter=None,
        )

    manifest, _ = enumerate_fd_psc_targets(wm, config, schema_sample=schema_sample)
    encoder_adapter = install_encoder_adapter_protocol(wm)
    # theta_0 is the entire base model, not merely selected old AdaJEPA params.
    for parameter in wm.parameters():
        parameter.requires_grad_(False)

    rank = int(_cfg_get(config, "episodic_lora.rank", 8))
    alpha = float(_cfg_get(config, "episodic_lora.alpha", 16.0))
    dropout = float(_cfg_get(config, "episodic_lora.dropout", 0.0))
    global_seed = int(_cfg_get(config, "seed", 0))
    entries_by_path: Dict[str, List[ManifestEntry]] = {}
    for entry in manifest.entries:
        entries_by_path.setdefault(entry.module_path, []).append(entry)

    adapters: Dict[str, LogicalLoRAAdapter] = {}
    physical_modules: Dict[str, nn.Module] = {}
    injected_paths: Set[str] = set()
    for path in sorted(entries_by_path):
        path_entries = entries_by_path[path]
        if not any(entry.default_inject for entry in path_entries):
            continue
        module = wm.get_submodule(path)
        parent, name = _resolve_parent(wm, path)
        if isinstance(module, nn.Linear):
            logical_id = path_entries[0].logical_layer_id
            wrapper = DualLoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                generator=_generator(module.weight.device, _stable_seed(global_seed, logical_id)),
                logical_id=logical_id,
            )
            _replace_child(parent, name, wrapper)
            adapters[logical_id] = wrapper
            physical_modules[path] = wrapper
        elif isinstance(module, nn.Conv2d):
            generators = [
                _generator(
                    module.weight.device,
                    _stable_seed(global_seed, f"{path}::group={group_index}"),
                )
                for group_index in range(module.groups)
            ]
            wrapper = DualLoRAConv2d(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                generators=generators,
                logical_id=path,
            )
            _replace_child(parent, name, wrapper)
            physical_modules[path] = wrapper
            for group in wrapper.iter_logical_groups():
                assert group.logical_id is not None
                adapters[group.logical_id] = group
        else:
            raise RuntimeError(f"target changed type before injection: {path}")
        injected_paths.add(path)

    injected_entries = tuple(
        replace(entry, injected=entry.default_inject and entry.module_path in injected_paths)
        for entry in manifest.entries
    )
    manifest = TargetManifest(
        entries=injected_entries,
        enabled=True,
        metadata=dict(manifest.metadata),
    )
    expected_ids = {entry.logical_layer_id for entry in manifest.entries if entry.injected}
    if expected_ids != set(adapters):
        raise RuntimeError(
            f"adapter registry/manifest mismatch: missing={sorted(expected_ids-set(adapters))}, "
            f"extra={sorted(set(adapters)-expected_ids)}"
        )

    # Reassert frozen backbone and base BatchNorm eval immediately before each
    # encoder forward, even if AdaJEPA calls encoder.train() for adapter updates.
    hook = wm.encoder.register_forward_pre_hook(
        lambda _module, _args: encoder_adapter.enforce_frozen_eval()
    )
    return InjectionResult(
        adapters=adapters,
        manifest=manifest,
        encoder_adapter=encoder_adapter,
        _physical_modules=physical_modules,
        _base_eval_hook=hook,
        _seed=global_seed,
    )


# Concise aliases for scripts written against early design drafts.
inject_adapters = inject_fd_psc_adapters
build_target_manifest = enumerate_fd_psc_targets


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManifestEntry",
    "TargetManifest",
    "InjectionResult",
    "enumerate_fd_psc_targets",
    "build_target_manifest",
    "inject_fd_psc_adapters",
    "inject_adapters",
]
