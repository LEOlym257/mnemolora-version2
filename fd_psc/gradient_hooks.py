"""Exact effective-weight gradient collection for Linear and Conv2d targets.

The base weights stay frozen in FD-PSC, so ``weight.grad`` is intentionally
absent.  For a module invocation we instead retain its detached input and use
the exact autograd ``grad_output`` to reconstruct ``dL/dW``.  An output-tensor
backward hook closes over a unique invocation id, which remains correct when a
module is called repeatedly or re-entered in a single graph.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_effective_weight_gradient(input_tensor: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Return the exact autograd-reduced Linear weight gradient in float32."""

    if input_tensor.shape[:-1] != grad_output.shape[:-1]:
        raise ValueError(
            "Linear input/grad_output leading dimensions differ: "
            f"{tuple(input_tensor.shape)} vs {tuple(grad_output.shape)}"
        )
    x = input_tensor.detach().reshape(-1, input_tensor.shape[-1]).to(dtype=torch.float32)
    d = grad_output.detach().reshape(-1, grad_output.shape[-1]).to(device=x.device, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(d).all():
        raise ValueError("non-finite Linear input or output gradient")
    # grad_output already contains the loss reduction.  Do not divide by N.
    return d.transpose(0, 1) @ x


def _conv_geometry(module: nn.Conv2d) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]:
    stride = tuple(int(x) for x in module.stride)
    dilation = tuple(int(x) for x in module.dilation)
    kernel = tuple(int(x) for x in module.kernel_size)
    return stride, dilation, kernel, int(module.groups)


def _prepare_conv_input(module: nn.Conv2d, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Mirror ``nn.Conv2d._conv_forward`` padding semantics exactly."""

    padding = module.padding
    if module.padding_mode != "zeros":
        padded = F.pad(input_tensor, module._reversed_padding_repeated_twice, mode=module.padding_mode)
        return padded, (0, 0)
    if isinstance(padding, str):
        if padding not in ("same", "valid"):
            raise ValueError(f"unsupported Conv2d padding string: {padding}")
        # PyTorch stores the possibly asymmetric SAME pad in this order for F.pad.
        padded = F.pad(input_tensor, module._reversed_padding_repeated_twice, mode="constant", value=0.0)
        return padded, (0, 0)
    return input_tensor, tuple(int(x) for x in padding)


def conv2d_effective_weight_gradient(
    module: nn.Conv2d,
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Return an exact Conv2d weight gradient, preserving all Conv2d geometry."""

    if input_tensor.ndim != 4 or grad_output.ndim != 4:
        raise ValueError("Conv2d hooks require NCHW input and grad_output")
    stride, dilation, _, groups = _conv_geometry(module)
    x = input_tensor.detach().to(dtype=torch.float32)
    d = grad_output.detach().to(device=x.device, dtype=torch.float32)
    x, internal_padding = _prepare_conv_input(module, x)
    if not torch.isfinite(x).all() or not torch.isfinite(d).all():
        raise ValueError("non-finite Conv2d input or output gradient")
    return torch.nn.grad.conv2d_weight(
        x,
        tuple(module.weight.shape),
        d,
        stride=stride,
        padding=internal_padding,
        dilation=dilation,
        groups=groups,
    )


def conv_weight_to_logical_matrices(module: nn.Conv2d, gradient: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Split and flatten a Conv2d weight gradient into independent groups."""

    if tuple(gradient.shape) != tuple(module.weight.shape):
        raise ValueError("Conv2d gradient shape does not match its base weight")
    out_per_group = module.out_channels // module.groups
    return tuple(
        gradient[g * out_per_group : (g + 1) * out_per_group].reshape(out_per_group, -1)
        for g in range(module.groups)
    )


def _resolve_base_module(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        return module
    for name in ("base_layer", "base_module", "base", "linear", "conv"):
        candidate = getattr(module, name, None)
        if isinstance(candidate, (nn.Linear, nn.Conv2d)):
            return candidate
    raise TypeError(
        f"{type(module).__name__} is not Linear/Conv2d and exposes no supported frozen base-layer attribute"
    )


@dataclass(frozen=True)
class HookStatistics:
    forward_invocations: int
    backward_invocations: int
    pending_invocations: int


class EffectiveWeightGradientCollector:
    """Collect exact effective-weight gradients for one logical module.

    Attach this collector to the adapter wrapper, not only to its frozen base
    submodule.  The wrapper's output gradient is the derivative of the actual
    effective function ``W0 + slow + exception + episodic``.
    """

    def __init__(self, module: nn.Module, logical_layer_id: str) -> None:
        self.module = module
        self.base_module = _resolve_base_module(module)
        self.logical_layer_id = str(logical_layer_id)
        self._counter = itertools.count()
        self._pending: MutableMapping[int, torch.Tensor] = OrderedDict()
        self._matrix_gradients: Dict[str, torch.Tensor] = {}
        self._weight_gradient: Optional[torch.Tensor] = None
        self._lock = threading.RLock()
        self._forward_count = 0
        self._backward_count = 0
        self._handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: nn.Module, inputs: Tuple[object, ...], output: object) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("effective-weight hook expected the first module input to be a Tensor")
        if not isinstance(output, torch.Tensor):
            raise TypeError("effective-weight hook requires a Tensor module output")
        invocation_id = next(self._counter)
        with self._lock:
            self._pending[invocation_id] = inputs[0].detach()
            self._forward_count += 1
        if not output.requires_grad:
            # There can be no backward callback for this invocation.
            with self._lock:
                self._pending.pop(invocation_id, None)
            return

        def capture(grad_output: torch.Tensor, call_id: int = invocation_id) -> torch.Tensor:
            with self._lock:
                input_tensor = self._pending.pop(call_id, None)
            if input_tensor is None:
                raise RuntimeError(f"missing cached input for hook invocation {call_id}")
            if isinstance(self.base_module, nn.Linear):
                full = linear_effective_weight_gradient(input_tensor, grad_output)
                matrices = {self.logical_layer_id: full}
            else:
                full = conv2d_effective_weight_gradient(self.base_module, input_tensor, grad_output)
                split = conv_weight_to_logical_matrices(self.base_module, full)
                matrices = {
                    (
                        self.logical_layer_id
                        if self.base_module.groups == 1
                        else f"{self.logical_layer_id}::group={group_index}"
                    ): matrix
                    for group_index, matrix in enumerate(split)
                }
            with self._lock:
                self._weight_gradient = full if self._weight_gradient is None else self._weight_gradient + full
                for layer_id, matrix in matrices.items():
                    previous = self._matrix_gradients.get(layer_id)
                    self._matrix_gradients[layer_id] = matrix if previous is None else previous + matrix
                self._backward_count += 1
            return grad_output

        output.register_hook(capture)

    @property
    def matrix_gradients(self) -> Dict[str, torch.Tensor]:
        with self._lock:
            return {key: value.detach().clone() for key, value in self._matrix_gradients.items()}

    @property
    def weight_gradient(self) -> Optional[torch.Tensor]:
        with self._lock:
            return None if self._weight_gradient is None else self._weight_gradient.detach().clone()

    @property
    def statistics(self) -> HookStatistics:
        with self._lock:
            return HookStatistics(self._forward_count, self._backward_count, len(self._pending))

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._matrix_gradients.clear()
            self._weight_gradient = None
            self._forward_count = 0
            self._backward_count = 0

    def pop_matrix_gradients(self) -> Dict[str, torch.Tensor]:
        with self._lock:
            result = self._matrix_gradients
            self._matrix_gradients = {}
            self._weight_gradient = None
            return result

    def close(self) -> None:
        self._handle.remove()
        self.reset()

    def __enter__(self) -> "EffectiveWeightGradientCollector":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


class EffectiveWeightGradientHooks:
    """A deterministic collection of per-module gradient collectors."""

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        self.collectors = OrderedDict(
            (layer_id, EffectiveWeightGradientCollector(module, layer_id))
            for layer_id, module in sorted(modules.items(), key=lambda item: item[0])
        )

    @property
    def gradients(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for collector in self.collectors.values():
            for layer_id, gradient in collector.matrix_gradients.items():
                if layer_id in result:
                    raise RuntimeError(f"duplicate logical layer gradient: {layer_id}")
                result[layer_id] = gradient
        return result

    def reset(self) -> None:
        for collector in self.collectors.values():
            collector.reset()

    def close(self) -> None:
        for collector in self.collectors.values():
            collector.close()

    def __enter__(self) -> "EffectiveWeightGradientHooks":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


WeightGradientHookManager = EffectiveWeightGradientHooks


__all__ = [
    "EffectiveWeightGradientCollector",
    "EffectiveWeightGradientHooks",
    "HookStatistics",
    "WeightGradientHookManager",
    "conv2d_effective_weight_gradient",
    "conv_weight_to_logical_matrices",
    "linear_effective_weight_gradient",
]
