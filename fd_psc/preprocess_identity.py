"""Deterministic identity for the online observation/action preprocessing path."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn


PREPROCESS_IDENTITY_SCHEMA = "adajepa_preprocess_v1"


class PreprocessIdentityError(ValueError):
    """Raised when a runtime transform cannot be identified deterministically."""


def _tensor_identity(value: Any) -> Mapping[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise PreprocessIdentityError("preprocess statistics must be finite")
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _callable_code_identity(value: Any) -> Mapping[str, Any]:
    result = {
        "module": str(getattr(value, "__module__", "")),
        "qualname": str(
            getattr(value, "__qualname__", getattr(value, "__name__", ""))
        ),
    }
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        code = getattr(value, "__code__", None)
        if code is None:
            raise PreprocessIdentityError(
                f"callable {result['module']}.{result['qualname']} has no stable source identity"
            )
        source = json.dumps(
            {
                "bytecode": code.co_code.hex(),
                "constants": [repr(item) for item in code.co_consts],
                "names": list(code.co_names),
            },
            sort_keys=True,
        )
    result["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return result


def _implementation_identity(value: Any, *, path: str) -> Mapping[str, str]:
    target = type(value)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        root_name = str(target.__module__).split(".", 1)[0]
        try:
            root_module = importlib.import_module(root_name)
        except (ImportError, ValueError) as exc:
            raise PreprocessIdentityError(
                f"{path} implementation cannot be versioned deterministically"
            ) from exc
        version = getattr(root_module, "__version__", None)
        if version is None:
            raise PreprocessIdentityError(
                f"{path} implementation has neither inspectable source nor package version"
            )
        return {"package": root_name, "version": str(version)}
    return {"source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}


def _component_identity(value: Any, *, path: str, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreprocessIdentityError(f"{path} must be finite")
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Enum):
        return {
            "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Tensor):
        return {"tensor": _tensor_identity(value)}
    if type(value).__module__.startswith("numpy"):
        try:
            return {"tensor": _tensor_identity(value)}
        except (TypeError, RuntimeError) as exc:
            raise PreprocessIdentityError(f"unsupported numpy value at {path}") from exc
    if isinstance(value, Mapping):
        return {
            str(key): _component_identity(
                value[key], path=f"{path}.{key}", seen=seen
            )
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [
            _component_identity(item, path=f"{path}[{index}]", seen=seen)
            for index, item in enumerate(value)
        ]

    identity = id(value)
    if identity in seen:
        raise PreprocessIdentityError(f"cyclic transform configuration at {path}")
    seen.add(identity)
    try:
        kind = f"{type(value).__module__}.{type(value).__qualname__}"
        result: dict[str, Any] = {
            "type": kind,
            "implementation": _implementation_identity(value, path=path),
        }
        if isinstance(value, nn.Module):
            result["state_dict"] = {
                name: _tensor_identity(tensor)
                for name, tensor in sorted(value.state_dict().items())
            }
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            public = {
                str(key): child
                for key, child in attributes.items()
                if not str(key).startswith("_")
            }
            if public:
                result["attributes"] = _component_identity(
                    public,
                    path=f"{path}.attributes",
                    seen=seen,
                )
        if callable(value) and "attributes" not in result and not isinstance(value, nn.Module):
            result["callable"] = _callable_code_identity(value)
        rendered = repr(value)
        if re.search(r"0x[0-9a-fA-F]+", rendered) and len(result) == 2:
            raise PreprocessIdentityError(
                f"{path} has only an address-dependent representation; provide a versioned transform"
            )
        return result
    finally:
        seen.remove(identity)


def preprocess_identity_payload(
    preprocessor: Any,
    *,
    encoder_transform: Optional[Any] = None,
    frameskip: int,
    num_hist: int,
    num_pred: int,
) -> Mapping[str, Any]:
    """Return the complete versioned preprocessing description to be hashed."""

    required = (
        "action_mean",
        "action_std",
        "state_mean",
        "state_std",
        "proprio_mean",
        "proprio_std",
        "transform",
    )
    missing = [name for name in required if not hasattr(preprocessor, name)]
    if missing:
        raise PreprocessIdentityError(
            f"runtime preprocessor is missing identity fields: {missing}"
        )
    if int(frameskip) <= 0 or int(num_hist) <= 0 or int(num_pred) <= 0:
        raise PreprocessIdentityError("frameskip/num_hist/num_pred must be positive")
    return {
        "schema": PREPROCESS_IDENTITY_SCHEMA,
        "visual_input": "BTHWC_uint8_to_BTCHW_float_div255",
        "frameskip": int(frameskip),
        "num_hist": int(num_hist),
        "num_pred": int(num_pred),
        "statistics": {
            name: _tensor_identity(getattr(preprocessor, name))
            for name in required
            if name != "transform"
        },
        "observation_transform": _component_identity(
            preprocessor.transform,
            path="observation_transform",
            seen=set(),
        ),
        "encoder_transform": _component_identity(
            encoder_transform,
            path="encoder_transform",
            seen=set(),
        ),
    }


def compute_preprocess_hash(
    preprocessor: Any,
    *,
    encoder_transform: Optional[Any] = None,
    frameskip: int,
    num_hist: int,
    num_pred: int,
) -> str:
    payload = preprocess_identity_payload(
        preprocessor,
        encoder_transform=encoder_transform,
        frameskip=frameskip,
        num_hist=num_hist,
        num_pred=num_pred,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PREPROCESS_IDENTITY_SCHEMA",
    "PreprocessIdentityError",
    "compute_preprocess_hash",
    "preprocess_identity_payload",
]
