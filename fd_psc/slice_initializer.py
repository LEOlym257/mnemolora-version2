"""Triggered Delayed SLICE initializers and first-step magnitude matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .gradient_geometry import gradient_cosine


@dataclass(frozen=True)
class RandomizedSVDResult:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    used_full_svd_fallback: bool = False


def _dedicated_generator(matrix: torch.Tensor, seed: int = 0) -> torch.Generator:
    device = matrix.device if matrix.is_cuda else torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    oversampling: int = 2,
    power_iterations: int = 1,
    generator: Optional[torch.Generator] = None,
) -> RandomizedSVDResult:
    """Deterministic-generator randomized SVD with validated full-SVD fallback."""

    if matrix.ndim != 2:
        raise ValueError("SLICE gradient must be a matrix")
    if rank <= 0:
        raise ValueError("requested SVD rank must be positive")
    if oversampling < 0 or power_iterations < 0:
        raise ValueError("randomized SVD controls must be non-negative")
    value = matrix.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("SLICE gradient contains non-finite values")
    minimum_dimension = min(value.shape)
    target = min(int(rank), minimum_dimension)
    if target == minimum_dimension:
        u, singular, vh = torch.linalg.svd(value, full_matrices=False)
        return RandomizedSVDResult(u[:, :target], singular[:target], vh[:target], True)
    sample_rank = min(minimum_dimension, target + int(oversampling))
    generator = generator or _dedicated_generator(value)
    try:
        omega = torch.randn(
            value.shape[1],
            sample_rank,
            device=value.device,
            dtype=value.dtype,
            generator=generator,
        )
        y = value @ omega
        for _ in range(int(power_iterations)):
            y = value @ (value.transpose(0, 1) @ y)
            y, _ = torch.linalg.qr(y, mode="reduced")
        q, _ = torch.linalg.qr(y, mode="reduced")
        small = q.transpose(0, 1) @ value
        u_small, singular, vh = torch.linalg.svd(small, full_matrices=False)
        u = q @ u_small
        u, singular, vh = u[:, :target], singular[:target], vh[:target]
        orth_u = u.transpose(0, 1) @ u
        orth_v = vh @ vh.transpose(0, 1)
        eye = torch.eye(target, device=value.device, dtype=value.dtype)
        approximation = (u * singular.unsqueeze(0)) @ vh
        residual = torch.linalg.vector_norm(value - approximation)
        input_norm = torch.linalg.vector_norm(value)
        valid = (
            torch.isfinite(u).all()
            and torch.isfinite(singular).all()
            and torch.isfinite(vh).all()
            and torch.isfinite(residual)
            and torch.allclose(orth_u, eye, atol=2.0e-4, rtol=2.0e-4)
            and torch.allclose(orth_v, eye, atol=2.0e-4, rtol=2.0e-4)
            # An orthonormal but unrelated basis can still look superficially
            # valid.  A truncated SVD reconstruction must not have more energy
            # in its residual than the zero approximation.
            and residual <= input_norm * (1.0 + 2.0e-4) + 1.0e-7
        )
        if not valid:
            raise RuntimeError("randomized SVD validation failed")
        return RandomizedSVDResult(u, singular, vh)
    except (RuntimeError, torch.linalg.LinAlgError):
        try:
            u, singular, vh = torch.linalg.svd(value, full_matrices=False)
        except (RuntimeError, torch.linalg.LinAlgError):
            u, singular, vh = torch.linalg.svd(value.to(dtype=torch.float64), full_matrices=False)
        if not (torch.isfinite(u).all() and torch.isfinite(singular).all() and torch.isfinite(vh).all()):
            raise RuntimeError("both randomized and full SLICE SVD failed")
        return RandomizedSVDResult(u[:, :target], singular[:target], vh[:target], True)


def _available_singular_directions(
    singular_values: torch.Tensor,
    rows: int,
    columns: int,
) -> int:
    if singular_values.numel() == 0:
        return 0
    largest = float(singular_values[0].detach().cpu())
    if largest <= 0.0 or not math.isfinite(largest):
        return 0
    tolerance = max(rows, columns) * torch.finfo(singular_values.dtype).eps * largest
    return int(torch.count_nonzero(torch.isfinite(singular_values) & (singular_values > tolerance)).item())


@dataclass(frozen=True)
class FirstStepResult:
    delta_weight: torch.Tensor
    norm: float
    descent_cosine: Optional[float]
    finite: bool


def simulate_factor_first_step(
    gradient: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    scaling: float = 1.0,
    optimizer_name: str = "adam",
    learning_rate: float = 5.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    optimizer_epsilon: float = 1.0e-8,
    weight_decay: Optional[float] = None,
) -> FirstStepResult:
    """Run one real zero-state optimizer step on cloned factors."""

    if gradient.shape != (b.shape[0], a.shape[1]) or b.shape[1] != a.shape[0]:
        raise ValueError("gradient and factor dimensions are inconsistent")
    b_var = b.detach().clone().requires_grad_(True)
    a_var = a.detach().clone().requires_grad_(True)
    b_initial = b_var.detach().clone()
    a_initial = a_var.detach().clone()
    name = str(optimizer_name).lower()
    kwargs = {"lr": float(learning_rate)}
    if name in ("adam", "adamw"):
        kwargs.update({"betas": tuple(float(x) for x in betas), "eps": float(optimizer_epsilon)})
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    if name not in optimizers:
        raise ValueError(f"unsupported online optimizer for first-step matching: {optimizer_name}")
    optimizer = optimizers[name]([b_var, a_var], **kwargs)
    g = gradient.detach().to(device=b_var.device, dtype=torch.float32)
    proxy = float(scaling) * torch.sum(
        b_var.to(dtype=torch.float32) * (g @ a_var.to(dtype=torch.float32).transpose(0, 1))
    )
    optimizer.zero_grad()
    proxy.backward()
    optimizer.step()
    delta = float(scaling) * (
        b_var.detach().to(dtype=torch.float32) @ a_var.detach().to(dtype=torch.float32)
        - b_initial.to(dtype=torch.float32) @ a_initial.to(dtype=torch.float32)
    )
    norm_t = torch.linalg.vector_norm(delta)
    cosine = gradient_cosine(delta, -g)
    finite = bool(torch.isfinite(delta).all() and torch.isfinite(norm_t))
    return FirstStepResult(
        delta,
        float(norm_t.detach().cpu()) if finite else float("inf"),
        cosine.value,
        finite,
    )


@dataclass(frozen=True)
class MagnitudeMatchResult:
    beta: float
    first_step_norm: float
    target_norm: float
    relative_error: float
    descent_cosine: Optional[float]
    matched: bool
    reason: str = ""


def match_first_step_magnitude(
    gradient: torch.Tensor,
    unscaled_b: torch.Tensor,
    unscaled_a: torch.Tensor,
    target_norm: float,
    *,
    scaling: float = 1.0,
    maximum_beta: float = 10.0,
    optimizer_name: str = "adam",
    learning_rate: float = 5.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    optimizer_epsilon: float = 1.0e-8,
    weight_decay: Optional[float] = None,
    relative_tolerance: float = 0.05,
) -> MagnitudeMatchResult:
    """Search a bounded non-negative beta using actual optimizer semantics."""

    if not math.isfinite(float(target_norm)) or target_norm <= 0.0:
        return MagnitudeMatchResult(0.0, 0.0, float(target_norm), float("inf"), None, False, "invalid_baseline_first_step_norm")
    if not math.isfinite(float(maximum_beta)) or maximum_beta <= 0.0:
        raise ValueError("maximum_beta must be finite and positive")

    cache = {}

    def evaluate(beta: float) -> Tuple[float, FirstStepResult]:
        beta = min(float(maximum_beta), max(0.0, float(beta)))
        key = float(beta)
        if key not in cache:
            root = math.sqrt(beta)
            cache[key] = simulate_factor_first_step(
                gradient,
                root * unscaled_b,
                root * unscaled_a,
                scaling=scaling,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                betas=betas,
                optimizer_epsilon=optimizer_epsilon,
                weight_decay=weight_decay,
            )
        result = cache[key]
        error = abs(result.norm - float(target_norm)) / max(float(target_norm), 1.0e-12)
        return error, result

    # A compact log grid handles both SGD-like (roughly beta) and Adam-like
    # (often roughly sqrt(beta)) scale laws without assuming either one.
    candidates = [0.0, float(maximum_beta)]
    lower = max(float(maximum_beta) * 1.0e-6, 1.0e-12)
    for index in range(17):
        fraction = index / 16.0
        candidates.append(math.exp(math.log(lower) * (1.0 - fraction) + math.log(float(maximum_beta)) * fraction))
    candidates = sorted(set(candidates))
    evaluated = [(beta, *evaluate(beta)) for beta in candidates]
    best_index = min(range(len(evaluated)), key=lambda index: evaluated[index][1])
    lo = evaluated[max(0, best_index - 1)][0]
    hi = evaluated[min(len(evaluated) - 1, best_index + 1)][0]
    # Refine the local bracket without assuming differentiability through Adam.
    for _ in range(10):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        left_error, _ = evaluate(left)
        right_error, _ = evaluate(right)
        if left_error <= right_error:
            hi = right
        else:
            lo = left
    refined = [(lo, *evaluate(lo)), (hi, *evaluate(hi)), ((lo + hi) / 2.0, *evaluate((lo + hi) / 2.0))]
    beta, error, result = min([*evaluated, *refined], key=lambda item: item[1])
    descending = result.descent_cosine is not None and result.descent_cosine > 0.0
    matched = result.finite and descending and error <= float(relative_tolerance)
    reason = "" if matched else (
        "first_step_not_descent" if not descending else "first_step_magnitude_mismatch"
    )
    return MagnitudeMatchResult(
        float(beta),
        result.norm,
        float(target_norm),
        float(error),
        result.descent_cosine,
        matched,
        reason,
    )


@dataclass(frozen=True)
class SliceInitialization:
    mode: str
    b0: Optional[torch.Tensor]
    a0: Optional[torch.Tensor]
    actual_rank: int
    beta: float
    success: bool
    fallback_to_pilot: bool
    reason: str
    singular_values: torch.Tensor
    magnitude_match: Optional[MagnitudeMatchResult] = None
    used_full_svd_fallback: bool = False


def _symmetric_factors(
    decomposition: RandomizedSVDResult,
    requested_rank: int,
    available: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    rank = min(int(requested_rank), int(available), int(decomposition.singular_values.numel()))
    if rank <= 0:
        return None, None, 0
    roots = torch.sqrt(torch.clamp(decomposition.singular_values[:rank], min=0.0))
    return (
        decomposition.u[:, :rank] * roots.unsqueeze(0),
        roots.unsqueeze(1) * decomposition.vh[:rank],
        rank,
    )


def initialize_slice(
    corrected_gradient: torch.Tensor,
    requested_rank: int = 8,
    *,
    mode: str = "slice_exact",
    fallback_mode: str = "slice_symmetric",
    oversampling: int = 2,
    power_iterations: int = 1,
    generator: Optional[torch.Generator] = None,
    alpha: float = 16.0,
    baseline_first_step_norm: Optional[float] = None,
    magnitude_mode: str = "first_step_match",
    maximum_beta: float = 10.0,
    optimizer_name: str = "adam",
    learning_rate: float = 5.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    optimizer_epsilon: float = 1.0e-8,
    weight_decay: Optional[float] = None,
) -> SliceInitialization:
    """Create exact/symmetric non-zero centered factors or retain Pilot safely."""

    if corrected_gradient.ndim != 2:
        raise ValueError("SLICE corrected gradient must be a matrix")
    if requested_rank <= 0:
        raise ValueError("requested_rank must be positive")
    if mode not in ("slice_exact", "slice_symmetric"):
        raise ValueError(f"unknown SLICE initialization: {mode}")
    if fallback_mode != "slice_symmetric":
        raise ValueError("the compliant SLICE fallback is slice_symmetric")
    rows, columns = corrected_gradient.shape
    minimum_dimension = min(rows, columns)
    required_directions = min(minimum_dimension, max(requested_rank, 2 * requested_rank if mode == "slice_exact" else requested_rank))
    try:
        decomposition = randomized_svd(
            corrected_gradient,
            max(1, required_directions),
            oversampling,
            power_iterations,
            generator,
        )
    except (RuntimeError, ValueError) as error:
        return SliceInitialization(
            "pilot_fallback", None, None, 0, 0.0, False, True,
            f"svd_failure:{error}", torch.empty(0, device=corrected_gradient.device)
        )
    available = _available_singular_directions(decomposition.singular_values, rows, columns)
    chosen_mode = mode
    b_hat: Optional[torch.Tensor]
    a_hat: Optional[torch.Tensor]
    actual_rank: int
    reason = ""
    if mode == "slice_exact" and minimum_dimension >= 2 * requested_rank:
        actual_rank = min(int(requested_rank), available // 2)
        if actual_rank > 0:
            b_hat = decomposition.u[:, :actual_rank]
            a_hat = decomposition.vh[actual_rank : 2 * actual_rank]
        else:
            b_hat, a_hat, actual_rank = _symmetric_factors(decomposition, requested_rank, available)
            chosen_mode = "slice_symmetric"
            reason = "slice_exact_no_paired_directions"
    elif mode == "slice_exact":
        b_hat, a_hat, actual_rank = _symmetric_factors(decomposition, requested_rank, available)
        chosen_mode = "slice_symmetric"
        reason = "slice_exact_dimension_fallback"
    else:
        b_hat, a_hat, actual_rank = _symmetric_factors(decomposition, requested_rank, available)
    if actual_rank <= 0 or b_hat is None or a_hat is None:
        return SliceInitialization(
            "pilot_fallback",
            None,
            None,
            0,
            0.0,
            False,
            True,
            "zero_or_rank_deficient_gradient",
            decomposition.singular_values,
            used_full_svd_fallback=decomposition.used_full_svd_fallback,
        )

    beta = 1.0
    match = None
    scaling = float(alpha) / float(actual_rank)
    if magnitude_mode == "first_step_match" and baseline_first_step_norm is not None:
        match = match_first_step_magnitude(
            corrected_gradient,
            b_hat,
            a_hat,
            float(baseline_first_step_norm),
            scaling=scaling,
            maximum_beta=maximum_beta,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            betas=betas,
            optimizer_epsilon=optimizer_epsilon,
            weight_decay=weight_decay,
        )
        if not match.matched:
            return SliceInitialization(
                "pilot_fallback",
                None,
                None,
                0,
                0.0,
                False,
                True,
                match.reason,
                decomposition.singular_values,
                match,
                decomposition.used_full_svd_fallback,
            )
        beta = match.beta
    elif magnitude_mode not in ("first_step_match", "none"):
        raise ValueError(f"unknown SLICE magnitude mode: {magnitude_mode}")
    root = math.sqrt(max(0.0, beta))
    b0 = root * b_hat
    a0 = root * a_hat
    if not (torch.isfinite(b0).all() and torch.isfinite(a0).all()):
        return SliceInitialization(
            "pilot_fallback", None, None, 0, 0.0, False, True,
            "non_finite_slice_factors", decomposition.singular_values,
            match, decomposition.used_full_svd_fallback
        )
    return SliceInitialization(
        chosen_mode,
        b0,
        a0,
        actual_rank,
        beta,
        True,
        False,
        reason,
        decomposition.singular_values,
        match,
        decomposition.used_full_svd_fallback,
    )


def slice_exact(corrected_gradient: torch.Tensor, rank: int, **kwargs) -> SliceInitialization:
    return initialize_slice(corrected_gradient, rank, mode="slice_exact", **kwargs)


def slice_symmetric(corrected_gradient: torch.Tensor, rank: int, **kwargs) -> SliceInitialization:
    return initialize_slice(corrected_gradient, rank, mode="slice_symmetric", **kwargs)


first_step_match = match_first_step_magnitude
SliceInitializer = initialize_slice


__all__ = [
    "FirstStepResult",
    "MagnitudeMatchResult",
    "RandomizedSVDResult",
    "SliceInitialization",
    "SliceInitializer",
    "first_step_match",
    "initialize_slice",
    "match_first_step_magnitude",
    "randomized_svd",
    "simulate_factor_first_step",
    "slice_exact",
    "slice_symmetric",
]
