"""Online image corruption for OOD evaluation.

Supports: blur, snp (salt-and-pepper), dark (brightness reduction).
Applied to uint8 HWC numpy arrays, compatible with observation dicts
from env.rollout().
"""

import numpy as np
import cv2


def apply_salt_pepper(frame, density, rng):
    """Salt-and-pepper noise on a single HxWxC uint8 frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    n = int(density * h * w)
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 255
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 0
    return out


def apply_blur(frame, sigma):
    """Gaussian blur with given sigma."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_dark(frame, factor):
    """Darken by multiplying pixel values by factor < 1."""
    out = frame.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# Registry: name -> (function, default_level)
CORRUPTIONS = {
    "blur":  (apply_blur, 2.0),
    "snp1":  (apply_salt_pepper, 0.01),
    "snp5":  (apply_salt_pepper, 0.05),
    "dark":  (apply_dark, 0.5),
}


def corrupt_frames(frames, corruption_name, level=None, seed=0):
    """Apply corruption to a batch of frames.

    Args:
        frames: np.ndarray, shape (..., H, W, C) uint8.
            Can be (H,W,C), (T,H,W,C), (B,T,H,W,C), etc.
        corruption_name: one of "blur", "snp1", "snp5", "dark", or None/"none".
        level: override default level. If None, use default.
        seed: RNG seed for stochastic corruptions (snp).

    Returns:
        Corrupted frames, same shape and dtype.
    """
    if corruption_name is None or corruption_name == "none":
        return frames

    if corruption_name not in CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption_name}'. "
            f"Choose from: {list(CORRUPTIONS.keys())}"
        )

    func, default_level = CORRUPTIONS[corruption_name]
    if level is None:
        level = default_level

    # seed=None -> truly random (system entropy), fresh noise each call.
    # seed=int -> reproducible noise for that seed.
    rng = np.random.RandomState(seed)
    orig_shape = frames.shape
    flat = frames.reshape(-1, *frames.shape[-3:])  # (N, H, W, C)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        if corruption_name.startswith("snp"):
            out[i] = func(flat[i], level, rng)
        else:
            out[i] = func(flat[i], level)
    return out.reshape(orig_shape)


def corrupt_obs_dict(obs, corruption_name, level=None, seed=0):
    """Apply corruption to the 'visual' key of an observation dict.

    Only modifies 'visual'; other keys (proprio, etc.) are unchanged.

    Args:
        obs: dict with "visual" key containing uint8 numpy array.
        corruption_name: corruption type or None/"none".
        level: optional level override.
        seed: RNG seed.

    Returns:
        New dict with corrupted visual, other keys shared.
    """
    if corruption_name is None or corruption_name == "none":
        return obs
    out = dict(obs)
    out["visual"] = corrupt_frames(obs["visual"], corruption_name, level, seed)
    return out
