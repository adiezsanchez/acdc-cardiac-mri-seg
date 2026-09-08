"""Minimal numpy augmentations (no extra deps, no matplotlib)."""

from __future__ import annotations

import numpy as np


def intensity_normalize(image: np.ndarray) -> np.ndarray:
    """Percentile clip + [0, 1] scale. Works on 2D slices or 3D volumes."""
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1.0, 99.0))
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def pad_or_crop_2d(image: np.ndarray, label: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = _pad_or_crop_hw(image, size)
    label = _pad_or_crop_hw(label, size)
    return image, label


def pad_or_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    depth: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = _pad_or_crop_d(image, depth)
    label = _pad_or_crop_d(label, depth)
    out_img = np.stack([_pad_or_crop_hw(sl, size) for sl in image], axis=0)
    out_lab = np.stack([_pad_or_crop_hw(sl, size) for sl in label], axis=0)
    return out_img, out_lab


def _pad_or_crop_hw(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    out = arr
    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        if out.ndim == 2:
            out = np.pad(out, pad)
        else:
            raise ValueError("Expected 2D array")
        h, w = out.shape
    start_h = max((h - size) // 2, 0)
    start_w = max((w - size) // 2, 0)
    return out[start_h : start_h + size, start_w : start_w + size]


def _pad_or_crop_d(arr: np.ndarray, depth: int) -> np.ndarray:
    d = arr.shape[0]
    if d < depth:
        pad = (depth - d)
        before = pad // 2
        after = pad - before
        return np.pad(arr, ((before, after), (0, 0), (0, 0)))
    start = max((d - depth) // 2, 0)
    return arr[start : start + depth]


def geometric_params(rng: np.random.Generator) -> dict[str, object]:
    return {
        "flip_w": rng.random() < 0.5,
        "flip_h": rng.random() < 0.5,
        "k90": int(rng.integers(0, 4)),
        "gain": float(rng.uniform(0.85, 1.15)),
        "bias": float(rng.uniform(-0.05, 0.05)),
    }


def apply_geometry(image: np.ndarray, params: dict[str, object], *, is_label: bool) -> np.ndarray:
    out = image
    if params["flip_w"]:
        out = out[..., :, ::-1]
    if params["flip_h"]:
        out = out[..., ::-1, :]
    k = int(params["k90"])
    if k:
        axes = (out.ndim - 2, out.ndim - 1)
        out = np.rot90(out, k, axes=axes)
    if not is_label:
        out = np.clip(out * float(params["gain"]) + float(params["bias"]), 0.0, 1.0)
    return np.ascontiguousarray(out)


def augment_2d(image: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random flips and a small intensity jitter. Geometry stays on the voxel grid."""
    params = geometric_params(rng)
    return apply_geometry(image, params, is_label=False), apply_geometry(label, params, is_label=True)


def augment_stack(stack: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Same geometric draw for every neighbouring-slice channel."""
    params = geometric_params(rng)
    out = np.stack([apply_geometry(stack[c], params, is_label=False) for c in range(stack.shape[0])], axis=0)
    return out, apply_geometry(label, params, is_label=True)
