"""Volumetric Dice — the metric ACDC actually ranks methods on.

Slice-wise Dice averaged over slices is *not* the same quantity: tiny apical
slices would count as much as basal slices, and empty slices (Dice undefined
or 0/1 depending on convention) distort the mean.

Volumetric Dice for class ``c`` on a 3D volume:

    2 |P_c ∩ G_c| / (|P_c| + |G_c|)

``spacing_aware=True`` weights each voxel by ``sx*sy*sz`` so the score is a
physical overlap of millilitres, not of voxels. With uniform spacing the two
coincide.
"""

from __future__ import annotations

import numpy as np

from acdc_seg.constants import FOREGROUND_LABELS, FOREGROUND_NAMES, NUM_CLASSES


def dice_coefficient(pred: np.ndarray, target: np.ndarray, *, smooth: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0  # both empty
    return float((2.0 * intersection + smooth) / (denom + smooth))


def volumetric_dice(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    spacing_aware: bool = False,
) -> dict[str, float]:
    """Per-class and mean volumetric Dice on a ``(D, H, W)`` label volume."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    weights = None
    if spacing_aware:
        if spacing_zyx is None:
            raise ValueError("spacing_zyx is required when spacing_aware=True")
        sz, sy, sx = spacing_zyx
        weights = float(sx * sy * sz)

    scores: dict[str, float] = {}
    fg = []
    for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
        p = pred == label
        g = target == label
        if weights is None:
            scores[name] = dice_coefficient(p, g)
        else:
            inter = float(np.logical_and(p, g).sum()) * weights
            denom = float(p.sum()) * weights + float(g.sum()) * weights
            if denom == 0:
                scores[name] = 1.0
            else:
                scores[name] = float((2.0 * inter) / denom)
        fg.append(scores[name])
    scores["mean"] = float(np.mean(fg))
    return scores


def confusion_counts(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    pred = pred.reshape(-1).astype(np.int64)
    target = target.reshape(-1).astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred, strict=False):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm
