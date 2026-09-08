from __future__ import annotations

import numpy as np

from acdc_seg.metrics import dice_coefficient, volumetric_dice


def test_dice_perfect_overlap():
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 3
    scores = volumetric_dice(mask, mask)
    assert scores["LV"] == 1.0
    assert scores["mean"] == 1.0


def test_dice_empty_class_is_one():
    pred = np.zeros((2, 4, 4), dtype=np.uint8)
    gt = np.zeros_like(pred)
    assert dice_coefficient(pred == 1, gt == 1) == 1.0


def test_dice_no_overlap():
    pred = np.zeros((2, 4, 4), dtype=np.uint8)
    gt = np.zeros_like(pred)
    pred[:, :, :2] = 1
    gt[:, :, 2:] = 1
    assert volumetric_dice(pred, gt)["RV"] < 1e-6


def test_spacing_aware_matches_voxel_when_uniform():
    pred = np.zeros((3, 6, 6), dtype=np.uint8)
    gt = pred.copy()
    pred[1, 2:4, 2:4] = 3
    gt[1, 2:4, 2:5] = 3
    voxel = volumetric_dice(pred, gt)
    phys = volumetric_dice(pred, gt, spacing_zyx=(1.0, 1.0, 1.0), spacing_aware=True)
    assert abs(voxel["LV"] - phys["LV"]) < 1e-6


def test_spacing_aware_changes_with_anisotropy():
    # A thin apical error should matter less than a thick-slice error when
    # voxels are weighted by physical volume — here we just check the API
    # returns a finite score in (0, 1).
    pred = np.zeros((4, 8, 8), dtype=np.uint8)
    gt = np.zeros_like(pred)
    pred[0, 2:6, 2:6] = 2
    gt[:, 2:6, 2:6] = 2
    scores = volumetric_dice(pred, gt, spacing_zyx=(10.0, 1.4, 1.4), spacing_aware=True)
    assert 0.0 < scores["MYO"] < 1.0
