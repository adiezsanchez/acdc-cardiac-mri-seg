"""Volume-wise inference and volumetric Dice tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acdc_seg.constants import CHECKPOINTS_DIR, METRICS_DIR, RESULTS_DIR
from acdc_seg.data.dataset import CaseFrame
from acdc_seg.data.transforms import intensity_normalize, pad_or_crop_2d, pad_or_crop_3d
from acdc_seg.device import get_device
from acdc_seg.io_nifti import Volume, load_nifti, save_nifti
from acdc_seg.metrics import volumetric_dice
from acdc_seg.train import load_checkpoint


def _center_crop_params(src: int, dst: int) -> tuple[int, int]:
    """Return ``(src_start, dst_start)`` to copy ``min(src,dst)`` samples."""
    if src >= dst:
        return (src - dst) // 2, 0
    return 0, (dst - src) // 2


def _paste_2d(canvas: np.ndarray, patch: np.ndarray) -> None:
    hs, hc = _center_crop_params(patch.shape[0], canvas.shape[0])
    ws, wc = _center_crop_params(patch.shape[1], canvas.shape[1])
    h = min(canvas.shape[0] - hc, patch.shape[0] - hs)
    w = min(canvas.shape[1] - wc, patch.shape[1] - ws)
    canvas[hc : hc + h, wc : wc + w] = patch[hs : hs + h, ws : ws + w]


@torch.no_grad()
def predict_volume_2d(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int,
    device: torch.device,
    context: int = 0,
) -> np.ndarray:
    """Predict a ``(D, H, W)`` volume slice-by-slice (phase 1 or 2)."""
    model.eval()
    depth, h, w = image.shape
    pred = np.zeros_like(image, dtype=np.uint8)
    for z in range(depth):
        if context <= 0:
            sl, _ = pad_or_crop_2d(intensity_normalize(image[z]), np.zeros((h, w), np.uint8), size)
            tensor = torch.from_numpy(sl[None, None].astype(np.float32)).to(device)
        else:
            chans = []
            center = image[z]
            vmin, vmax = np.percentile(center, (1, 99))
            scale = max(vmax - vmin, 1e-6)
            for offset in range(-context, context + 1):
                zi = int(np.clip(z + offset, 0, depth - 1))
                chans.append(np.clip((image[zi] - vmin) / scale, 0.0, 1.0))
            stacked = np.stack(chans, axis=0)
            cropped = []
            dummy = np.zeros((h, w), np.uint8)
            for c in range(stacked.shape[0]):
                img_c, _ = pad_or_crop_2d(stacked[c], dummy, size)
                cropped.append(img_c)
            tensor = torch.from_numpy(np.stack(cropped)[None].astype(np.float32)).to(device)
        logits = model(tensor)
        sl_pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        _paste_2d(pred[z], sl_pred)
    return pred


@torch.no_grad()
def predict_volume_3d(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int,
    depth: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    d, h, w = image.shape
    norm = intensity_normalize(image)
    dummy = np.zeros_like(norm, dtype=np.uint8)
    cropped, _ = pad_or_crop_3d(norm, dummy, depth, size)
    tensor = torch.from_numpy(cropped[None, None].astype(np.float32)).to(device)
    logits = model(tensor)
    pred_c = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    canvas = np.zeros((d, h, w), dtype=np.uint8)
    # Paste the (depth, size, size) prediction back to the original grid.
    zs, zc = _center_crop_params(pred_c.shape[0], d)
    zd = min(d - zc, pred_c.shape[0] - zs)
    for zi in range(zd):
        _paste_2d(canvas[zc + zi], pred_c[zs + zi])
    return canvas


def evaluate_cases(
    *,
    phase: int,
    cases: list[CaseFrame],
    checkpoint: str | Path | None = None,
    size: int = 128,
    depth: int = 10,
    context: int = 1,
    device: torch.device | None = None,
    save_preds: bool = True,
) -> pd.DataFrame:
    device = device or get_device()
    checkpoint = Path(checkpoint or CHECKPOINTS_DIR / f"phase{phase}_best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}. Train this phase first.")
    in_channels = 1 if phase in (1, 3) else 2 * context + 1
    model = load_checkpoint(checkpoint, phase, device=device)
    # If the checkpoint stored in_channels, load_checkpoint already set it.

    rows = []
    pred_dir = RESULTS_DIR / "predictions" / f"phase{phase}"
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        vol = load_nifti(case.image_path)
        gt = load_nifti(case.label_path, dtype=np.float32)
        if phase == 1:
            pred = predict_volume_2d(model, vol.data, size=size, device=device, context=0)
        elif phase == 2:
            pred = predict_volume_2d(model, vol.data, size=size, device=device, context=context)
        else:
            pred = predict_volume_3d(model, vol.data, size=size, depth=depth, device=device)
        gt_lab = np.round(gt.data).astype(np.uint8)
        scores = volumetric_dice(pred, gt_lab, spacing_zyx=vol.spacing_zyx, spacing_aware=False)
        phys = volumetric_dice(pred, gt_lab, spacing_zyx=vol.spacing_zyx, spacing_aware=True)
        row = {
            "phase": phase,
            "patient": case.patient_id,
            "frame": case.frame,
            "dice_RV": scores["RV"],
            "dice_MYO": scores["MYO"],
            "dice_LV": scores["LV"],
            "dice_mean": scores["mean"],
            "dice_phys_mean": phys["mean"],
            "anisotropy": vol.anisotropy_ratio,
            "spacing_zyx": vol.spacing_zyx,
        }
        rows.append(row)
        if save_preds:
            out = Volume(data=pred.astype(np.float32), affine_xyz=vol.affine_xyz, spacing_zyx=vol.spacing_zyx)
            save_nifti(pred_dir / f"{case.patient_id}_frame{case.frame:02d}_pred.nii.gz", out, is_label=True)

    df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / f"phase{phase}_test_dice.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "phase": phase,
        "n_volumes": int(len(df)),
        "mean": {k: float(df[k].mean()) for k in ("dice_RV", "dice_MYO", "dice_LV", "dice_mean", "dice_phys_mean")},
    }
    (METRICS_DIR / f"phase{phase}_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df
