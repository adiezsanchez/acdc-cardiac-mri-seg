"""Napari viewer for interactive multi-slice MRI + multi-class overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acdc_seg.constants import CLASS_COLORS_RGB, LABEL_LV, LABEL_MYO, LABEL_RV, NUM_CLASSES
from acdc_seg.io_nifti import load_nifti


def _label_colormap() -> np.ndarray:
    cmap = np.zeros((NUM_CLASSES, 4), dtype=np.float32)
    cmap[LABEL_RV] = (*[c / 255 for c in CLASS_COLORS_RGB[LABEL_RV]], 1.0)
    cmap[LABEL_MYO] = (*[c / 255 for c in CLASS_COLORS_RGB[LABEL_MYO]], 1.0)
    cmap[LABEL_LV] = (*[c / 255 for c in CLASS_COLORS_RGB[LABEL_LV]], 1.0)
    return cmap


def view_nifti(
    image_path: str | Path,
    label_path: str | Path | None = None,
    pred_path: str | Path | None = None,
    *,
    physical_scale: bool = True,
) -> None:
    """Open cine MRI + GT/pred label layers.

    ``physical_scale=True`` applies NIfTI zooms so Napari's 3D view shows the
    true pancake anisotropy (thick slices). Toggle it off to inspect voxel
    space, where the stack looks like a cube.
    """
    try:
        import napari
    except ImportError as exc:
        raise SystemExit(
            "Napari is not importable in this environment. Use `pixi run napari-view` "
            "after `pixi install`. A display (or Windows desktop / X11 / WSLg) is required."
        ) from exc

    image = load_nifti(image_path)
    scale = image.spacing_zyx if physical_scale else (1.0, 1.0, 1.0)
    cmap = _label_colormap()
    viewer = napari.Viewer(title="ACDC cardiac MRI — Napari")
    viewer.add_image(
        image.data,
        name="cine-SSFP",
        colormap="gray",
        scale=scale,
        blending="additive",
    )
    if label_path is not None:
        gt = np.round(load_nifti(label_path).data).astype(np.uint8)
        viewer.add_labels(gt, name="GT  RV/MYO/LV", scale=scale, opacity=0.45, color=_labels_color_dict(cmap))
    if pred_path is not None and Path(pred_path).is_file():
        pred = np.round(load_nifti(pred_path).data).astype(np.uint8)
        viewer.add_labels(pred, name="Pred RV/MYO/LV", scale=scale, opacity=0.45, color=_labels_color_dict(cmap))
    print(
        "Napari controls: scroll = slice, Ctrl+Y (or the 3D button) = volume view. "
        f"scale={scale} mm  (physical_scale={physical_scale})"
    )
    napari.run()


def _labels_color_dict(cmap: np.ndarray) -> dict[int, np.ndarray]:
    return {i: cmap[i] for i in range(1, NUM_CLASSES)}
