"""Plotly-only figures: contours, training curves, Dice bars, 3D meshes.

Matplotlib is intentionally not a dependency. Interactive HTML is always
written; PNG export uses Kaleido when available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

from acdc_seg.constants import (
    CLASS_COLORS,
    FIGURES_DIR,
    FOREGROUND_LABELS,
    FOREGROUND_NAMES,
    LABEL_LV,
    LABEL_MYO,
    LABEL_RV,
    METRICS_DIR,
)
from acdc_seg.io_nifti import Volume

PLOTLY_TEMPLATE = "plotly_white"


def _write(fig: go.Figure, stem: str, *, width: int = 1100, height: int = 720) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template=PLOTLY_TEMPLATE, width=width, height=height, margin=dict(l=40, r=20, t=60, b=40))
    html_path = FIGURES_DIR / f"{stem}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    written = [html_path]
    png_path = FIGURES_DIR / f"{stem}.png"
    try:
        fig.write_image(str(png_path), scale=2)
        written.append(png_path)
    except Exception as exc:  # kaleido / chrome may be missing in some CI images
        (FIGURES_DIR / f"{stem}_png_export_skipped.txt").write_text(
            f"PNG export skipped ({type(exc).__name__}: {exc})\nHTML is available at {html_path.name}\n",
            encoding="utf-8",
        )
    return written


def _contour_trace(mask: np.ndarray, color: str, name: str) -> go.Contour:
    return go.Contour(
        z=mask.astype(np.float32),
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
        line=dict(color=color, width=2),
        name=name,
        hoverinfo="skip",
    )


def slice_with_contours(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    pred: np.ndarray | None = None,
    title: str = "Short-axis slice",
    stem: str = "slice_contours",
) -> list[Path]:
    """MRI slice as a heatmap plus GT (solid) and optional pred (dashed) contours."""
    cols = 1 if pred is None else 2
    titles = [title] if pred is None else [f"{title} — ground truth", f"{title} — prediction"]
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    panels = [(image, labels, 1)]
    if pred is not None:
        panels.append((image, pred, 2))
    for img, lab, col in panels:
        fig.add_trace(
            go.Heatmap(z=img, colorscale="Gray", showscale=col == 1, name="MRI"),
            row=1,
            col=col,
        )
        for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
            fig.add_trace(_contour_trace(lab == label, CLASS_COLORS[label], name if col == 1 else f"{name} pred"), row=1, col=col)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if col == 1 else col}", scaleratio=1, row=1, col=col)
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    fig.update_layout(title="Multi-class contour overlay (RV / MYO / LV)")
    return _write(fig, stem, width=1200 if pred is not None else 700, height=640)


def slice_montage(
    volume: np.ndarray,
    labels: np.ndarray,
    *,
    pred: np.ndarray | None = None,
    max_slices: int = 8,
    stem: str = "slice_montage",
    title: str = "Base → apex short-axis stack",
) -> list[Path]:
    depth = volume.shape[0]
    idxs = np.linspace(0, depth - 1, num=min(max_slices, depth), dtype=int)
    n = len(idxs)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[f"z={int(i)}" for i in idxs])
    for col, z in enumerate(idxs, start=1):
        fig.add_trace(go.Heatmap(z=volume[z], colorscale="Gray", showscale=False), row=1, col=col)
        for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
            fig.add_trace(_contour_trace(labels[z] == label, CLASS_COLORS[label], name if col == 1 else None), row=1, col=col)
        if pred is not None:
            for label in FOREGROUND_LABELS:
                fig.add_trace(
                    go.Contour(
                        z=(pred[z] == label).astype(np.float32),
                        showscale=False,
                        contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                        line=dict(color=CLASS_COLORS[label], width=1, dash="dash"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if col == 1 else col}", scaleratio=1, row=1, col=col)
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    fig.update_layout(title=title)
    return _write(fig, stem, width=min(220 * n + 80, 1600), height=380)


def anisotropy_figure(
    spacing_zyx: tuple[float, float, float],
    volume: np.ndarray,
    labels: np.ndarray,
    stem: str = "anisotropy",
) -> list[Path]:
    """Show why voxel Dice ≠ physical Dice: a mid-slice vs a coronal reformatting."""
    sz, sy, sx = spacing_zyx
    zmid = volume.shape[0] // 2
    # Coronal slab: axis 1 (Y) mid, displayed with physical aspect (Z stretched).
    ymid = volume.shape[1] // 2
    coronal_img = volume[:, ymid, :]
    coronal_lab = labels[:, ymid, :]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Short-axis (in-plane {sx:.1f}×{sy:.1f} mm)",
            f"Coronal reformat (slice thickness {sz:.1f} mm)",
        ),
    )
    fig.add_trace(go.Heatmap(z=volume[zmid], colorscale="Gray", showscale=False), row=1, col=1)
    for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
        fig.add_trace(_contour_trace(labels[zmid] == label, CLASS_COLORS[label], name), row=1, col=1)
    fig.add_trace(go.Heatmap(z=coronal_img, colorscale="Gray", showscale=False), row=1, col=2)
    for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
        fig.add_trace(_contour_trace(coronal_lab == label, CLASS_COLORS[label], None), row=1, col=2)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
    # Physical aspect for coronal: each row is sz mm, each col is sx mm.
    fig.update_yaxes(autorange="reversed", scaleanchor="x2", scaleratio=sz / max(sx, 1e-6), row=1, col=2)
    fig.update_layout(
        title=f"ACDC-like anisotropy: through-plane / in-plane = {sz / max(0.5 * (sx + sy), 1e-6):.1f}×"
    )
    return _write(fig, stem, width=1100, height=560)


def training_curves(history_paths: dict[int, Path], stem: str = "training_curves") -> list[Path]:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Validation volumetric Dice"))
    palette = {1: "#636EFA", 2: "#EF553B", 3: "#00CC96"}
    names = {1: "Phase 1 — 2D U-Net", 2: "Phase 2 — neighbouring slices", 3: "Phase 3 — light 3D U-Net"}
    for phase, path in sorted(history_paths.items()):
        if not path.is_file():
            continue
        hist = json.loads(path.read_text(encoding="utf-8"))
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["train_loss"],
                mode="lines+markers",
                name=f"{names[phase]} train",
                line=dict(color=palette[phase]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["val_loss"],
                mode="lines+markers",
                name=f"{names[phase]} val",
                line=dict(color=palette[phase], dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["val_dice"],
                mode="lines+markers",
                name=names[phase],
                line=dict(color=palette[phase]),
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Dice+CE loss", row=1, col=1)
    fig.update_yaxes(title_text="Mean Dice (RV/MYO/LV)", range=[0, 1], row=1, col=2)
    fig.update_layout(title="Demo training curves (synthetic ACDC-like volumes)")
    return _write(fig, stem, width=1200, height=520)


def dice_bars(summaries: dict[int, dict], stem: str = "volumetric_dice") -> list[Path]:
    phases = sorted(summaries)
    fig = go.Figure()
    for cls, color in (("dice_RV", CLASS_COLORS[LABEL_RV]), ("dice_MYO", CLASS_COLORS[LABEL_MYO]), ("dice_LV", CLASS_COLORS[LABEL_LV])):
        short = cls.replace("dice_", "")
        fig.add_trace(
            go.Bar(
                x=[f"Phase {p}" for p in phases],
                y=[summaries[p]["mean"][cls] for p in phases],
                name=short,
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group",
        title="Volumetric Dice on held-out synthetic volumes",
        yaxis=dict(title="Dice", range=[0, 1]),
    )
    return _write(fig, stem, width=900, height=520)


def reconstruction_3d(
    labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    pred: np.ndarray | None = None,
    stem: str = "reconstruction_3d",
    title: str = "Simple 3D reconstruction (marching cubes → Plotly Mesh3d)",
) -> list[Path]:
    """Surface meshes in millimetres so the stack looks like a real ventricle, not a cube."""
    cols = 1 if pred is None else 2
    titles = ["Ground truth"] if pred is None else ["Ground truth", "Prediction"]
    fig = make_subplots(rows=1, cols=cols, specs=[[{"type": "scene"}] * cols], subplot_titles=titles)

    def _add_meshes(vol: np.ndarray, col: int, legend: bool) -> None:
        for label, name in zip(FOREGROUND_LABELS, FOREGROUND_NAMES, strict=True):
            mask = vol == label
            if mask.sum() < 20:
                continue
            try:
                verts, faces, *_ = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing_zyx)
            except (ValueError, RuntimeError):
                continue
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 2],  # X mm
                    y=verts[:, 1],  # Y mm
                    z=verts[:, 0],  # Z mm (through-plane)
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=CLASS_COLORS[label],
                    opacity=0.45 if label == LABEL_MYO else 0.35,
                    name=name,
                    showlegend=legend,
                    lighting=dict(ambient=0.45, diffuse=0.6),
                    flatshading=True,
                ),
                row=1,
                col=col,
            )

    _add_meshes(labels, 1, True)
    if pred is not None:
        _add_meshes(pred, 2, False)
    scene = dict(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        aspectmode="data",
    )
    fig.update_layout(title=title)
    fig.update_scenes(scene)
    return _write(fig, stem, width=1100 if pred is not None else 700, height=700)


def write_all_demo_figures(
    *,
    image: Volume,
    labels: np.ndarray,
    pred: np.ndarray | None,
    history_paths: dict[int, Path],
    summaries: dict[int, dict],
) -> list[Path]:
    labels_u8 = np.round(labels).astype(np.uint8)
    mid = image.data.shape[0] // 2
    written: list[Path] = []
    written += slice_with_contours(image.data[mid], labels_u8[mid], pred=None if pred is None else pred[mid], stem="01_slice_gt_contours")
    written += slice_montage(image.data, labels_u8, pred=pred, stem="02_slice_montage")
    written += anisotropy_figure(image.spacing_zyx, image.data, labels_u8, stem="03_anisotropy")
    written += training_curves(history_paths, stem="04_training_curves")
    if summaries:
        written += dice_bars(summaries, stem="05_volumetric_dice")
    written += reconstruction_3d(labels_u8, image.spacing_zyx, pred=pred, stem="06_reconstruction_3d")
    return written
