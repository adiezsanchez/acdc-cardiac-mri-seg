"""ACDC-like synthetic short-axis cine volumes.

The official ACDC dataset is **not** redistributed here. These NIfTIs exist so
the Pixi demo can run end-to-end without a registration wall: they mimic the
folder layout, label ids, SSFP contrast, and through-plane anisotropy of ACDC.

Geometry (short-axis, base → apex along Z)
------------------------------------------
* LV cavity: ellipse, smaller toward the apex, smaller still at end-systole.
* Myocardium: annular wall around the LV; thicker at ES.
* RV cavity: crescent on the lateral side of the LV, fading at the apex.

Intensities follow cine-SSFP (blood bright, myocardium mid-grey) plus a
smooth bias field and Rician-like noise so a U-Net has something to learn
beyond a binary threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate

from acdc_seg.constants import (
    DATA_SYNTHETIC,
    DEFAULT_SPACING_XYZ_MM,
    LABEL_LV,
    LABEL_MYO,
    LABEL_RV,
    SYNTH_INTENSITY,
)
from acdc_seg.io_nifti import Volume, affine_from_spacing_xyz, save_nifti


@dataclass
class SynthSpec:
    n_patients: int = 10
    shape_hw: tuple[int, int] = (128, 128)
    n_slices: int = 10
    spacing_xyz_mm: tuple[float, float, float] = DEFAULT_SPACING_XYZ_MM
    seed: int = 2026


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(np.float32), xx.astype(np.float32)


def _ellipse(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    return ((yy - cy) / max(ry, 0.5)) ** 2 + ((xx - cx) / max(rx, 0.5)) ** 2 <= 1.0


def _short_axis_labels(
    h: int,
    w: int,
    z_norm: float,
    *,
    phase: str,
    rng: np.random.Generator,
    patient_jitter: dict[str, float],
) -> np.ndarray:
    """One short-axis slice. ``z_norm=0`` apex, ``z_norm=1`` base."""
    yy, xx = _grid(h, w)
    cy = h * 0.50 + patient_jitter["cy"]
    cx = w * 0.52 + patient_jitter["cx"]

    # Cavity size grows from apex to base; ES is a contracted copy of ED.
    scale_z = 0.42 + 0.58 * z_norm  # apex is ~42% of basal radius
    es_shrink = 0.62 if phase == "ES" else 1.0
    wall_boost = 1.35 if phase == "ES" else 1.0

    lv_rx = (w * 0.16 * scale_z * es_shrink) * patient_jitter["lv"]
    lv_ry = lv_rx * patient_jitter["lv_aspect"]
    wall = (w * 0.045 * wall_boost) * patient_jitter["wall"]

    lv = _ellipse(yy, xx, cy, cx, lv_ry, lv_rx)
    outer = _ellipse(yy, xx, cy, cx, lv_ry + wall, lv_rx + wall)
    myo = outer & ~lv

    # RV crescent: a larger ellipse shifted toward the patient's right, minus LV wall.
    # It is almost absent at the apex and prominent at the base.
    if z_norm < 0.18:
        rv = np.zeros((h, w), dtype=bool)
    else:
        rv_vis = np.clip((z_norm - 0.18) / 0.82, 0.0, 1.0)
        shift = (lv_rx + wall) * 0.95
        rv_cx = cx - shift * patient_jitter["rv_side"]
        rv_cy = cy + patient_jitter["rv_up"] * 0.02 * h
        rv_rx = lv_rx * (1.15 + 0.35 * rv_vis) * patient_jitter["rv"]
        rv_ry = lv_ry * (1.45 + 0.25 * rv_vis) * patient_jitter["rv"]
        rv = _ellipse(yy, xx, rv_cy, rv_cx, rv_ry, rv_rx) & ~outer
        # Hollow the RV a little so it stays a cavity, not a blob overlapping MYO.
        rv = rv & ~lv

    labels = np.zeros((h, w), dtype=np.uint8)
    labels[rv] = LABEL_RV
    labels[myo] = LABEL_MYO
    labels[lv] = LABEL_LV

    # Small in-plane rotation (degrees) to vary anatomy across patients.
    angle = patient_jitter["angle"]
    if abs(angle) > 0.05:
        labels = ndi_rotate(labels, angle=angle, reshape=False, order=0, prefilter=False)
        labels = np.clip(labels, 0, 3).astype(np.uint8)
    return labels


def _bias_field(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    # Very low-frequency coil-like shading.
    field = gaussian_filter(noise, sigma=(0.6, shape[1] / 8.0, shape[2] / 8.0))
    field = field - field.min()
    field = field / (field.max() + 1e-6)
    return 0.65 + 0.7 * field


def _synthesize_frame(
    spec: SynthSpec,
    *,
    phase: str,
    rng: np.random.Generator,
    patient_jitter: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    d = spec.n_slices
    h, w = spec.shape_hw
    labels = np.zeros((d, h, w), dtype=np.uint8)
    for z in range(d):
        z_norm = z / max(d - 1, 1)
        labels[z] = _short_axis_labels(h, w, z_norm, phase=phase, rng=rng, patient_jitter=patient_jitter)

    means = np.array(
        [
            SYNTH_INTENSITY["background"],
            SYNTH_INTENSITY["blood"],  # RV
            SYNTH_INTENSITY["myo"],
            SYNTH_INTENSITY["blood"],  # LV
        ],
        dtype=np.float32,
    )
    image = means[labels]
    # Slight texture so the myocardium is not a perfectly flat grey.
    texture = gaussian_filter(rng.normal(0.0, 8.0, size=image.shape).astype(np.float32), sigma=(0.0, 1.2, 1.2))
    image = image + texture
    image = image * _bias_field(image.shape, rng)
    noise_sigma = 7.5
    real = rng.normal(image, noise_sigma)
    imag = rng.normal(0.0, noise_sigma, size=image.shape)
    image = np.sqrt(real**2 + imag**2).astype(np.float32)
    return image, labels


def _patient_jitter(rng: np.random.Generator) -> dict[str, float]:
    return {
        "cy": float(rng.uniform(-4, 4)),
        "cx": float(rng.uniform(-5, 5)),
        "lv": float(rng.uniform(0.88, 1.14)),
        "lv_aspect": float(rng.uniform(0.90, 1.08)),
        "wall": float(rng.uniform(0.85, 1.20)),
        "rv": float(rng.uniform(0.85, 1.18)),
        "rv_side": float(rng.uniform(0.9, 1.15)),
        "rv_up": float(rng.uniform(-1.0, 1.0)),
        "angle": float(rng.uniform(-18, 18)),
    }


def _write_info_cfg(path: Path, ed: int, es: int) -> None:
    path.write_text(
        "\n".join(
            [
                f"ED: {ed}",
                f"ES: {es}",
                "Group: SYNTH",
                "Height: 170.0",
                "NbFrame: 12",
                "Weight: 70.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_synthetic_dataset(
    out_dir: str | Path = DATA_SYNTHETIC,
    spec: SynthSpec | None = None,
) -> Path:
    """Write ACDC-layout patient folders with ED/ES NIfTI frames."""
    spec = spec or SynthSpec()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    affine = affine_from_spacing_xyz(spec.spacing_xyz_mm)
    spacing_zyx = (
        spec.spacing_xyz_mm[2],
        spec.spacing_xyz_mm[1],
        spec.spacing_xyz_mm[0],
    )

    for i in range(1, spec.n_patients + 1):
        patient = out_dir / f"patient{i:03d}"
        patient.mkdir(parents=True, exist_ok=True)
        jitter = _patient_jitter(rng)
        ed_frame, es_frame = 1, 8
        _write_info_cfg(patient / "Info.cfg", ed_frame, es_frame)
        for phase, frame in (("ED", ed_frame), ("ES", es_frame)):
            image, labels = _synthesize_frame(spec, phase=phase, rng=rng, patient_jitter=jitter)
            stem = f"patient{i:03d}_frame{frame:02d}"
            save_nifti(
                patient / f"{stem}.nii.gz",
                Volume(data=image, affine_xyz=affine, spacing_zyx=spacing_zyx),
                is_label=False,
            )
            save_nifti(
                patient / f"{stem}_gt.nii.gz",
                Volume(data=labels.astype(np.float32), affine_xyz=affine, spacing_zyx=spacing_zyx),
                is_label=True,
            )
    return out_dir
