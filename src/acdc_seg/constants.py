"""ACDC label convention, class colors, and default paths.

The official ACDC ground-truth NIfTI files use integer labels:

    0  background
    1  right ventricle cavity (RV)
    2  left-ventricle myocardium (MYO)
    3  left-ventricle cavity (LV)

This mapping is used by every phase so a checkpoint from the 2D U-Net can be
compared with the 2.5D and 3D models on the same Dice tables.
"""

from __future__ import annotations

from pathlib import Path

# Integer labels stored in NIfTI ground truth (Bernard et al., IEEE TMI 2018).
LABEL_BACKGROUND = 0
LABEL_RV = 1
LABEL_MYO = 2
LABEL_LV = 3

NUM_CLASSES = 4

CLASS_NAMES: dict[int, str] = {
    LABEL_BACKGROUND: "background",
    LABEL_RV: "RV",
    LABEL_MYO: "MYO",
    LABEL_LV: "LV",
}

FOREGROUND_LABELS: tuple[int, ...] = (LABEL_RV, LABEL_MYO, LABEL_LV)
FOREGROUND_NAMES: tuple[str, ...] = ("RV", "MYO", "LV")

# Plotly / overlay colors. RV blue, myocardium green, LV red — a common
# cardiac-MRI convention that keeps the three structures separable.
CLASS_COLORS: dict[int, str] = {
    LABEL_RV: "#1f77b4",
    LABEL_MYO: "#2ca02c",
    LABEL_LV: "#d62728",
}

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    LABEL_RV: (31, 119, 180),
    LABEL_MYO: (44, 160, 44),
    LABEL_LV: (214, 39, 40),
}

# Typical ACDC cine-SSFP voxel size. In-plane ~1.4 mm, through-plane 5–10 mm.
# The large slice thickness is the anisotropy the three phases are designed
# to confront.
DEFAULT_SPACING_XYZ_MM: tuple[float, float, float] = (1.4, 1.4, 8.0)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SYNTHETIC = REPO_ROOT / "data" / "synthetic"
DATA_RAW = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CONFIGS_DIR = REPO_ROOT / "configs"

# SSFP-like intensities used by the synthetic generator (blood bright).
SYNTH_INTENSITY = {
    "background": 18.0,
    "myo": 95.0,
    "blood": 185.0,
}
