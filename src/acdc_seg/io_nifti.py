"""NIfTI I/O, voxel spacing, and anisotropy helpers.

Array convention used everywhere after loading
---------------------------------------------
nibabel returns arrays as ``(X, Y, Z)`` with an affine that maps voxel
indices to millimetres. Training code prefers ``(Z, Y, X)`` i.e.
``(D, H, W)`` so 2D slices are ``image[z]`` with shape ``(H, W)``.

All public loaders in this module convert to ``(D, H, W)`` and report
spacing as ``(sz, sy, sx)`` millimetres (depth, row, column). Saving
converts back to ``(X, Y, Z)`` so the written NIfTI stays RAS-friendly
and compatible with ITK-SNAP / 3D Slicer / the official ACDC files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom

Array = np.ndarray


@dataclass
class Volume:
    """One cine frame (image or label) plus geometry."""

    data: Array  # (D, H, W)
    affine_xyz: np.ndarray  # 4x4, maps (x, y, z) voxel indices to mm
    spacing_zyx: tuple[float, float, float]  # millimetres

    @property
    def shape(self) -> tuple[int, int, int]:
        d, h, w = (int(x) for x in self.data.shape)
        return d, h, w

    @property
    def voxel_volume_mm3(self) -> float:
        sz, sy, sx = self.spacing_zyx
        return float(sx * sy * sz)

    @property
    def anisotropy_ratio(self) -> float:
        """Through-plane / in-plane spacing. ACDC is typically 4–7×."""
        sz, sy, sx = self.spacing_zyx
        in_plane = 0.5 * (sx + sy)
        return float(sz / max(in_plane, 1e-6))


def _xyz_to_zyx(data_xyz: Array) -> Array:
    if data_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data_xyz.shape}")
    return np.transpose(data_xyz, (2, 1, 0))


def _zyx_to_xyz(data_zyx: Array) -> Array:
    return np.transpose(data_zyx, (2, 1, 0))


def load_nifti(path: str | Path, *, dtype: np.dtype | type | None = None) -> Volume:
    """Load a NIfTI image or label map as ``(D, H, W)``."""
    path = Path(path)
    img = nib.load(str(path))
    data_xyz = np.asanyarray(img.dataobj)
    if data_xyz.ndim == 4:
        # ACDC ships a 4D cine (``patientXXX_4d.nii.gz``). Take the first
        # time point — callers that need ED/ES should load the frame files.
        data_xyz = data_xyz[..., 0]
    if dtype is not None:
        data_xyz = data_xyz.astype(dtype, copy=False)
    else:
        data_xyz = data_xyz.astype(np.float32, copy=False)
    data_zyx = np.ascontiguousarray(_xyz_to_zyx(data_xyz))
    zooms = img.header.get_zooms()[:3]
    spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    affine = np.array(img.affine, dtype=np.float64)
    return Volume(data=data_zyx, affine_xyz=affine, spacing_zyx=spacing_zyx)


def save_nifti(path: str | Path, volume: Volume, *, is_label: bool = False) -> None:
    """Write ``(D, H, W)`` data back to a NIfTI file with the original affine."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_xyz = _zyx_to_xyz(volume.data)
    if is_label:
        data_xyz = np.round(data_xyz).astype(np.uint8)
    else:
        data_xyz = data_xyz.astype(np.float32)
    img = nib.Nifti1Image(data_xyz, volume.affine_xyz)
    sz, sy, sx = volume.spacing_zyx
    hdr = img.header
    hdr.set_zooms((sx, sy, sz))
    if is_label:
        hdr.set_data_dtype(np.uint8)
    nib.save(img, str(path))


def spacing_from_header(path: str | Path) -> tuple[float, float, float]:
    """Return ``(sx, sy, sz)`` millimetres from a NIfTI header (XYZ order)."""
    img = nib.load(str(path))
    zooms = img.header.get_zooms()[:3]
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def describe_anisotropy(spacing_zyx: tuple[float, float, float]) -> dict[str, float]:
    """Summarise why a 3×3×3 kernel is a bad default on ACDC."""
    sz, sy, sx = (float(v) for v in spacing_zyx)
    in_plane = 0.5 * (sx + sy)
    return {
        "sx_mm": sx,
        "sy_mm": sy,
        "sz_mm": sz,
        "in_plane_mm": in_plane,
        "anisotropy_ratio": sz / max(in_plane, 1e-6),
        "voxel_volume_mm3": sx * sy * sz,
    }


def resample_isotropic(volume: Volume, target_mm: float = 1.4, *, order: int = 1) -> Volume:
    """Resample to isotropic voxels (pedagogical; used optionally in phase 3).

    ACDC slices are 5–10 mm thick. Isotropic resampling fabricates extra
    slices — useful for 3D kernels, expensive in memory. Labels must use
    ``order=0`` (nearest neighbour) to keep integer class ids.
    """
    sz, sy, sx = volume.spacing_zyx
    zoom_factors = (sz / target_mm, sy / target_mm, sx / target_mm)
    resampled = ndi_zoom(volume.data, zoom_factors, order=order, prefilter=order > 1)
    affine = volume.affine_xyz.copy()
    # Scale the spatial columns of the XYZ affine to the new voxel size.
    # data was (Z,Y,X); affine maps X,Y,Z. New spacing is isotropic target_mm.
    scales = np.array([target_mm / sx, target_mm / sy, target_mm / sz, 1.0])
    affine = affine @ np.diag(scales)
    new_spacing = (target_mm, target_mm, target_mm)
    return Volume(data=resampled, affine_xyz=affine, spacing_zyx=new_spacing)


def affine_from_spacing_xyz(spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Axis-aligned affine with the given XYZ millimetre spacing."""
    sx, sy, sz = spacing_xyz
    return np.diag([sx, sy, sz, 1.0]).astype(np.float64)
