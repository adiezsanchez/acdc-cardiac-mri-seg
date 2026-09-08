from __future__ import annotations

import numpy as np

from acdc_seg.constants import DEFAULT_SPACING_XYZ_MM
from acdc_seg.io_nifti import Volume, affine_from_spacing_xyz, describe_anisotropy, load_nifti, save_nifti


def test_roundtrip_nifti(tmp_path):
    spacing_xyz = DEFAULT_SPACING_XYZ_MM
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    data = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    vol = Volume(data=data, affine_xyz=affine_from_spacing_xyz(spacing_xyz), spacing_zyx=spacing_zyx)
    path = tmp_path / "vol.nii.gz"
    save_nifti(path, vol)
    loaded = load_nifti(path)
    assert loaded.data.shape == (2, 4, 5)
    np.testing.assert_allclose(loaded.data, data)
    np.testing.assert_allclose(loaded.spacing_zyx, spacing_zyx, rtol=1e-5)


def test_label_roundtrip(tmp_path):
    spacing_xyz = (1.4, 1.4, 8.0)
    spacing_zyx = (8.0, 1.4, 1.4)
    labels = np.zeros((3, 6, 6), dtype=np.float32)
    labels[1, 2:4, 2:4] = 3
    vol = Volume(data=labels, affine_xyz=affine_from_spacing_xyz(spacing_xyz), spacing_zyx=spacing_zyx)
    path = tmp_path / "lab.nii.gz"
    save_nifti(path, vol, is_label=True)
    loaded = load_nifti(path)
    assert loaded.data.dtype == np.float32
    assert set(np.unique(np.round(loaded.data))) <= {0, 3}


def test_anisotropy_ratio():
    info = describe_anisotropy((8.0, 1.4, 1.4))
    assert info["anisotropy_ratio"] == 8.0 / 1.4
    assert info["voxel_volume_mm3"] == 8.0 * 1.4 * 1.4
