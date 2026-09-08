from __future__ import annotations

import numpy as np

from acdc_seg.constants import LABEL_LV, LABEL_MYO, LABEL_RV
from acdc_seg.data.dataset import discover_cases
from acdc_seg.data.synthetic import SynthSpec, generate_synthetic_dataset
from acdc_seg.io_nifti import load_nifti


def test_synthetic_layout_and_labels(tmp_path):
    out = generate_synthetic_dataset(tmp_path, spec=SynthSpec(n_patients=2, n_slices=6, seed=0))
    cases = discover_cases(out)
    assert len(cases) == 4  # 2 patients × ED/ES
    vol = load_nifti(cases[0].image_path)
    gt = load_nifti(cases[0].label_path)
    assert vol.data.shape[0] == 6
    labels = set(np.unique(np.round(gt.data).astype(int)))
    assert {LABEL_RV, LABEL_MYO, LABEL_LV}.issubset(labels)
    assert vol.anisotropy_ratio > 3.0
    assert (out / "patient001" / "Info.cfg").is_file()
