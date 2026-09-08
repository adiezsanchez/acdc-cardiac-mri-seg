"""Discover ACDC-layout cases and build 2D / 2.5D / 3D torch datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from acdc_seg.data.transforms import augment_2d, augment_stack, intensity_normalize, pad_or_crop_2d, pad_or_crop_3d
from acdc_seg.io_nifti import Volume, load_nifti

_FRAME_RE = re.compile(r"(patient\d+)_frame(\d+)\.nii(?:\.gz)?$", re.IGNORECASE)


@dataclass
class CaseFrame:
    """One ED or ES volume pair."""

    patient_id: str
    frame: int
    image_path: Path
    label_path: Path
    info: dict[str, str]


def parse_info_cfg(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.is_file():
        return info
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def discover_cases(root: str | Path) -> list[CaseFrame]:
    """Walk ``root`` (or ``root/training``) for ``patientXXX_frameYY.nii.gz`` + GT.

    Compatible with:
    * this repo's ``data/synthetic/``
    * official ACDC ``training/patientXXX/`` after you download it yourself
    """
    root = Path(root)
    if (root / "training").is_dir():
        search = root / "training"
    else:
        search = root
    cases: list[CaseFrame] = []
    for patient_dir in sorted(p for p in search.iterdir() if p.is_dir() and p.name.startswith("patient")):
        info = parse_info_cfg(patient_dir / "Info.cfg")
        for image_path in sorted(patient_dir.glob("*.nii*")):
            name = image_path.name
            if "_gt" in name or "_4d" in name:
                continue
            match = _FRAME_RE.search(name)
            if match is None:
                continue
            patient_id, frame_s = match.group(1), match.group(2)
            label_path = image_path.with_name(f"{patient_id}_frame{frame_s}_gt.nii.gz")
            if not label_path.is_file():
                # Some dumps drop the .gz
                alt = image_path.with_name(f"{patient_id}_frame{frame_s}_gt.nii")
                if alt.is_file():
                    label_path = alt
                else:
                    continue
            cases.append(
                CaseFrame(
                    patient_id=patient_id,
                    frame=int(frame_s),
                    image_path=image_path,
                    label_path=label_path,
                    info=info,
                )
            )
    if not cases:
        raise FileNotFoundError(
            f"No ACDC-style frames under {root}. Run `pixi run synth-data` or download ACDC (see README)."
        )
    return cases


def split_cases(
    cases: list[CaseFrame],
    *,
    val_patients: int = 2,
    test_patients: int = 2,
) -> dict[str, list[CaseFrame]]:
    """Deterministic patient-level split (no slice leakage across splits)."""
    patients = sorted({c.patient_id for c in cases})
    if len(patients) < val_patients + test_patients + 1:
        # Tiny sets: 1 patient each for val/test if possible.
        val_patients = min(val_patients, 1)
        test_patients = min(test_patients, 1)
    test_ids = set(patients[-test_patients:]) if test_patients else set()
    remaining = [p for p in patients if p not in test_ids]
    val_ids = set(remaining[-val_patients:]) if val_patients else set()
    train_ids = set(p for p in remaining if p not in val_ids)
    grouped = {"train": [], "val": [], "test": []}
    for case in cases:
        if case.patient_id in train_ids:
            grouped["train"].append(case)
        elif case.patient_id in val_ids:
            grouped["val"].append(case)
        else:
            grouped["test"].append(case)
    return grouped


def _load_pair(case: CaseFrame) -> tuple[Volume, Volume]:
    image = load_nifti(case.image_path, dtype=np.float32)
    label = load_nifti(case.label_path, dtype=np.float32)
    if image.data.shape != label.data.shape:
        raise ValueError(f"Shape mismatch {case.image_path} vs {case.label_path}")
    return image, label


class CardiacSliceDataset(Dataset):
    """Phase 1: each short-axis slice is an independent 2D sample."""

    def __init__(
        self,
        cases: list[CaseFrame],
        *,
        size: int = 128,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.slices: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
        self.meta: list[dict[str, object]] = []
        for case in cases:
            image, label = _load_pair(case)
            for z in range(image.data.shape[0]):
                self.slices.append((image.data[z], label.data[z], image.spacing_zyx))
                self.meta.append({"patient": case.patient_id, "frame": case.frame, "z": z})

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, label, spacing = self.slices[idx]
        image = intensity_normalize(image)
        image, label = pad_or_crop_2d(image, label, self.size)
        if self.augment:
            image, label = augment_2d(image, label, self.rng)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }


class NeighborSliceDataset(Dataset):
    """Phase 2: 2.5D — stack ``2*context+1`` adjacent slices as input channels.

    The target is still the *center* slice. Boundary slices use replicate
    padding. 2D convolutions then see a thin slab of through-plane context
    without the cost of 3D kernels.
    """

    def __init__(
        self,
        cases: list[CaseFrame],
        *,
        size: int = 128,
        context: int = 1,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.context = context
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.items: list[tuple[np.ndarray, np.ndarray, int, tuple[float, float, float]]] = []
        for case in cases:
            image, label = _load_pair(case)
            depth = image.data.shape[0]
            for z in range(depth):
                self.items.append((image.data, label.data[z], z, image.spacing_zyx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        volume, label, z, spacing = self.items[idx]
        depth = volume.shape[0]
        channels = []
        for offset in range(-self.context, self.context + 1):
            zi = int(np.clip(z + offset, 0, depth - 1))
            channels.append(volume[zi])
        stack = np.stack(channels, axis=0)
        center = self.context
        # Normalize neighbors with the same stats as the center slice.
        vmin, vmax = np.percentile(stack[center], (1, 99))
        scale = max(vmax - vmin, 1e-6)
        stack = np.clip((stack - vmin) / scale, 0.0, 1.0).astype(np.float32)
        label_out = label
        cropped = []
        for c in range(stack.shape[0]):
            img_c, lab_c = pad_or_crop_2d(stack[c], label_out if c == center else label_out, self.size)
            cropped.append(img_c)
            if c == center:
                label_out = lab_c
        stack = np.stack(cropped, axis=0)
        if self.augment:
            stack, label_out = augment_stack(stack, label_out, self.rng)
        return {
            "image": torch.from_numpy(stack.astype(np.float32)),
            "label": torch.from_numpy(label_out.astype(np.int64)),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }


class CardiacVolumeDataset(Dataset):
    """Phase 3: whole ED/ES volumes for a light 3D U-Net."""

    def __init__(
        self,
        cases: list[CaseFrame],
        *,
        size_hw: int = 128,
        depth: int = 10,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.size_hw = size_hw
        self.depth = depth
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.volumes: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
        self.meta: list[dict[str, object]] = []
        for case in cases:
            image, label = _load_pair(case)
            self.volumes.append((image.data, label.data, image.spacing_zyx))
            self.meta.append({"patient": case.patient_id, "frame": case.frame})

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, label, spacing = self.volumes[idx]
        image = intensity_normalize(image)
        image, label = pad_or_crop_3d(image, label, self.depth, self.size_hw)
        if self.augment and self.rng.random() < 0.5:
            image = image[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        if self.augment and self.rng.random() < 0.5:
            image = image[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
