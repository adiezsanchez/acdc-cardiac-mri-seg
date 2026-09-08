"""Command-line entry points used by Pixi tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from acdc_seg.constants import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    DATA_SYNTHETIC,
    FIGURES_DIR,
    METRICS_DIR,
    RESULTS_DIR,
)
from acdc_seg.data.dataset import (
    CardiacSliceDataset,
    CardiacVolumeDataset,
    NeighborSliceDataset,
    discover_cases,
    split_cases,
)
from acdc_seg.data.synthetic import SynthSpec, generate_synthetic_dataset
from acdc_seg.device import describe_device, get_device
from acdc_seg.evaluate import evaluate_cases
from acdc_seg.io_nifti import load_nifti
from acdc_seg.train import run_training
from acdc_seg.visualize.plotly_viz import write_all_demo_figures


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def cmd_check_stack(_: argparse.Namespace) -> int:
    import napari
    import plotly
    import torch

    info = describe_device()
    print("PyTorch        ", torch.__version__)
    print("CUDA compiled  ", torch.version.cuda)
    print("CUDA available ", torch.cuda.is_available())
    print("Device         ", info.get("device"), info.get("gpu_name", ""))
    print("Napari         ", napari.__version__)
    print("Plotly         ", plotly.__version__)
    print("OK — CUDA-enabled PyTorch + Napari + Plotly are importable.")
    return 0


def cmd_check_cuda(_: argparse.Namespace) -> int:
    info = describe_device()
    print(json.dumps(info, indent=2))
    if not info["cuda_available"]:
        print(
            "\nPyTorch is the GPU build (see torch.version.cuda) but no NVIDIA GPU "
            "is visible. Training will fall back to CPU. Install an NVIDIA driver "
            "and re-run `pixi info` — you should see a `__cuda=...` virtual package."
        )
        return 0
    print("\nCUDA is ready.")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    spec = SynthSpec(
        n_patients=int(args.n_patients),
        n_slices=int(args.n_slices),
        seed=int(args.seed),
    )
    out = generate_synthetic_dataset(args.out, spec=spec)
    print(f"Wrote synthetic ACDC-layout dataset to {out}")
    return 0


def _loaders(phase: int, cfg: dict, split: dict, *, train: bool) -> tuple[DataLoader, DataLoader]:
    size = int(cfg.get("size", 128))
    context = int(cfg.get("context", 1))
    depth = int(cfg.get("depth", 10))
    bs = int(cfg.get("batch_size", 8 if phase < 3 else 2))
    if phase == 1:
        train_ds = CardiacSliceDataset(split["train"], size=size, augment=train, seed=1)
        val_ds = CardiacSliceDataset(split["val"], size=size, augment=False, seed=2)
    elif phase == 2:
        train_ds = NeighborSliceDataset(split["train"], size=size, context=context, augment=train, seed=1)
        val_ds = NeighborSliceDataset(split["val"], size=size, context=context, augment=False, seed=2)
    else:
        train_ds = CardiacVolumeDataset(split["train"], size_hw=size, depth=depth, augment=train, seed=1)
        val_ds = CardiacVolumeDataset(split["val"], size_hw=size, depth=depth, augment=False, seed=2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(1, bs // 2), shuffle=False, num_workers=0)
    return train_loader, val_loader


_PHASE_CONFIG = {
    1: CONFIGS_DIR / "phase1_2d.yaml",
    2: CONFIGS_DIR / "phase2_neighbors.yaml",
    3: CONFIGS_DIR / "phase3_3d.yaml",
}


def cmd_train(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config) if args.config else _PHASE_CONFIG[int(args.phase)]
    cfg = _load_yaml(cfg_path)
    root = Path(cfg.get("data_root", args.data_root))
    cases = discover_cases(root)
    split = split_cases(cases)
    phase = int(args.phase)
    epochs = int(args.epochs or cfg.get("epochs", 8))
    lr = float(cfg.get("lr", 1e-3))
    context = int(cfg.get("context", 1))
    in_channels = 1 if phase != 2 else 2 * context + 1
    train_loader, val_loader = _loaders(phase, cfg, split, train=True)
    result = run_training(
        phase=phase,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        in_channels=in_channels,
        base=cfg.get("base"),
        device=get_device(prefer_cuda=not args.cpu),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load_yaml(CONFIGS_DIR / "demo.yaml")
    root = Path(cfg.get("data_root", args.data_root))
    split = split_cases(discover_cases(root))
    phases = (1, 2, 3) if args.phase == "all" else (int(args.phase),)
    for phase in phases:
        ckpt = CHECKPOINTS_DIR / f"phase{phase}_best.pt"
        if not ckpt.is_file():
            print(f"Skipping phase {phase}: no checkpoint at {ckpt}")
            continue
        df = evaluate_cases(
            phase=phase,
            cases=split["test"],
            checkpoint=ckpt,
            size=int(cfg.get("size", 128)),
            depth=int(cfg.get("depth", 10)),
            context=int(cfg.get("context", 1)),
            device=get_device(prefer_cuda=not args.cpu),
        )
        print(f"\n=== phase {phase} volumetric Dice (test) ===")
        print(df[["patient", "frame", "dice_RV", "dice_MYO", "dice_LV", "dice_mean"]].to_string(index=False))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    cfg = _load_yaml(CONFIGS_DIR / "demo.yaml")
    root = Path(cfg.get("data_root", args.data_root))
    split = split_cases(discover_cases(root))
    case = split["test"][0]
    image = load_nifti(case.image_path)
    labels = load_nifti(case.label_path).data
    pred = None
    pred_path = RESULTS_DIR / "predictions" / "phase1" / f"{case.patient_id}_frame{case.frame:02d}_pred.nii.gz"
    if pred_path.is_file():
        pred = load_nifti(pred_path).data.astype("uint8")
    histories = {p: METRICS_DIR / f"phase{p}_history.json" for p in (1, 2, 3)}
    summaries = {}
    for p in (1, 2, 3):
        sp = METRICS_DIR / f"phase{p}_test_summary.json"
        if sp.is_file():
            summaries[p] = json.loads(sp.read_text(encoding="utf-8"))
    written = write_all_demo_figures(
        image=image,
        labels=labels,
        pred=pred,
        history_paths=histories,
        summaries=summaries,
    )
    print("Wrote:")
    for path in written:
        print(" ", path)
    return 0


def cmd_napari(args: argparse.Namespace) -> int:
    from acdc_seg.visualize.napari_viewer import view_nifti

    cfg = _load_yaml(CONFIGS_DIR / "demo.yaml")
    root = Path(cfg.get("data_root", args.data_root))
    if args.image:
        image_path = Path(args.image)
        label_path = Path(args.label) if args.label else None
        pred_path = Path(args.pred) if args.pred else None
    else:
        cases = discover_cases(root)
        case = cases[0]
        image_path = case.image_path
        label_path = case.label_path
        pred_path = RESULTS_DIR / "predictions" / "phase1" / f"{case.patient_id}_frame{case.frame:02d}_pred.nii.gz"
        if not pred_path.is_file():
            pred_path = None
    view_nifti(image_path, label_path, pred_path, physical_scale=not args.voxel_scale)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end pedagogical demo: synth → three phases → Dice → figures."""
    print("=== 1/5  synthetic NIfTI volumes ===")
    cmd_synth(argparse.Namespace(out=str(DATA_SYNTHETIC), n_patients=args.n_patients, n_slices=10, seed=2026))
    for phase in (1, 2, 3):
        print(f"\n=== 2-4/5  train phase {phase} ===")
        cmd_train(
            argparse.Namespace(
                phase=phase,
                config=str(_PHASE_CONFIG[phase]),
                data_root=str(DATA_SYNTHETIC),
                epochs=args.epochs,
                cpu=args.cpu,
            )
        )
    print("\n=== 5/5  evaluate + figures ===")
    cmd_evaluate(argparse.Namespace(phase="all", data_root=str(DATA_SYNTHETIC), cpu=args.cpu))
    cmd_figures(argparse.Namespace(data_root=str(DATA_SYNTHETIC)))
    print(f"\nDemo complete. Figures: {FIGURES_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACDC cardiac MRI segmentation (PyTorch + Pixi)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-stack", help="Import torch, napari, plotly and print versions")
    sub.add_parser("check-cuda", help="Print CUDA / GPU diagnostics")

    p_s = sub.add_parser("synth", help="Write synthetic ACDC-layout NIfTIs")
    p_s.add_argument("--out", default=str(DATA_SYNTHETIC))
    p_s.add_argument("--n-patients", default=10, type=int)
    p_s.add_argument("--n-slices", default=10, type=int)
    p_s.add_argument("--seed", default=2026, type=int)

    p_t = sub.add_parser("train", help="Train one phase")
    p_t.add_argument("--phase", type=int, required=True, choices=(1, 2, 3))
    p_t.add_argument("--config", default=None)
    p_t.add_argument("--data-root", default=str(DATA_SYNTHETIC))
    p_t.add_argument("--epochs", default=None, type=int)
    p_t.add_argument("--cpu", action="store_true")

    p_e = sub.add_parser("evaluate", help="Volumetric Dice on the test split")
    p_e.add_argument("--phase", default="all")
    p_e.add_argument("--data-root", default=str(DATA_SYNTHETIC))
    p_e.add_argument("--cpu", action="store_true")

    p_f = sub.add_parser("figures", help="Write Plotly HTML/PNG under results/figures")
    p_f.add_argument("--data-root", default=str(DATA_SYNTHETIC))

    p_n = sub.add_parser("napari", help="Interactive Napari viewer")
    p_n.add_argument("--image", default=None)
    p_n.add_argument("--label", default=None)
    p_n.add_argument("--pred", default=None)
    p_n.add_argument("--data-root", default=str(DATA_SYNTHETIC))
    p_n.add_argument("--voxel-scale", action="store_true", help="Ignore NIfTI spacing in the 3D view")

    p_d = sub.add_parser("demo", help="Full synthetic demo used in the README")
    p_d.add_argument("--n-patients", default=10, type=int)
    p_d.add_argument("--epochs", default=None, type=int, help="Override epochs for every phase (default: each config)")
    p_d.add_argument("--cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "check-stack": cmd_check_stack,
        "check-cuda": cmd_check_cuda,
        "synth": cmd_synth,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "figures": cmd_figures,
        "napari": cmd_napari,
        "demo": cmd_demo,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
