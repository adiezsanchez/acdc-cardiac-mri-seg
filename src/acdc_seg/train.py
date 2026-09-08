"""Shared training loop for phases 1–3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from acdc_seg.constants import CHECKPOINTS_DIR, METRICS_DIR, NUM_CLASSES
from acdc_seg.device import describe_device, get_device
from acdc_seg.losses import DiceCELoss
from acdc_seg.metrics import volumetric_dice
from acdc_seg.models import LightUNet3D, UNet2D


def seed_everything(seed: int = 2026) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(phase: int, *, in_channels: int = 1, base: int | None = None) -> torch.nn.Module:
    if phase in (1, 2):
        return UNet2D(in_channels=in_channels, num_classes=NUM_CLASSES, base=base or 16)
    if phase == 3:
        return LightUNet3D(in_channels=1, num_classes=NUM_CLASSES, base=base or 8)
    raise ValueError(f"Unknown phase {phase}")


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


@torch.no_grad()
def _batch_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = _logits_to_labels(logits).cpu().numpy()
    tgt = target.cpu().numpy()
    scores = []
    for p, t in zip(pred, tgt, strict=False):
        scores.append(volumetric_dice(p, t)["mean"])
    return float(np.mean(scores)) if scores else 0.0


def run_training(
    *,
    phase: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    in_channels: int = 1,
    base: int | None = None,
    out_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    device = device or get_device()
    seed_everything()
    out_dir = Path(out_dir or CHECKPOINTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(phase, in_channels=in_channels, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceCELoss()
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    ckpt_path = out_dir / f"phase{phase}_best.pt"

    print(f"[phase {phase}] device={describe_device(device)}")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"phase{phase} train {epoch}/{epochs}", leave=False):
            batch = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch["image"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                logits = model(batch["image"])
                val_losses.append(float(criterion(logits, batch["label"]).item()))
                val_dices.append(_batch_dice(logits, batch["label"]))
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_dice={val_dice:.3f}")
        if val_dice >= best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "phase": phase,
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "base": base,
                    "val_dice": val_dice,
                    "epoch": epoch,
                },
                ckpt_path,
            )

    history_path = METRICS_DIR / f"phase{phase}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "checkpoint": str(ckpt_path), "best_val_dice": best_dice, "device": str(device)}


def load_checkpoint(path: str | Path, phase: int, device: torch.device | None = None) -> torch.nn.Module:
    device = device or get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    in_channels = int(payload.get("in_channels", 1))
    base = payload.get("base")
    model = build_model(phase, in_channels=in_channels, base=base)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model
