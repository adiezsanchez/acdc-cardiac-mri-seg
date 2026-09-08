"""Dice + cross-entropy loss used by all three phases."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from acdc_seg.constants import NUM_CLASSES


class DiceCELoss(nn.Module):
    """Soft Dice on one-hot classes plus CE. Background is included in CE only.

    Dice is averaged over RV / MYO / LV so a huge background class cannot
    dominate the gradient — the usual trick in cardiac segmentation.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        dice = 0.0
        n_fg = self.num_classes - 1
        for cls in range(1, self.num_classes):
            p = probs[:, cls]
            g = (target == cls).float()
            intersection = (p * g).sum()
            denom = p.sum() + g.sum()
            dice = dice + (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice / max(n_fg, 1)
        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice_loss


def one_hot(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3) if labels.ndim == 4 else F.one_hot(
        labels.long(), num_classes=num_classes
    ).permute(0, 3, 1, 2)
