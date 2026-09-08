"""Phase 3: a *light* 3D U-Net that respects ACDC anisotropy.

Why not a textbook 3×3×3 U-Net?
--------------------------------
ACDC voxels are ~1.4 × 1.4 × 5–10 mm. A 3×3×3 kernel therefore spans
~4 mm in-plane but 15–30 mm through-plane — several anatomical slices.
With only 8–12 short-axis slices per volume, isotropic 2×2×2 pooling
also collapses depth immediately.

This module therefore:

1. Uses ``(1, 3, 3)`` convolutions at the first level (in-plane).
2. Downsamples **in-plane first** with stride ``(1, 2, 2)``.
3. Only then uses a shallow 3×3×3 bottleneck.

The network is intentionally small (base=8, two downsampling levels) so it
fits a pedagogical GPU/CPU demo. It is a 3D *inductive bias*, not a SOTA
nnU-Net.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int, int] = (3, 3, 3)) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 8) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        # Anisotropic first block: do not mix distant slices yet.
        self.enc1 = ConvBlock3d(in_channels, c1, kernel=(1, 3, 3))
        self.down1 = nn.Conv3d(c1, c2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = ConvBlock3d(c2, c2, kernel=(3, 3, 3))
        self.down2 = nn.Conv3d(c2, c3, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bottleneck = ConvBlock3d(c3, c3, kernel=(3, 3, 3))
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = ConvBlock3d(c2 * 2, c2, kernel=(3, 3, 3))
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3d(c1 * 2, c1, kernel=(1, 3, 3))
        self.head = nn.Conv3d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self._align_and_decode(self.up2(b), e2, self.dec2)
        d1 = self._align_and_decode(self.up1(d2), e1, self.dec1)
        return self.head(d1)

    @staticmethod
    def _align_and_decode(up: torch.Tensor, skip: torch.Tensor, block: nn.Module) -> torch.Tensor:
        # Odd depths / sizes can be one voxel off after stride-2 transpose convs.
        if up.shape[2:] != skip.shape[2:]:
            up = _center_crop_or_pad(up, skip.shape[2:])
        return block(torch.cat([up, skip], dim=1))


def _center_crop_or_pad(tensor: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
    _, _, d, h, w = tensor.shape
    td, th, tw = spatial
    # Pad first if smaller.
    pad_d = max(td - d, 0)
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    if pad_d or pad_h or pad_w:
        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_d // 2,
                pad_d - pad_d // 2,
            ),
        )
        _, _, d, h, w = tensor.shape
    sd = max((d - td) // 2, 0)
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    return tensor[:, :, sd : sd + td, sh : sh + th, sw : sw + tw]
