from __future__ import annotations

import torch

from acdc_seg.models import LightUNet3D, UNet2D


def test_unet2d_shapes():
    net = UNet2D(in_channels=1, num_classes=4, base=8)
    x = torch.zeros(2, 1, 64, 64)
    y = net(x)
    assert y.shape == (2, 4, 64, 64)


def test_unet2d_neighbor_channels():
    net = UNet2D(in_channels=3, num_classes=4, base=8)
    x = torch.zeros(1, 3, 64, 64)
    y = net(x)
    assert y.shape == (1, 4, 64, 64)


def test_unet3d_shapes():
    net = LightUNet3D(in_channels=1, num_classes=4, base=4)
    x = torch.zeros(1, 1, 10, 64, 64)
    y = net(x)
    assert y.shape[0] == 1 and y.shape[1] == 4
    assert y.shape[2] == 10
    assert y.shape[3] == 64 and y.shape[4] == 64
