"""Baselines compactes 2D et 3D (2+1)D."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _DenseBlock2D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )


class TinyUNet2D(nn.Module):
    """U-Net dense d'environ 0,5 M paramètres pour les expériences A et B."""

    def __init__(self, input_mode: str = "central", num_classes: int = 4, widths: Sequence[int] = (8, 16, 32, 64, 128)) -> None:
        super().__init__()
        if input_mode not in {"central", "stack"}:
            raise ValueError("input_mode doit valoir central ou stack")
        self.input_mode = input_mode
        self.in_channels = 2 if input_mode == "central" else 10
        self.encoders = nn.ModuleList()
        previous = self.in_channels
        for width in widths:
            self.encoders.append(_DenseBlock2D(previous, width))
            previous = width
        self.projects = nn.ModuleList(nn.Conv2d(widths[i], widths[i - 1], 1) for i in range(4, 0, -1))
        self.decoders = nn.ModuleList(_DenseBlock2D(2 * widths[i - 1], widths[i - 1]) for i in range(4, 0, -1))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: Tensor, plane: Tensor | None = None) -> Tensor:
        del plane
        x = x[:, :2, 2] if self.input_mode == "central" else x[:, :2].flatten(1, 2)
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.encoders) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for project, decoder, skip in zip(self.projects, self.decoders, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((project(x), skip), dim=1))
        return self.head(x)


class _Block21D(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1), groups=channels, bias=False),
            nn.BatchNorm3d(channels), nn.GELU(),
        )
        self.depth = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.BatchNorm3d(channels), nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels * mlp_ratio, 1), nn.GELU(),
            nn.Conv3d(channels * mlp_ratio, channels, 1), nn.BatchNorm3d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.depth(self.spatial(x)))


class UNet3D21D(nn.Module):
    """Concurrent 3D compact à convolutions depthwise (2+1)D."""

    def __init__(self, in_channels: int = 6, num_classes: int = 4, widths: Sequence[int] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stems = nn.ModuleList()
        self.blocks = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.stems.append(nn.Sequential(nn.Conv3d(previous, width, 1, bias=False), nn.BatchNorm3d(width), nn.GELU()))
            self.blocks.append(_Block21D(width))
            previous = width
        self.projects = nn.ModuleList(nn.Conv3d(widths[i], widths[i - 1], 1) for i in range(3, 0, -1))
        self.decoders = nn.ModuleList(
            nn.Sequential(nn.Conv3d(2 * widths[i - 1], widths[i - 1], 1), nn.GELU(), _Block21D(widths[i - 1]))
            for i in range(3, 0, -1)
        )
        self.head = nn.Conv3d(widths[0], num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for index, (stem, block) in enumerate(zip(self.stems, self.blocks, strict=True)):
            x = block(stem(x))
            if index < len(self.blocks) - 1:
                skips.append(x)
                x = F.avg_pool3d(x, 2)
        for project, decoder, skip in zip(self.projects, self.decoders, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((project(x), skip), dim=1))
        return self.head(x)
