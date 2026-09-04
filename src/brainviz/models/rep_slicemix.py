"""TriPlane Rep-SliceMix-Net et fusion structurelle des branches RepDW."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Compte les paramètres, en excluant éventuellement ceux qui sont gelés."""
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


class _RepDW(nn.Module):
    """Convolution depthwise multi-branche fusionnable, en 2D ou 3D."""

    def __init__(
        self,
        channels: int,
        kernel_size: Sequence[int],
        *,
        multi_branch: bool = True,
        auxiliary_kernel_sizes: Sequence[Sequence[int]] = (),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = tuple(kernel_size)
        self.dim = len(self.kernel_size)
        self.multi_branch = multi_branch
        if self.dim not in (2, 3) or any(k % 2 == 0 for k in self.kernel_size):
            raise ValueError("RepDW attend un kernel 2D/3D de dimensions impaires")
        auxiliary_kernel_sizes = tuple(tuple(kernel) for kernel in auxiliary_kernel_sizes)
        if any(
            len(kernel) != self.dim
            or any(k % 2 == 0 or k > main for k, main in zip(kernel, self.kernel_size, strict=True))
            for kernel in auxiliary_kernel_sizes
        ):
            raise ValueError("les kernels auxiliaires doivent être impairs et inclus dans le kernel principal")
        conv_cls = nn.Conv2d if self.dim == 2 else nn.Conv3d
        bn_cls = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d
        padding = tuple(k // 2 for k in self.kernel_size)
        one = (1,) * self.dim
        self.main_conv = conv_cls(channels, channels, self.kernel_size, padding=padding, groups=channels, bias=False)
        self.main_bn = bn_cls(channels)
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_bns = nn.ModuleList()
        if multi_branch:
            for kernel in auxiliary_kernel_sizes:
                auxiliary_padding = tuple(k // 2 for k in kernel)
                self.auxiliary_convs.append(
                    conv_cls(channels, channels, kernel, padding=auxiliary_padding, groups=channels, bias=False)
                )
                self.auxiliary_bns.append(bn_cls(channels))
            self.point_conv = conv_cls(channels, channels, one, groups=channels, bias=False)
            self.point_bn = bn_cls(channels)
            self.identity_bn = bn_cls(channels)
        self.reparam_conv: nn.Module | None = None

    @property
    def is_reparameterized(self) -> bool:
        return self.reparam_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.reparam_conv is not None:
            return self.reparam_conv(x)
        y = self.main_bn(self.main_conv(x))
        if self.multi_branch:
            for conv, bn in zip(self.auxiliary_convs, self.auxiliary_bns, strict=True):
                y = y + bn(conv(x))
            y = y + self.point_bn(self.point_conv(x)) + self.identity_bn(x)
        return y

    @staticmethod
    def _fuse_conv_bn(conv: nn.Module, bn: nn.modules.batchnorm._BatchNorm) -> tuple[Tensor, Tensor]:
        kernel = conv.weight
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_kernel = kernel * scale.reshape((-1, 1) + (1,) * (kernel.ndim - 2))
        fused_bias = bn.bias - bn.running_mean * scale
        return fused_kernel, fused_bias

    def _fuse_identity_bn(self, bn: nn.modules.batchnorm._BatchNorm) -> tuple[Tensor, Tensor]:
        kernel = torch.zeros(
            (self.channels, 1, *self.kernel_size),
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        center = (slice(None), 0, *(k // 2 for k in self.kernel_size))
        kernel[center] = 1
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        kernel = kernel * scale.reshape((-1, 1) + (1,) * self.dim)
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    def equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Retourne le kernel et le biais de la convolution déployée."""
        if self.reparam_conv is not None:
            return self.reparam_conv.weight, self.reparam_conv.bias
        kernel, bias = self._fuse_conv_bn(self.main_conv, self.main_bn)
        if not self.multi_branch:
            return kernel, bias
        for auxiliary_conv, auxiliary_bn in zip(self.auxiliary_convs, self.auxiliary_bns, strict=True):
            auxiliary_kernel, auxiliary_bias = self._fuse_conv_bn(auxiliary_conv, auxiliary_bn)
            padded_auxiliary = torch.zeros_like(kernel)
            slices = zip(
                self.kernel_size,
                auxiliary_kernel.shape[2:],
                strict=True,
            )
            destination = (
                slice(None),
                slice(None),
                *(slice((main - small) // 2, (main + small) // 2) for main, small in slices),
            )
            padded_auxiliary[destination] = auxiliary_kernel
            kernel = kernel + padded_auxiliary
            bias = bias + auxiliary_bias
        point_kernel, point_bias = self._fuse_conv_bn(self.point_conv, self.point_bn)
        padded_point = torch.zeros_like(kernel)
        center = (slice(None), slice(None), *(k // 2 for k in self.kernel_size))
        padded_point[center] = point_kernel[(slice(None), slice(None), *((0,) * self.dim))]
        identity_kernel, identity_bias = self._fuse_identity_bn(self.identity_bn)
        return kernel + padded_point + identity_kernel, bias + point_bias + identity_bias

    def reparameterize(self) -> "_RepDW":
        """Remplace les branches par une unique convolution depthwise avec biais."""
        if self.reparam_conv is not None:
            return self
        kernel, bias = self.equivalent_kernel_bias()
        conv_cls = nn.Conv2d if self.dim == 2 else nn.Conv3d
        conv = conv_cls(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=tuple(k // 2 for k in self.kernel_size),
            groups=self.channels,
            bias=True,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        conv.weight.data.copy_(kernel)
        conv.bias.data.copy_(bias)
        self.reparam_conv = conv
        for name in (
            "main_conv",
            "main_bn",
            "auxiliary_convs",
            "auxiliary_bns",
            "point_conv",
            "point_bn",
            "identity_bn",
        ):
            if hasattr(self, name):
                delattr(self, name)
        return self


class RepDW3d(_RepDW):
    def __init__(self, channels: int, kernel_size: Sequence[int], *, multi_branch: bool = True) -> None:
        super().__init__(channels, kernel_size, multi_branch=multi_branch)


class RepDW2d(_RepDW):
    def __init__(
        self,
        channels: int,
        kernel_size: Sequence[int] = (3, 3),
        *,
        multi_branch: bool = True,
        auxiliary_kernel_sizes: Sequence[Sequence[int]] = (),
    ) -> None:
        super().__init__(
            channels,
            kernel_size,
            multi_branch=multi_branch,
            auxiliary_kernel_sizes=auxiliary_kernel_sizes,
        )


class RepSliceMix(nn.Module):
    def __init__(self, channels: int, slices: int, *, mlp_ratio: int = 2, multi_branch: bool = True) -> None:
        super().__init__()
        self.spatial = RepDW3d(channels, (1, 3, 3), multi_branch=multi_branch)
        self.slice = RepDW3d(channels, (3, 1, 1), multi_branch=multi_branch) if slices > 1 else None
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )
        self.norm = nn.BatchNorm3d(channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, x: Tensor) -> Tensor:
        y = F.gelu(self.spatial(x))
        if self.slice is not None:
            y = F.gelu(self.slice(y))
        y = self.norm(self.mlp(y))
        return x + self.layer_scale.view(1, -1, 1, 1, 1) * y


class Block2D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        mlp_ratio: int = 2,
        multi_branch: bool = True,
        spatial_kernel_size: int = 3,
        auxiliary_kernel_sizes: Sequence[int] = (),
    ) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.spatial = RepDW2d(
            channels,
            (spatial_kernel_size, spatial_kernel_size),
            multi_branch=multi_branch,
            auxiliary_kernel_sizes=tuple((kernel, kernel) for kernel in auxiliary_kernel_sizes),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, x: Tensor) -> Tensor:
        y = F.gelu(self.spatial(x))
        y = self.norm(self.mlp(y))
        return x + self.layer_scale.view(1, -1, 1, 1) * y


class PlaneFiLM(nn.Module):
    def __init__(self, channels: int, enabled: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.enabled = enabled
        if enabled:
            self.embedding = nn.Embedding(3, 2 * channels)
            with torch.no_grad():
                self.embedding.weight[:, :channels].fill_(1)
                self.embedding.weight[:, channels:].zero_()

    def forward(self, x: Tensor, plane: Tensor) -> Tensor:
        if not self.enabled:
            return x
        gamma, beta = self.embedding(plane).chunk(2, dim=-1)
        return x * gamma[:, :, None, None, None] + beta[:, :, None, None, None]


class SlicePool(nn.Module):
    def __init__(self, channels: int, slices: int) -> None:
        super().__init__()
        self.slices = slices
        self.weights = nn.Parameter(torch.zeros(3, channels, slices)) if slices > 1 else None

    def forward(self, x: Tensor, plane: Tensor) -> Tensor:
        if self.slices == 1:
            return x.squeeze(2)
        attention = self.weights[plane].softmax(dim=-1)
        return (x * attention[:, :, :, None, None]).sum(dim=2)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, multi_branch: bool = True) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.block = Block2D(out_channels, multi_branch=multi_branch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        return self.block(self.fuse(torch.cat((x, skip), dim=1)))


class TriPlaneRepSliceMixNet(nn.Module):
    """Réseau principal. L'entrée doit avoir la forme ``[B,C,5,H,W]``."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        widths: Sequence[int] = (24, 48, 96, 192),
        depths: Sequence[int] = (2, 2, 2, 1),
        *,
        mlp_ratio: int = 2,
        multi_branch: bool = True,
        film: bool = True,
        down3_mode: str = "dense",
        bottleneck_kernel_size: int = 3,
        deep_supervision: bool = False,
        input_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if len(widths) != 4:
            raise ValueError("widths doit contenir quatre valeurs")
        if len(depths) != 4 or any(int(depth) < 1 for depth in depths):
            raise ValueError("depths doit contenir quatre entiers positifs")
        if down3_mode not in {"dense", "separable"}:
            raise ValueError("down3_mode doit valoir 'dense' ou 'separable'")
        if bottleneck_kernel_size < 3 or bottleneck_kernel_size % 2 == 0:
            raise ValueError("bottleneck_kernel_size doit être impair et >= 3")
        c1, c2, c3, c4 = widths
        self.in_channels = in_channels
        self.input_indices = tuple(range(in_channels)) if input_indices is None else tuple(input_indices)
        if len(self.input_indices) != in_channels:
            raise ValueError("input_indices doit contenir in_channels éléments")
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.deep_supervision = deep_supervision
        self.center_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(c1),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            *(
                RepSliceMix(c1, 5, mlp_ratio=mlp_ratio, multi_branch=multi_branch)
                for _ in range(depths[0])
            )
        )
        self.film1, self.pool1 = PlaneFiLM(c1, film), SlicePool(c1, 5)
        self.down1 = nn.Sequential(
            nn.Conv3d(c1, c2, (3, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(c2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            *(
                RepSliceMix(c2, 3, mlp_ratio=mlp_ratio, multi_branch=multi_branch)
                for _ in range(depths[1])
            )
        )
        self.film2, self.pool2 = PlaneFiLM(c2, film), SlicePool(c2, 3)
        self.down2 = nn.Sequential(
            nn.Conv3d(c2, c3, (3, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(c3),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            *(
                RepSliceMix(c3, 1, mlp_ratio=mlp_ratio, multi_branch=multi_branch)
                for _ in range(depths[2])
            )
        )
        self.film3, self.pool3 = PlaneFiLM(c3, film), SlicePool(c3, 1)
        if down3_mode == "dense":
            down_conv: nn.Module = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False)
        else:
            down_conv = nn.Sequential(
                nn.Conv2d(c3, c3, 3, stride=2, padding=1, groups=c3, bias=False),
                nn.Conv2d(c3, c4, 1, bias=False),
            )
        self.down3 = nn.Sequential(down_conv, nn.BatchNorm2d(c4), nn.GELU())
        auxiliary_kernels = (3,) if bottleneck_kernel_size > 3 else ()
        self.bottle = nn.Sequential(
            *(
                Block2D(
                    c4,
                    mlp_ratio=mlp_ratio,
                    multi_branch=multi_branch,
                    spatial_kernel_size=bottleneck_kernel_size,
                    auxiliary_kernel_sizes=auxiliary_kernels,
                )
                for _ in range(depths[3])
            )
        )
        self.decoder = nn.ModuleList(
            [
                DecoderStage(c4, c3, multi_branch=multi_branch),
                DecoderStage(c3, c2, multi_branch=multi_branch),
                DecoderStage(c2, c1, multi_branch=multi_branch),
                DecoderStage(c1, 16, multi_branch=multi_branch),
            ]
        )
        self.head = nn.Conv2d(16, num_classes, 1)
        self.auxiliary_heads = nn.ModuleList(
            (nn.Conv2d(c1, num_classes, 1), nn.Conv2d(c2, num_classes, 1)) if deep_supervision else ()
        )

    def forward(self, x: Tensor, plane: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 5 or x.shape[2] != 5:
            raise ValueError(f"entrée [B,C,5,H,W] attendue, obtenu {tuple(x.shape)}")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("H et W doivent être divisibles par 16")
        if x.shape[1] != self.in_channels:
            x = x[:, self.input_indices]
        skip0 = self.center_branch(x[:, :, 2])
        x = self.stem(x)
        u1 = self.film1(self.stage1(x), plane)
        skip1 = self.pool1(u1, plane)
        x = self.down1(u1)
        u2 = self.film2(self.stage2(x), plane)
        skip2 = self.pool2(u2, plane)
        x = self.down2(u2)
        u3 = self.film3(self.stage3(x), plane)
        skip3 = self.pool3(u3, plane)
        x = self.bottle(self.down3(skip3))
        auxiliary_features: list[Tensor] = []
        for index, (decoder, skip) in enumerate(zip(self.decoder, (skip3, skip2, skip1, skip0), strict=True)):
            x = decoder(x, skip)
            if self.training and self.deep_supervision and index in (1, 2):
                auxiliary_features.append(x)
        logits = self.head(x)
        if self.training and self.deep_supervision:
            # Ordre décroissant de résolution : sortie principale, H/2, H/4.
            h4, h2 = auxiliary_features
            return logits, self.auxiliary_heads[0](h2), self.auxiliary_heads[1](h4)
        return logits

    def reparameterize(self, *, inplace: bool = True) -> "TriPlaneRepSliceMixNet":
        """Fusionne tous les RepDW. Le modèle doit être en mode évaluation."""
        if self.training:
            raise RuntimeError("appeler eval() avant reparameterize()")
        model = self if inplace else copy.deepcopy(self)
        for module in model.modules():
            if isinstance(module, _RepDW):
                module.reparameterize()
        if model.deep_supervision:
            del model.auxiliary_heads
            model.deep_supervision = False
        return model
