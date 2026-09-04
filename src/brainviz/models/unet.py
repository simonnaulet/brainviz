"""U-Net compact pour la segmentation iSeg-2017, optimisé pour le ratio Dice/paramètres.

Les convolutions standard sont remplacées par des convolutions depthwise-separable
(depthwise 3x3 + pointwise 1x1), qui réduisent le nombre de paramètres d'un facteur
proche de k^2 (k=3, donc ~9x avant l'ajout de la pointwise) par rapport à une conv
classique à largeur égale, pour un coût en Dice généralement faible sur ce genre de
tâche à peu de données.
"""

import torch
import torch.nn as nn


class SeparableConvBlock(nn.Module):
    """Deux convolutions depthwise-separable + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            *self._separable_conv(in_channels, out_channels),
            *self._separable_conv(out_channels, out_channels),
        )

    @staticmethod
    def _separable_conv(in_channels, out_channels):
        return [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def forward(self, x):
        return self.block(x)


class CompactUNet(nn.Module):
    """U-Net compact à convolutions depthwise-separable.

    Args:
        in_channels (int): nombre de canaux d'entrée (1 pour T1 ou T2 seul, 3 pour T1+T2+ratio).
        num_classes (int): nombre de classes de segmentation en sortie.
        base_channels (int): nombre de canaux du premier niveau ; double à chaque niveau
            d'encodeur. 16 par défaut pour rester très frugal en paramètres.
        depth (int): nombre de niveaux de l'encodeur/décodeur (hors bottleneck). 3 par défaut.

    Le forward attend une entrée (B, in_channels, H, W) avec H et W divisibles par 2**depth,
    et renvoie des logits (B, num_classes, H, W).
    """

    def __init__(self, in_channels=1, num_classes=4, base_channels=16, depth=3):
        super().__init__()
        self.depth = depth

        encoder_channels = [base_channels * (2**i) for i in range(depth)]
        bottleneck_channels = base_channels * (2**depth)

        self.pool = nn.MaxPool2d(2)

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for channels in encoder_channels:
            self.encoders.append(SeparableConvBlock(prev_channels, channels))
            prev_channels = channels

        self.bottleneck = SeparableConvBlock(prev_channels, bottleneck_channels)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_channels = bottleneck_channels
        for channels in reversed(encoder_channels):
            self.upsamples.append(nn.ConvTranspose2d(prev_channels, channels, kernel_size=2, stride=2))
            self.decoders.append(SeparableConvBlock(channels * 2, channels))
            prev_channels = channels

        self.out_conv = nn.Conv2d(prev_channels, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.out_conv(x)

    def num_parameters(self):
        """int: nombre total de paramètres entraînables du modèle."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)