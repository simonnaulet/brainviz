"""Fonction objectif du modèle Rep-SliceMix."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def boundary_weights(target: Tensor, valid: Tensor, radius: int = 2) -> Tensor:
    """Poids 3 près de GM/WM, 2 près des autres interfaces, 1 ailleurs."""
    one_hot = F.one_hot(target.clamp(0, 3), num_classes=4).movedim(-1, 1).float()
    one_hot = one_hot * valid[:, None]
    pool = F.max_pool2d if target.ndim == 3 else F.max_pool3d
    neighborhoods = pool(one_hot, kernel_size=3, stride=1, padding=1)
    interface = neighborhoods.sum(dim=1) >= 2
    gm_wm = (neighborhoods[:, 2] > 0) & (neighborhoods[:, 3] > 0)
    # Le voisinage 3x3 initial couvre déjà une distance d'un pixel.
    expansion = max(radius - 1, 0)
    kernel = 2 * expansion + 1
    interface = pool(interface[:, None].float(), kernel, stride=1, padding=expansion)[:, 0].bool()
    gm_wm = pool(gm_wm[:, None].float(), kernel, stride=1, padding=expansion)[:, 0].bool()
    weights = torch.ones_like(target, dtype=torch.float32)
    weights[interface] = 2.0
    weights[gm_wm] = 3.0
    return weights * valid


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, ce_weight: float = 0.3, boundary_weight: float = 0.1) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits: Tensor, target: Tensor, valid: Tensor | None = None) -> dict[str, Tensor]:
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        valid_f = valid.float()
        probabilities = logits.softmax(dim=1)
        target_one_hot = F.one_hot(target.clamp(0, 3), num_classes=4).movedim(-1, 1).float()
        mask = valid_f[:, None]
        reduce_dims = (0, *range(2, logits.ndim))
        intersection = (probabilities * target_one_hot * mask).sum(dim=reduce_dims)
        denominator = ((probabilities + target_one_hot) * mask).sum(dim=reduce_dims)
        dice_per_class = (2 * intersection + 1e-5) / (denominator + 1e-5)
        dice = 1 - dice_per_class[1:].mean()

        ce_map = F.cross_entropy(logits, target, reduction="none")
        valid_count = valid_f.sum().clamp_min(1)
        ce = (ce_map * valid_f).sum() / valid_count
        weights = boundary_weights(target, valid)
        boundary_ce = (ce_map * weights).sum() / weights.sum().clamp_min(1)
        total = self.dice_weight * dice + self.ce_weight * ce + self.boundary_weight * boundary_ce
        return {
            "loss": total,
            "dice_loss": dice.detach(),
            "ce": ce.detach(),
            "boundary_ce": boundary_ce.detach(),
            "mean_soft_dice": dice_per_class[1:].mean().detach(),
        }
