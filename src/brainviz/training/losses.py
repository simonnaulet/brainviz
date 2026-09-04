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
        if self.boundary_weight:
            weights = boundary_weights(target, valid)
            boundary_ce = (ce_map * weights).sum() / weights.sum().clamp_min(1)
        else:
            # Les sorties de supervision profonde n'utilisent pas BoundaryCE.
            # Éviter de construire leurs cartes de frontières inutilement.
            boundary_ce = logits.new_zeros(())
        total = self.dice_weight * dice + self.ce_weight * ce + self.boundary_weight * boundary_ce
        return {
            "loss": total,
            "dice_loss": dice.detach(),
            "ce": ce.detach(),
            "boundary_ce": boundary_ce.detach(),
            "mean_soft_dice": dice_per_class[1:].mean().detach(),
        }


class DeepSupervisionLoss(nn.Module):
    """Applique la loss principale à H et Dice+CE aux sorties H/2 et H/4."""

    def __init__(
        self,
        main_loss: CompositeSegmentationLoss,
        auxiliary_weights: tuple[float, ...] = (0.5, 0.25),
    ) -> None:
        super().__init__()
        if not auxiliary_weights or any(weight < 0 for weight in auxiliary_weights):
            raise ValueError("les poids de supervision profonde doivent être positifs")
        non_boundary_total = main_loss.dice_weight + main_loss.ce_weight
        if non_boundary_total <= 0:
            raise ValueError("Dice+CE doit avoir un poids non nul pour les sorties auxiliaires")
        self.main_loss = main_loss
        self.auxiliary_loss = CompositeSegmentationLoss(
            dice_weight=main_loss.dice_weight / non_boundary_total,
            ce_weight=main_loss.ce_weight / non_boundary_total,
            boundary_weight=0.0,
        )
        raw_weights = (1.0, *auxiliary_weights)
        weight_sum = sum(raw_weights)
        self.output_weights = tuple(weight / weight_sum for weight in raw_weights)

    @staticmethod
    def _resize_target(target: Tensor, valid: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        resized_target = F.interpolate(target[:, None].float(), size=size, mode="nearest")[:, 0].long()
        resized_valid = F.interpolate(valid[:, None].float(), size=size, mode="nearest")[:, 0].bool()
        return resized_target, resized_valid

    def forward(
        self,
        logits: Tensor | tuple[Tensor, ...],
        target: Tensor,
        valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(logits, Tensor):
            return self.main_loss(logits, target, valid)
        if len(logits) != len(self.output_weights):
            raise ValueError(
                f"{len(self.output_weights)} sorties attendues pour la supervision profonde, obtenu {len(logits)}"
            )
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        main = self.main_loss(logits[0], target, valid)
        total = self.output_weights[0] * main["loss"]
        result = dict(main)
        result["main_loss"] = main["loss"].detach()
        for index, (auxiliary_logits, weight) in enumerate(
            zip(logits[1:], self.output_weights[1:], strict=True), start=1
        ):
            auxiliary_target, auxiliary_valid = self._resize_target(
                target, valid, tuple(auxiliary_logits.shape[-2:])
            )
            auxiliary = self.auxiliary_loss(auxiliary_logits, auxiliary_target, auxiliary_valid)
            total = total + weight * auxiliary["loss"]
            result[f"aux_{index}_loss"] = auxiliary["loss"].detach()
            result[f"aux_{index}_soft_dice"] = auxiliary["mean_soft_dice"]
        result["loss"] = total
        return result
