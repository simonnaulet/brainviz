"""Loss, métriques et boucle d'entraînement."""

from .losses import CompositeSegmentationLoss, DeepSupervisionLoss

__all__ = ["CompositeSegmentationLoss", "DeepSupervisionLoss"]
