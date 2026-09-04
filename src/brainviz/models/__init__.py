"""Architectures de segmentation du projet."""

from .rep_slicemix import TriPlaneRepSliceMixNet, count_parameters
from .baselines import TinyUNet2D, UNet3D21D

__all__ = ["TinyUNet2D", "TriPlaneRepSliceMixNet", "UNet3D21D", "count_parameters"]
