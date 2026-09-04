"""Métriques volumiques iSeg."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def binary_metrics(prediction: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict[str, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return {"dice": 1.0, "hd95": 0.0, "asd": 0.0}
    if not prediction.any() or not target.any():
        return {"dice": 0.0, "hd95": float("inf"), "asd": float("inf")}
    dice = 2 * np.logical_and(prediction, target).sum() / (prediction.sum() + target.sum())
    pred_surface, target_surface = _surface(prediction), _surface(target)
    to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    distances = np.concatenate((to_target, to_prediction))
    asd = 0.5 * (to_target.mean() + to_prediction.mean())
    return {"dice": float(dice), "hd95": float(np.percentile(distances, 95)), "asd": float(asd)}


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict[str, float]:
    names = ("csf", "gm", "wm")
    result: dict[str, float] = {}
    for label, name in enumerate(names, start=1):
        for metric, value in binary_metrics(prediction == label, target == label, spacing).items():
            result[f"{name}_{metric}"] = value
    result["mean_dice"] = float(np.mean([result[f"{name}_dice"] for name in names]))
    return result
