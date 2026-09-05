"""Inférence axiale ou tri-plan et reconstruction volumique."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from brainviz.data.triplane import PLANE_NAMES, from_plane, to_plane


def _prediction_support(item: dict) -> np.ndarray:
    """Tolère un voxel de bord sans autoriser le padding structurel."""
    support = ndimage.binary_dilation(
        item["brain_mask"].astype(bool), structure=np.ones((3, 3, 3), dtype=bool), iterations=5
    )
    return support & item["fov_valid"].astype(bool)


@torch.inference_mode()
def predict_preprocessed(
    model: nn.Module,
    subject_path: str | Path,
    *,
    planes: Iterable[int] = (0, 1, 2),
    slice_spacings: Iterable[int] = (1,),
    batch_size: int = 16,
    device: str | torch.device = "cuda",
    amp: bool = True,
    amp_dtype: str = "float16",
) -> tuple[np.ndarray, dict]:
    """Retourne les probabilités `[4,X,Y,Z]` moyennées sur les vues demandées.

    Une vue est un couple ``(plan, espacement inter-coupes)``. Le comportement
    historique correspond à ``slice_spacings=(1,)``.
    """
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype doit valoir 'float16' ou 'bfloat16'")
    torch_amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    with np.load(subject_path, allow_pickle=False) as archive:
        item = {key: archive[key] for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    channels = np.stack(
        [item[name] for name in ("t1", "t2", "coord_x", "coord_y", "coord_z", "brain_mask")], axis=0
    )
    device = torch.device(device)
    model.eval()
    planes = tuple(int(plane) for plane in planes)
    slice_spacings = tuple(int(spacing) for spacing in slice_spacings)
    if not planes or any(plane not in range(3) for plane in planes):
        raise ValueError("planes doit contenir au moins un plan parmi 0, 1, 2")
    if not slice_spacings or any(spacing < 1 for spacing in slice_spacings):
        raise ValueError("slice_spacings doit contenir des entiers strictement positifs")
    view_probabilities = []
    for plane in planes:
        oriented = to_plane(channels, plane, channel_first=True)
        for spacing in slice_spacings:
            results = []
            offsets = spacing * np.arange(-2, 3)
            for start in range(0, oriented.shape[1], batch_size):
                centers = np.arange(start, min(start + batch_size, oriented.shape[1]))
                stacks = [oriented[:, np.clip(center + offsets, 0, oriented.shape[1] - 1)] for center in centers]
                x = torch.from_numpy(np.ascontiguousarray(np.stack(stacks))).float().to(device)
                p = torch.full((len(stacks),), plane, dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=torch_amp_dtype, enabled=amp and device.type == "cuda"):
                    probabilities = model(x, p).softmax(dim=1)
                results.append(probabilities.float().cpu().numpy())
            # [S,C,H,W] -> [C,S,H,W] -> [C,X,Y,Z]
            probability = np.concatenate(results, axis=0).transpose(1, 0, 2, 3)
            view_probabilities.append(from_plane(probability, plane, channel_first=True))
    probabilities = np.mean(view_probabilities, axis=0)
    outside = ~_prediction_support(item)
    probabilities[:, outside] = 0
    probabilities[0, outside] = 1
    return probabilities, metadata


def parse_planes(value: str) -> tuple[int, ...]:
    if value == "all":
        return (0, 1, 2)
    names = [part.strip() for part in value.split(",")]
    unknown = set(names) - set(PLANE_NAMES)
    if unknown:
        raise ValueError(f"plans inconnus: {sorted(unknown)}")
    return tuple(PLANE_NAMES.index(name) for name in names)


def _sliding_starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    step = max(1, round(patch * (1 - overlap)))
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.inference_mode()
def predict_preprocessed_3d(
    model: nn.Module,
    subject_path: str | Path,
    *,
    patch_size: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    device: str | torch.device = "cuda",
    amp: bool = True,
    amp_dtype: str = "float16",
) -> tuple[np.ndarray, dict]:
    """Sliding-window du concurrent 3D, sortie `[4,X,Y,Z]`."""
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype doit valoir 'float16' ou 'bfloat16'")
    torch_amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    with np.load(subject_path, allow_pickle=False) as archive:
        item = {key: archive[key] for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    xyz = np.stack([item[name] for name in ("t1", "t2", "coord_x", "coord_y", "coord_z", "brain_mask")])
    volume = torch.from_numpy(np.ascontiguousarray(xyz.transpose(0, 3, 1, 2))).float()
    original_shape = volume.shape[-3:]
    patch_shape = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
    target_shape = tuple(max(size, patch) for size, patch in zip(original_shape, patch_shape, strict=True))
    total_pad = tuple(target - original for target, original in zip(target_shape, original_shape, strict=True))
    before = tuple(value // 2 for value in total_pad)
    after = tuple(value - start for value, start in zip(total_pad, before, strict=True))
    pad = (before[2], after[2], before[1], after[1], before[0], after[0])
    volume = F.pad(volume, pad)
    coordinates = [torch.linspace(-1, 1, size) for size in patch_shape]
    zz, yy, xx = torch.meshgrid(*coordinates, indexing="ij")
    gaussian = torch.exp(-4 * (xx.square() + yy.square() + zz.square())).clamp_min(1e-3).numpy()
    probabilities = np.zeros((4, *target_shape), dtype=np.float32)
    weights = np.zeros(target_shape, dtype=np.float32)
    device = torch.device(device)
    model.eval()
    starts = [_sliding_starts(size, patch, overlap) for size, patch in zip(target_shape, patch_shape, strict=True)]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                pd, ph, pw = patch_shape
                patch = volume[:, z:z + pd, y:y + ph, x:x + pw][None].to(device)
                with torch.autocast(device_type=device.type, dtype=torch_amp_dtype, enabled=amp and device.type == "cuda"):
                    output = model(patch).softmax(dim=1)[0].float().cpu().numpy()
                probabilities[:, z:z + pd, y:y + ph, x:x + pw] += output * gaussian
                weights[z:z + pd, y:y + ph, x:x + pw] += gaussian
    probabilities /= weights[None].clip(min=1e-6)
    d, h, w = original_shape
    probabilities = probabilities[
        :,
        before[0]:before[0] + d,
        before[1]:before[1] + h,
        before[2]:before[2] + w,
    ].transpose(0, 2, 3, 1)
    outside = ~_prediction_support(item)
    probabilities[:, outside] = 0
    probabilities[0, outside] = 1
    return probabilities, metadata
