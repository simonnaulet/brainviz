"""Prétraitement et échantillonnage 2.5D tri-plan pour iSeg-2017."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
from scipy import ndimage
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

LABEL_MAP = {0: 0, 10: 1, 150: 2, 250: 3}
INVERSE_LABEL_MAP = np.asarray((0, 10, 150, 250), dtype=np.uint8)
PLANE_NAMES = ("axial", "coronal", "sagittal")
# Les tableaux canoniques suivent [X, Y, Z].
PLANE_AXES = (2, 1, 0)


def _load_3d(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    if len(image.shape) not in (3, 4):
        raise ValueError(f"{path}: volume 3D attendu, obtenu {image.shape}")
    if len(image.shape) == 4 and image.shape[-1] != 1:
        raise ValueError(f"{path}: seule une dimension singleton finale est acceptée")
    return nib.squeeze_image(image)


def _canonical_isotropic(image: nib.spatialimages.SpatialImage, *, order: int) -> nib.spatialimages.SpatialImage:
    canonical = nib.as_closest_canonical(image)
    spacing = np.asarray(canonical.header.get_zooms()[:3], dtype=float)
    if not np.allclose(spacing, 1.0, atol=1e-3):
        canonical = resample_to_output(canonical, voxel_sizes=(1.0, 1.0, 1.0), order=order)
    return canonical


def _brain_mask(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    mask = np.isfinite(t1) & np.isfinite(t2) & ((t1 != 0) | (t2 != 0))
    mask = ndimage.binary_closing(mask, structure=ndimage.generate_binary_structure(3, 1), iterations=1)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    mask = ndimage.binary_fill_holes(mask)
    if not mask.any():
        raise ValueError("le masque cérébral dérivé des images est vide")
    return mask


def _crop_box(mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.argwhere(mask)
    start = np.maximum(positions.min(axis=0) - margin, 0)
    stop = np.minimum(positions.max(axis=0) + margin + 1, mask.shape)
    return start, stop


def _pad_to_multiple(shape: Sequence[int], multiple: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray([(-int(size)) % multiple for size in shape], dtype=np.int16)
    return total // 2, total - total // 2


def preprocess_subject(
    subject_dir: str | Path,
    output_path: str | Path,
    *,
    margin: int = 8,
    multiple: int = 16,
    require_label: bool = True,
) -> Path:
    """Prétraite un sujet sans utiliser son label pour le crop ou les statistiques."""
    subject_dir, output_path = Path(subject_dir), Path(output_path)
    t1_native, t2_native = _load_3d(subject_dir / "T1.img"), _load_3d(subject_dir / "T2.img")
    if t1_native.shape != t2_native.shape or not np.allclose(t1_native.affine, t2_native.affine):
        raise ValueError(f"{subject_dir}: T1 et T2 ne sont pas co-enregistrés")
    t1_image = _canonical_isotropic(t1_native, order=3)
    t2_image = _canonical_isotropic(t2_native, order=3)
    if t1_image.shape != t2_image.shape or not np.allclose(t1_image.affine, t2_image.affine, atol=1e-4):
        raise ValueError(f"{subject_dir}: géométrie différente après canonicalisation")
    t1 = np.asarray(t1_image.dataobj, dtype=np.float32)
    t2 = np.asarray(t2_image.dataobj, dtype=np.float32)
    brain = _brain_mask(t1, t2)
    start, stop = _crop_box(brain, margin)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))

    normalized = []
    for image in (t1, t2):
        values = image[brain]
        std = float(values.std())
        if std < 1e-8:
            raise ValueError(f"{subject_dir}: modalité constante dans le masque")
        value = ((image - float(values.mean())) / std).astype(np.float32)
        value[~brain] = 0
        normalized.append(value[crop])

    axes = [np.linspace(-1, 1, size, dtype=np.float32) for size in t1.shape]
    coords = np.meshgrid(*axes, indexing="ij", sparse=False)
    arrays: dict[str, np.ndarray] = {
        "t1": normalized[0],
        "t2": normalized[1],
        "coord_x": coords[0][crop],
        "coord_y": coords[1][crop],
        "coord_z": coords[2][crop],
        "brain_mask": brain[crop].astype(np.uint8),
        "fov_valid": np.ones(tuple(stop - start), dtype=np.uint8),
    }

    label_path = subject_dir / "label.img"
    if label_path.exists():
        label_native = _load_3d(label_path)
        if label_native.shape != t1_native.shape or not np.allclose(label_native.affine, t1_native.affine):
            raise ValueError(f"{subject_dir}: label non co-enregistré")
        label_image = _canonical_isotropic(label_native, order=0)
        if label_image.shape != t1_image.shape or not np.allclose(label_image.affine, t1_image.affine, atol=1e-4):
            raise ValueError(f"{subject_dir}: label différent après canonicalisation")
        raw_label = np.rint(np.asarray(label_image.dataobj)).astype(np.int16)
        unknown = set(np.unique(raw_label).tolist()) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"{subject_dir}: labels inconnus {sorted(unknown)}")
        label = np.zeros_like(raw_label, dtype=np.uint8)
        for source, target in LABEL_MAP.items():
            label[raw_label == source] = target
        arrays["label"] = label[crop]
    elif require_label:
        raise FileNotFoundError(label_path)

    before, after = _pad_to_multiple(arrays["t1"].shape, multiple)
    pad = tuple((int(a), int(b)) for a, b in zip(before, after, strict=True))
    for key, value in arrays.items():
        arrays[key] = np.pad(value, pad, mode="constant")

    original_ornt = nib.orientations.io_orientation(t1_native.affine)
    canonical_ornt = nib.orientations.io_orientation(t1_image.affine)
    metadata = {
        "subject_id": subject_dir.name,
        "source_t1": str((subject_dir / "T1.img").resolve()),
        "original_shape": list(t1_native.shape),
        "canonical_shape": list(t1_image.shape),
        "original_affine": t1_native.affine.tolist(),
        "canonical_affine": t1_image.affine.tolist(),
        "canonical_to_original_ornt": nib.orientations.ornt_transform(canonical_ornt, original_ornt).tolist(),
        "crop_start": start.tolist(),
        "crop_stop": stop.tolist(),
        "pad_before": before.tolist(),
        "pad_after": after.tolist(),
        "spacing": [float(value) for value in t1_image.header.get_zooms()[:3]],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return output_path


@dataclass(frozen=True)
class SliceRequest:
    subject_index: int
    plane: int
    center: int
    spacing: int
    augmentation_seed: int = 0


@dataclass(frozen=True)
class VolumePatchRequest:
    subject_index: int
    center: tuple[int, int, int]
    augmentation_seed: int = 0


class SubjectStore:
    """Cache LRU borné de sujets `.npz`."""

    def __init__(self, paths: Sequence[str | Path], cache_size: int = 2) -> None:
        self.paths = tuple(Path(p) for p in paths)
        if not self.paths:
            raise ValueError("aucun sujet prétraité")
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray | dict]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> dict[str, np.ndarray | dict]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        with np.load(self.paths[index], allow_pickle=False) as archive:
            item = {key: archive[key] for key in archive.files if key != "metadata"}
            item["metadata"] = json.loads(str(archive["metadata"]))
        self._cache[index] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item


def to_plane(array: np.ndarray, plane: int, *, channel_first: bool = False) -> np.ndarray:
    """Place l'axe de coupe en premier (ou juste après les canaux)."""
    if plane not in range(3):
        raise ValueError("plane doit valoir 0, 1 ou 2")
    axis = PLANE_AXES[plane] + int(channel_first)
    return np.moveaxis(array, axis, int(channel_first))


def from_plane(array: np.ndarray, plane: int, *, channel_first: bool = False) -> np.ndarray:
    """Inverse exacte de :func:`to_plane`."""
    axis = PLANE_AXES[plane] + int(channel_first)
    return np.moveaxis(array, int(channel_first), axis)


class SliceStackDataset(Dataset[dict[str, Tensor | int]]):
    def __init__(self, paths: Sequence[str | Path], *, augment: bool = False, cache_size: int = 2) -> None:
        self.store = SubjectStore(paths, cache_size=cache_size)
        self.augmenter = SliceAugmenter() if augment else None
        self._candidates: dict[tuple[int, int, bool], np.ndarray] = {}
        for subject, path in enumerate(self.store.paths):
            with np.load(path, allow_pickle=False) as archive:
                brain_mask = archive["brain_mask"].astype(bool)
                fov_valid = archive["fov_valid"].astype(bool)
            for plane in range(3):
                mask = to_plane(brain_mask, plane)
                present = mask.reshape(mask.shape[0], -1).any(axis=1)
                valid = to_plane(fov_valid, plane).reshape(mask.shape[0], -1).any(axis=1)
                self._candidates[subject, plane, True] = np.flatnonzero(present)
                self._candidates[subject, plane, False] = np.flatnonzero(valid & ~present)

    def __len__(self) -> int:
        return len(self.store)

    def candidate_centers(self, subject: int, plane: int, brain: bool) -> np.ndarray:
        return self._candidates[subject, plane, brain]

    def __getitem__(self, request: SliceRequest) -> dict[str, Tensor | int]:
        item = self.store.get(request.subject_index)
        channels = np.stack(
            [item[name] for name in ("t1", "t2", "coord_x", "coord_y", "coord_z", "brain_mask")], axis=0
        )
        oriented = to_plane(channels, request.plane, channel_first=True)
        indices = np.clip(
            request.center + request.spacing * np.asarray((-2, -1, 0, 1, 2)), 0, oriented.shape[1] - 1
        )
        image = torch.from_numpy(np.ascontiguousarray(oriented[:, indices])).float()
        label_volume = to_plane(item["label"], request.plane)
        label = torch.from_numpy(np.ascontiguousarray(label_volume[request.center])).long()
        valid_volume = to_plane(item["fov_valid"], request.plane)
        valid = torch.from_numpy(np.ascontiguousarray(valid_volume[request.center])).bool()
        sample: dict[str, Tensor | int] = {"image": image, "label": label, "valid": valid, "plane": request.plane}
        return self.augmenter(sample, request.augmentation_seed) if self.augmenter is not None else sample


class RandomPlaneBatchSampler(Sampler[list[SliceRequest]]):
    """Émet un nombre fixe de batches, chacun associé à un seul plan."""

    def __init__(
        self,
        dataset: SliceStackDataset,
        *,
        batch_size: int = 16,
        iterations: int = 250,
        seed: int = 12345,
        d1_probability: float = 0.75,
        brain_probability: float = 0.9,
        planes: Sequence[int] = (0, 1, 2),
    ) -> None:
        self.dataset, self.batch_size, self.iterations = dataset, batch_size, iterations
        self.seed, self.epoch = seed, 0
        self.d1_probability, self.brain_probability = d1_probability, brain_probability
        self.planes = tuple(planes)

    def __len__(self) -> int:
        return self.iterations

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[SliceRequest]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.iterations):
            plane = rng.choice(self.planes)
            batch = []
            for _ in range(self.batch_size):
                subject = rng.randrange(len(self.dataset))
                wants_brain = rng.random() < self.brain_probability
                centers = self.dataset.candidate_centers(subject, plane, wants_brain)
                if not len(centers):
                    centers = self.dataset.candidate_centers(subject, plane, not wants_brain)
                center = int(centers[rng.randrange(len(centers))])
                spacing = 1 if rng.random() < self.d1_probability else 2
                batch.append(SliceRequest(subject, plane, center, spacing, rng.getrandbits(63)))
            yield batch


class SliceAugmenter:
    def __init__(self) -> None:
        self.gamma = (0.8, 1.25)

    @staticmethod
    def _affine(x: Tensor, angle: float, scale: float, interpolation: InterpolationMode) -> Tensor:
        return TF.affine(x, angle=angle, translate=(0, 0), scale=scale, shear=(0.0, 0.0), interpolation=interpolation, fill=0)

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high, generator=generator))

    def __call__(self, sample: dict[str, Tensor | int], seed: int) -> dict[str, Tensor | int]:
        generator = torch.Generator().manual_seed(seed)
        image, label, valid = sample["image"], sample["label"], sample["valid"]
        if torch.rand((), generator=generator) < 0.5:
            image, label, valid = image.flip(-1), label.flip(-1), valid.flip(-1)
        if torch.rand((), generator=generator) < 0.5:
            image, label, valid = image.flip(-2), label.flip(-2), valid.flip(-2)
        angle = self._uniform(generator, -15, 15)
        scale = self._uniform(generator, 0.9, 1.1)
        continuous = self._affine(image[:5].flatten(0, 1), angle, scale, InterpolationMode.BILINEAR).reshape_as(image[:5])
        brain_channel = self._affine(image[5:6].flatten(0, 1), angle, scale, InterpolationMode.NEAREST).reshape_as(image[5:6])
        image = torch.cat((continuous, brain_channel), dim=0)
        label = self._affine(label[None].float(), angle, scale, InterpolationMode.NEAREST)[0].long()
        valid = self._affine(valid[None].float(), angle, scale, InterpolationMode.NEAREST)[0].bool()

        brain = image[5:6].clamp(0, 1)
        for modality in range(2):
            x = image[modality]
            gamma = self._uniform(generator, *self.gamma)
            x = x.sign() * x.abs().clamp_min(1e-6).pow(gamma)
            sigma = self._uniform(generator, 0, 0.05)
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x + noise * sigma
            if torch.rand((), generator=generator) < 0.3:
                coarse = torch.randn(1, 1, 4, 4, generator=generator, device=x.device, dtype=x.dtype) * 0.12
                field = F.interpolate(coarse, x.shape[-2:], mode="bicubic", align_corners=False).exp()[0, 0]
                x = x * field
            image[modality] = x * brain[0]
        sample.update(image=image, label=label, valid=valid)
        return sample


def collate_slice_stacks(samples: Sequence[dict[str, Tensor | int]]) -> dict[str, Tensor]:
    planes = {int(sample["plane"]) for sample in samples}
    if len(planes) != 1:
        raise ValueError("un batch doit contenir un seul plan")
    max_h = math.ceil(max(sample["label"].shape[-2] for sample in samples) / 16) * 16
    max_w = math.ceil(max(sample["label"].shape[-1] for sample in samples) / 16) * 16
    images, labels, valid = [], [], []
    for sample in samples:
        h, w = sample["label"].shape[-2:]
        pad = (0, max_w - w, 0, max_h - h)
        images.append(F.pad(sample["image"], pad))
        labels.append(F.pad(sample["label"], pad))
        valid.append(F.pad(sample["valid"], pad))
    return {
        "image": torch.stack(images),
        "label": torch.stack(labels),
        "valid": torch.stack(valid),
        "plane": torch.full((len(samples),), planes.pop(), dtype=torch.long),
    }


class VolumePatchDataset(Dataset[dict[str, Tensor]]):
    """Patches 3D `[C,D,H,W]` extraits des mêmes sujets prétraités."""

    def __init__(self, paths: Sequence[str | Path], patch_size: int | Sequence[int], *, cache_size: int = 2, augment: bool = True) -> None:
        patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        if len(patch_size) != 3 or any(size % 8 for size in patch_size):
            raise ValueError("patch_size doit contenir trois dimensions divisibles par 8")
        self.store = SubjectStore(paths, cache_size)
        self.patch_size, self.augment = tuple(int(size) for size in patch_size), augment
        self._candidates: dict[tuple[int, bool], np.ndarray] = {}
        self.shapes: list[tuple[int, int, int]] = []
        for subject, path in enumerate(self.store.paths):
            with np.load(path, allow_pickle=False) as archive:
                brain = archive["brain_mask"].astype(bool)
                valid = archive["fov_valid"].astype(bool)
            self.shapes.append(brain.shape)
            self._candidates[subject, True] = np.flatnonzero(brain)
            self._candidates[subject, False] = np.flatnonzero(valid & ~brain)

    def __len__(self) -> int:
        return len(self.store)

    @staticmethod
    def _extract(array: np.ndarray, center: tuple[int, int, int], size: Sequence[int], channel_first: bool) -> np.ndarray:
        spatial_shape = array.shape[1:] if channel_first else array.shape
        size = np.asarray(size)
        starts = np.asarray(center) - size // 2
        stops = starts + size
        source_start = np.maximum(starts, 0)
        source_stop = np.minimum(stops, spatial_shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_stop, strict=True))
        cropped = array[(slice(None), *slices)] if channel_first else array[slices]
        before = source_start - starts
        after = stops - source_stop
        pad = tuple((int(a), int(b)) for a, b in zip(before, after, strict=True))
        return np.pad(cropped, ((0, 0), *pad) if channel_first else pad)

    def candidates(self, subject: int, brain: bool) -> np.ndarray:
        return self._candidates[subject, brain]

    def __getitem__(self, request: VolumePatchRequest) -> dict[str, Tensor]:
        item = self.store.get(request.subject_index)
        channels = np.stack([item[name] for name in ("t1", "t2", "coord_x", "coord_y", "coord_z", "brain_mask")])
        # La configuration est exprimée en D,H,W, tandis que les fichiers sont X,Y,Z.
        extract_shape = (self.patch_size[1], self.patch_size[2], self.patch_size[0])
        image = self._extract(channels, request.center, extract_shape, True)
        label = self._extract(item["label"], request.center, extract_shape, False)
        valid = self._extract(item["fov_valid"], request.center, extract_shape, False)
        # XYZ -> D(=Z), H(=X), W(=Y)
        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(0, 3, 1, 2))).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label.transpose(2, 0, 1))).long()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid.transpose(2, 0, 1))).bool()
        if self.augment:
            image_t, label_t, valid_t = augment_volume_patch(
                image_t, label_t, valid_t, request.augmentation_seed
            )
        return {"image": image_t, "label": label_t, "valid": valid_t}


class Random3DPatchBatchSampler(Sampler[list[VolumePatchRequest]]):
    def __init__(self, dataset: VolumePatchDataset, *, batch_size: int, iterations: int, seed: int, brain_probability: float = 0.9) -> None:
        self.dataset, self.batch_size, self.iterations = dataset, batch_size, iterations
        self.seed, self.epoch, self.brain_probability = seed, 0, brain_probability

    def __len__(self) -> int:
        return self.iterations

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[VolumePatchRequest]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.iterations):
            batch = []
            for _ in range(self.batch_size):
                subject = rng.randrange(len(self.dataset))
                candidates = self.dataset.candidates(subject, rng.random() < self.brain_probability)
                if not len(candidates):
                    candidates = self.dataset.candidates(subject, True)
                shape = self.dataset.shapes[subject]
                sampled = tuple(int(v) for v in np.unravel_index(int(candidates[rng.randrange(len(candidates))]), shape))
                patch_xyz = (self.dataset.patch_size[1], self.dataset.patch_size[2], self.dataset.patch_size[0])
                center = tuple(
                    dimension // 2 if patch >= dimension else coordinate
                    for coordinate, patch, dimension in zip(sampled, patch_xyz, shape, strict=True)
                )
                batch.append(VolumePatchRequest(subject, center, rng.getrandbits(63)))
            yield batch


def _affine_volume(
    tensor: Tensor,
    theta: Tensor,
    *,
    mode: str,
) -> Tensor:
    batched = tensor[None]
    grid = F.affine_grid(theta[None], batched.shape, align_corners=False)
    return F.grid_sample(batched, grid, mode=mode, padding_mode="zeros", align_corners=False)[0]


def augment_volume_patch(image: Tensor, label: Tensor, valid: Tensor, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Augmentation 3D déterministe, analogue à la politique 2.5D."""
    generator = torch.Generator().manual_seed(seed)
    for axis in (-1, -2, -3):
        if torch.rand((), generator=generator) < 0.5:
            image, label, valid = image.flip(axis), label.flip(axis), valid.flip(axis)
    angles = torch.empty(3).uniform_(-math.pi / 12, math.pi / 12, generator=generator)
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)
    rx = torch.tensor(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=image.dtype)
    ry = torch.tensor(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=image.dtype)
    rz = torch.tensor(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=image.dtype)
    scale = float(torch.empty(()).uniform_(0.9, 1.1, generator=generator))
    theta = torch.cat(((rz @ ry @ rx) / scale, torch.zeros(3, 1, dtype=image.dtype)), dim=1)
    continuous = _affine_volume(image[:5], theta, mode="bilinear")
    brain_channel = _affine_volume(image[5:6], theta, mode="nearest")
    image = torch.cat((continuous, brain_channel), dim=0)
    label = _affine_volume(label[None].float(), theta, mode="nearest")[0].long()
    valid = _affine_volume(valid[None].float(), theta, mode="nearest")[0].bool()
    brain = image[5].clamp(0, 1)
    for modality in range(2):
        x = image[modality]
        gamma = float(torch.empty(()).uniform_(0.8, 1.25, generator=generator))
        x = x.sign() * x.abs().clamp_min(1e-6).pow(gamma)
        sigma = float(torch.empty(()).uniform_(0, 0.05, generator=generator))
        x = x + torch.randn(x.shape, generator=generator, dtype=x.dtype) * sigma
        if torch.rand((), generator=generator) < 0.3:
            coarse = torch.randn(1, 1, 4, 4, 4, generator=generator, dtype=x.dtype) * 0.12
            field = F.interpolate(coarse, x.shape, mode="trilinear", align_corners=False)[0, 0].exp()
            x = x * field
        image[modality] = x * brain
    return image, label, valid


def restore_prediction(canonical_prediction: np.ndarray, metadata: dict, *, raw_labels: bool = True) -> nib.Nifti1Image:
    """Replace un label canonique croppé/paddé dans le volume natif."""
    before, after = np.asarray(metadata["pad_before"]), np.asarray(metadata["pad_after"])
    unpad = tuple(slice(int(a), int(size - b) if b else None) for size, a, b in zip(canonical_prediction.shape, before, after, strict=True))
    cropped = canonical_prediction[unpad]
    full = np.zeros(metadata["canonical_shape"], dtype=np.uint8)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(metadata["crop_start"], metadata["crop_stop"], strict=True))
    full[crop] = cropped
    if raw_labels:
        full = INVERSE_LABEL_MAP[full]
    canonical_image = nib.Nifti1Image(full, np.asarray(metadata["canonical_affine"]))
    original_shape = tuple(metadata["original_shape"])
    original_affine = np.asarray(metadata["original_affine"])
    if canonical_image.shape != original_shape or not np.allclose(canonical_image.affine, original_affine):
        return resample_from_to(canonical_image, (original_shape, original_affine), order=0)
    return canonical_image
