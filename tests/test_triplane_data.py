from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from brainviz.data.triplane import (
    SliceRequest,
    SliceAugmenter,
    SliceStackDataset,
    Random3DPatchBatchSampler,
    RandomPlaneBatchSampler,
    VolumePatchDataset,
    VolumePatchRequest,
    augment_volume_patch,
    collate_slice_stacks,
    from_plane,
    preprocess_subject,
    restore_prediction,
    to_plane,
)


@pytest.mark.parametrize("plane", [0, 1, 2])
@pytest.mark.parametrize("channel_first", [False, True])
def test_plane_round_trip(plane, channel_first):
    shape = (4, 5, 6, 7) if channel_first else (5, 6, 7)
    volume = np.arange(np.prod(shape)).reshape(shape)
    assert np.array_equal(from_plane(to_plane(volume, plane, channel_first=channel_first), plane, channel_first=channel_first), volume)


def test_collate_pads_and_rejects_mixed_planes():
    a = {"image": torch.ones(6, 5, 31, 47), "label": torch.ones(31, 47).long(), "valid": torch.ones(31, 47).bool(), "plane": 1}
    b = {"image": torch.ones(6, 5, 32, 64), "label": torch.ones(32, 64).long(), "valid": torch.ones(32, 64).bool(), "plane": 1}
    batch = collate_slice_stacks((a, b))
    assert batch["image"].shape == (2, 6, 5, 32, 64)
    assert not batch["valid"][0, -1].any()
    b["plane"] = 2
    with pytest.raises(ValueError):
        collate_slice_stacks((a, b))


def _save_pair(path: Path, data: np.ndarray) -> None:
    affine = np.diag((-1.0, 1.0, 1.0, 1.0))
    nib.save(nib.Nifti1Pair(data[..., None], affine), path)


def test_preprocessing_geometry_is_independent_from_label(tmp_path):
    subject_a, subject_b = tmp_path / "subject-1", tmp_path / "subject-2"
    subject_a.mkdir(); subject_b.mkdir()
    grid = np.indices((20, 24, 28))
    brain = ((grid[0] - 10) ** 2 / 36 + (grid[1] - 12) ** 2 / 49 + (grid[2] - 14) ** 2 / 64) < 1
    t1 = brain.astype(np.float32) * (10 + grid[0])
    t2 = brain.astype(np.float32) * (20 + grid[1])
    labels = [brain.astype(np.int16) * 10, brain.astype(np.int16) * 250]
    for subject, label in zip((subject_a, subject_b), labels, strict=True):
        _save_pair(subject / "T1.img", t1)
        _save_pair(subject / "T2.img", t2)
        _save_pair(subject / "label.img", label)
    out_a, out_b = tmp_path / "a.npz", tmp_path / "b.npz"
    preprocess_subject(subject_a, out_a, margin=2)
    preprocess_subject(subject_b, out_b, margin=2)
    with np.load(out_a) as a, np.load(out_b) as b:
        for key in ("t1", "t2", "brain_mask", "fov_valid"):
            assert np.array_equal(a[key], b[key])
        assert not np.array_equal(a["label"], b["label"])
        restored = restore_prediction(a["label"], json.loads(str(a["metadata"])))
    expected = np.asarray(nib.squeeze_image(nib.load(subject_a / "label.img")).dataobj)
    assert restored.shape == expected.shape
    assert np.array_equal(np.asarray(restored.dataobj), expected)


def test_stack_targets_central_slice(tmp_path):
    shape = (16, 32, 48)
    data = {
        "t1": np.zeros(shape, np.float32), "t2": np.zeros(shape, np.float32),
        "coord_x": np.zeros(shape, np.float32), "coord_y": np.zeros(shape, np.float32), "coord_z": np.zeros(shape, np.float32),
        "brain_mask": np.ones(shape, np.uint8), "fov_valid": np.ones(shape, np.uint8),
        "label": np.broadcast_to(np.arange(shape[2]), shape).astype(np.uint8),
        "metadata": np.asarray("{}"),
    }
    path = tmp_path / "subject.npz"
    np.savez(path, **data)
    dataset = SliceStackDataset([path])
    sample = dataset[SliceRequest(0, 0, 12, 2)]
    assert sample["image"].shape == (6, 5, 16, 32)
    assert torch.all(sample["label"] == 12)


def test_2d_augmentation_is_deterministic_and_keeps_mask_binary():
    sample = {
        "image": torch.cat((torch.randn(5, 5, 32, 32), torch.ones(1, 5, 32, 32))),
        "label": torch.ones(32, 32, dtype=torch.long),
        "valid": torch.ones(32, 32, dtype=torch.bool),
        "plane": 0,
    }
    first = SliceAugmenter()({key: value.clone() if torch.is_tensor(value) else value for key, value in sample.items()}, 42)
    second = SliceAugmenter()({key: value.clone() if torch.is_tensor(value) else value for key, value in sample.items()}, 42)
    assert torch.equal(first["image"], second["image"])
    assert set(torch.unique(first["image"][5]).tolist()) <= {0.0, 1.0}


def test_3d_augmentation_is_deterministic_and_keeps_mask_binary():
    image = torch.cat((torch.randn(5, 16, 16, 16), torch.ones(1, 16, 16, 16)))
    label = torch.ones(16, 16, 16, dtype=torch.long)
    valid = torch.ones_like(label, dtype=torch.bool)
    first = augment_volume_patch(image.clone(), label.clone(), valid.clone(), 7)
    second = augment_volume_patch(image.clone(), label.clone(), valid.clone(), 7)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert set(torch.unique(first[0][5]).tolist()) <= {0.0, 1.0}


def test_non_cubic_3d_patch_uses_dhw_convention(tmp_path):
    shape = (24, 32, 40)
    data = {
        "t1": np.zeros(shape, np.float32), "t2": np.zeros(shape, np.float32),
        "coord_x": np.zeros(shape, np.float32), "coord_y": np.zeros(shape, np.float32), "coord_z": np.zeros(shape, np.float32),
        "brain_mask": np.ones(shape, np.uint8), "fov_valid": np.ones(shape, np.uint8),
        "label": np.zeros(shape, np.uint8), "metadata": np.asarray("{}"),
    }
    path = tmp_path / "volume.npz"
    np.savez(path, **data)
    dataset = VolumePatchDataset([path], (16, 24, 32), augment=False)
    sample = dataset[VolumePatchRequest(0, (12, 16, 20))]
    assert sample["image"].shape == (6, 16, 24, 32)


def test_sampler_is_deterministic_and_full_patch_is_centered(tmp_path):
    shape = (24, 32, 40)
    data = {
        "t1": np.zeros(shape, np.float32), "t2": np.zeros(shape, np.float32),
        "coord_x": np.zeros(shape, np.float32), "coord_y": np.zeros(shape, np.float32), "coord_z": np.zeros(shape, np.float32),
        "brain_mask": np.ones(shape, np.uint8), "fov_valid": np.ones(shape, np.uint8),
        "label": np.zeros(shape, np.uint8), "metadata": np.asarray("{}"),
    }
    path = tmp_path / "volume.npz"
    np.savez(path, **data)
    slices = SliceStackDataset([path])
    first = list(RandomPlaneBatchSampler(slices, batch_size=2, iterations=2, seed=9))
    second = list(RandomPlaneBatchSampler(slices, batch_size=2, iterations=2, seed=9))
    assert first == second
    volumes = VolumePatchDataset([path], (40, 24, 32), augment=False)
    request = next(iter(Random3DPatchBatchSampler(volumes, batch_size=1, iterations=1, seed=9)))[0]
    assert request.center == (shape[0] // 2, shape[1] // 2, shape[2] // 2)
