from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from brainviz.data.loader import (
    _brain_bbox,
    _scale_slices,
    compute_crop_size,
    extract_data_slices,
)


def _save_pair(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Pair(data, np.eye(4)), path)


def _make_subject(path: Path) -> None:
    path.mkdir()
    t1 = np.zeros((8, 10, 12), dtype=np.float32)
    t2 = np.zeros_like(t1)
    label = np.zeros_like(t1, dtype=np.int16)
    t1[1:6, 2:8, 3:10] = 1
    t2[1:6, 2:8, 3:10] = 2
    label[1:6, 2:8, 3:10] = 150
    _save_pair(path / "T1.img", t1)
    _save_pair(path / "T2.img", t2)
    _save_pair(path / "label.img", label)


def test_brain_bbox_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _brain_bbox(margin=0)
    with pytest.raises(ValueError, match="empty volumes"):
        _brain_bbox(np.zeros((2, 2, 2)), margin=0)
    with pytest.raises(ValueError, match="same shape"):
        _brain_bbox(np.ones((2, 2, 2)), np.ones((3, 2, 2)), margin=0)
    with pytest.raises(ValueError, match="non-negative"):
        _brain_bbox(np.ones((2, 2, 2)), margin=-1)


def test_padding_rejects_a_canvas_smaller_than_the_slice() -> None:
    slices = torch.zeros(2, 1, 8, 9)

    with pytest.raises(ValueError, match="smaller than slice size"):
        _scale_slices(slices, 8, "padding", "bilinear")


def test_crop_size_and_extraction_preserve_label_alignment(tmp_path: Path) -> None:
    subject = tmp_path / "subject-1"
    _make_subject(subject)

    assert compute_crop_size(subject, margin=1) == 16
    data, label = extract_data_slices(
        subject,
        axes=(2,),
        crop=True,
        crop_margin=1,
        target_size=16,
    )

    assert data.shape == (9, 2, 16, 16)
    assert label.shape == (9, 1, 16, 16)
    assert torch.equal(data[:, 0] > 0, label[:, 0] > 0)
    assert set(torch.unique(label).tolist()) == {0.0, 150.0}


def test_extract_rejects_invalid_axes_and_missing_files(tmp_path: Path) -> None:
    subject = tmp_path / "subject-1"
    subject.mkdir()

    with pytest.raises(ValueError, match="axes"):
        extract_data_slices(subject, axes=())
    with pytest.raises(ValueError, match="T1, T2 and label"):
        extract_data_slices(subject)

    with pytest.raises(ValueError, match="T1 and T2"):
        compute_crop_size(subject)
