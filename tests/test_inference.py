import argparse
import json

import numpy as np
import pytest
import torch

from brainviz.cli import positive_int_list
from brainviz.inference import predict_preprocessed, predict_preprocessed_3d


class _Uniform3D(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 4, *x.shape[-3:], device=x.device)


class _Recording2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stacks = []

    def forward(self, x, plane):
        del plane
        self.stacks.append(x[:, 0, :, 0, 0].clone())
        return torch.zeros(x.shape[0], 4, *x.shape[-2:], device=x.device)


def _write_2d_subject(path):
    shape = (8, 8, 7)
    z_values = np.broadcast_to(np.arange(shape[2], dtype=np.float32), shape)
    arrays = {
        "t1": z_values, "t2": np.ones(shape, np.float32),
        "coord_x": np.zeros(shape, np.float32), "coord_y": np.zeros(shape, np.float32),
        "coord_z": np.zeros(shape, np.float32), "brain_mask": np.ones(shape, np.uint8),
        "fov_valid": np.ones(shape, np.uint8), "label": np.zeros(shape, np.uint8),
        "metadata": np.asarray(json.dumps({})),
    }
    np.savez(path, **arrays)


def test_2d_inference_supports_d1_and_d2(tmp_path):
    path = tmp_path / "subject.npz"
    _write_2d_subject(path)
    model = _Recording2D()
    probabilities, _ = predict_preprocessed(
        model, path, planes=(0,), slice_spacings=(1, 2), batch_size=7, device="cpu", amp=False
    )
    assert probabilities.shape == (4, 8, 8, 7)
    assert len(model.stacks) == 2
    torch.testing.assert_close(model.stacks[0][0], torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    torch.testing.assert_close(model.stacks[1][0], torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0]))


def test_2d_inference_rejects_invalid_spacing(tmp_path):
    path = tmp_path / "subject.npz"
    _write_2d_subject(path)
    with pytest.raises(ValueError, match="strictement positifs"):
        predict_preprocessed(_Recording2D(), path, planes=(0,), slice_spacings=(0,), device="cpu", amp=False)


def test_cli_parses_slice_spacings():
    assert positive_int_list("1,2") == (1, 2)
    with pytest.raises(argparse.ArgumentTypeError, match="strictement positifs"):
        positive_int_list("0")


def test_non_cubic_3d_inference_restores_original_shape(tmp_path):
    shape = (8, 16, 24)
    arrays = {
        "t1": np.ones(shape, np.float32), "t2": np.ones(shape, np.float32),
        "coord_x": np.zeros(shape, np.float32), "coord_y": np.zeros(shape, np.float32), "coord_z": np.zeros(shape, np.float32),
        "brain_mask": np.ones(shape, np.uint8), "fov_valid": np.ones(shape, np.uint8),
        "label": np.zeros(shape, np.uint8), "metadata": np.asarray(json.dumps({})),
    }
    path = tmp_path / "subject.npz"
    np.savez(path, **arrays)
    probabilities, _ = predict_preprocessed_3d(
        _Uniform3D(), path, patch_size=(32, 16, 24), device="cpu", amp=False
    )
    assert probabilities.shape == (4, *shape)
    assert np.allclose(probabilities, 0.25)
