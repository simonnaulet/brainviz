import json

import numpy as np
import torch

from brainviz.inference import predict_preprocessed_3d


class _Uniform3D(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 4, *x.shape[-3:], device=x.device)


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
