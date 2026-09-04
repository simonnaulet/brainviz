import pytest
import torch

from brainviz.config import build_model, load_config
from brainviz.models import CompactUNet, count_parameters


def test_original_compact_unet_shape_and_parameter_count():
    model = CompactUNet(in_channels=3, num_classes=4, base_channels=16, depth=3).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, 4, 256, 256)
    assert count_parameters(model) == 99_503


def test_compact_unet_uses_central_slice_in_shared_pipeline():
    model = CompactUNet(in_channels=6, num_classes=4, base_channels=8, depth=3).eval()
    central = torch.randn(2, 6, 64, 64)
    stack = torch.randn(2, 6, 5, 64, 64)
    stack[:, :, 2] = central
    plane = torch.tensor([0, 2])
    with torch.no_grad():
        expected = model(central)
        actual = model(stack, plane)
    torch.testing.assert_close(actual, expected)


def test_compact_unet_rejects_even_stack():
    model = CompactUNet(in_channels=6)
    with pytest.raises(ValueError, match="impair"):
        model(torch.randn(1, 6, 4, 64, 64))


def test_fair_compact_unet_config_builds():
    config = load_config("configs/experiments/compact_unet_fair.toml")
    model = build_model(config)
    assert isinstance(model, CompactUNet)
    assert model.in_channels == 6
    assert config["sampling"]["planes"] == [0, 1, 2]
