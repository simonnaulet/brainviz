import copy

import pytest
import torch

from brainviz.models import TriPlaneRepSliceMixNet, UNet3D21D, count_parameters
from brainviz.models.rep_slicemix import RepDW2d, RepDW3d


@pytest.mark.parametrize("module,input_shape", [
    (RepDW2d(8), (2, 8, 17, 19)),
    (RepDW2d(8, (7, 7), auxiliary_kernel_sizes=((3, 3),)), (2, 8, 17, 19)),
    (RepDW3d(8, (1, 3, 3)), (2, 8, 5, 17, 19)),
    (RepDW3d(8, (3, 1, 1)), (2, 8, 5, 17, 19)),
])
def test_repdw_reparameterization(module, input_shape):
    module.eval()
    x = torch.randn(input_shape)
    expected = module(x)
    module.reparameterize()
    actual = module(x)
    assert (expected - actual).abs().max() < 1e-4
    state = copy.deepcopy(module.state_dict())
    module.reparameterize()
    assert state.keys() == module.state_dict().keys()


def test_full_model_shapes_and_reparameterization():
    model = TriPlaneRepSliceMixNet().eval()
    x = torch.randn(2, 6, 5, 64, 80)
    plane = torch.tensor([0, 0])
    expected = model(x, plane)
    deployed = model.reparameterize(inplace=False)
    actual = deployed(x, plane)
    assert expected.shape == (2, 4, 64, 80)
    assert (expected - actual).abs().max() < 1e-4
    assert count_parameters(model) == 548_516
    assert count_parameters(deployed) == 543_380


def test_large_kernel_model_reparameterization():
    model = TriPlaneRepSliceMixNet(bottleneck_kernel_size=7).eval()
    x = torch.randn(2, 6, 5, 64, 80)
    plane = torch.tensor([0, 2])
    expected = model(x, plane)
    deployed = model.reparameterize(inplace=False)
    actual = deployed(x, plane)

    assert (expected - actual).abs().max() < 1e-4
    assert deployed.bottle[0].spatial.reparam_conv.kernel_size == (7, 7)
    assert count_parameters(deployed) - 543_380 == 7_680


def test_deep_supervision_outputs_are_training_only():
    model = TriPlaneRepSliceMixNet(deep_supervision=True)
    x = torch.randn(2, 6, 5, 64, 80)
    plane = torch.tensor([0, 2])

    main, h2, h4 = model(x, plane)
    assert main.shape == (2, 4, 64, 80)
    assert h2.shape == (2, 4, 32, 40)
    assert h4.shape == (2, 4, 16, 20)

    model.eval()
    assert model(x, plane).shape == main.shape
    deployed = model.reparameterize(inplace=False)
    assert not deployed.deep_supervision
    assert not hasattr(deployed, "auxiliary_heads")
    assert count_parameters(deployed) == 543_380


def test_film_starts_as_identity_and_pool_as_mean():
    model = TriPlaneRepSliceMixNet().eval()
    x = torch.randn(3, 24, 5, 8, 8)
    plane = torch.arange(3)
    assert torch.equal(model.film1(x, plane), x)
    assert torch.allclose(model.pool1(x, plane), x.mean(dim=2), atol=1e-6)


def test_invalid_model_input_is_rejected():
    model = TriPlaneRepSliceMixNet()
    with pytest.raises(ValueError):
        model(torch.randn(1, 6, 3, 64, 64), torch.zeros(1, dtype=torch.long))


def test_input_indices_ignores_excluded_channels():
    """Verrouille le slicing de forward() avant l'ablation D0/D0' (coords)."""
    torch.manual_seed(0)
    model = TriPlaneRepSliceMixNet(in_channels=3, input_indices=(0, 1, 5)).eval()
    plane = torch.zeros(2, dtype=torch.long)
    x = torch.randn(2, 6, 5, 64, 80)
    with torch.no_grad():
        baseline = model(x, plane)
        perturbed = x.clone()
        perturbed[:, [2, 3, 4]] = torch.randn_like(perturbed[:, [2, 3, 4]])
        actual = model(perturbed, plane)
    assert torch.equal(baseline, actual)


def test_input_indices_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        TriPlaneRepSliceMixNet(in_channels=3, input_indices=(0, 1))


def test_3d_competitor_shape_and_capacity():
    model = UNet3D21D(widths=(32, 64, 128, 256)).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 6, 16, 16, 16))
    assert output.shape == (1, 4, 16, 16, 16)
    assert 500_000 < count_parameters(model) < 700_000
