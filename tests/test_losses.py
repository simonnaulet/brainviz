import torch

from brainviz.training.losses import CompositeSegmentationLoss, boundary_weights


def test_boundary_weights_prioritize_gm_wm():
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 2:8] = 2
    target[:, 8:14] = 3
    valid = torch.ones_like(target, dtype=torch.bool)
    weights = boundary_weights(target, valid)
    assert weights.max() == 3
    assert (weights == 2).any()
    assert weights[0, 7, 8] == 3


def test_loss_ignores_invalid_padding_and_backpropagates():
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16))
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :, 12:] = False
    criterion = CompositeSegmentationLoss()
    result = criterion(logits, target, valid)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert logits.grad is not None


def test_loss_supports_3d_patches():
    logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    result = CompositeSegmentationLoss()(logits, target)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
