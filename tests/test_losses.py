import pytest
import torch

from brainviz.training.losses import CompositeSegmentationLoss, DeepSupervisionLoss, boundary_weights


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


def test_deep_supervision_loss_resizes_targets_and_backpropagates():
    outputs = (
        torch.randn(2, 4, 32, 48, requires_grad=True),
        torch.randn(2, 4, 16, 24, requires_grad=True),
        torch.randn(2, 4, 8, 12, requires_grad=True),
    )
    target = torch.randint(0, 4, (2, 32, 48))
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -3:] = False
    criterion = DeepSupervisionLoss(CompositeSegmentationLoss())

    result = criterion(outputs, target, valid)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert {"main_loss", "aux_1_loss", "aux_2_loss", "aux_1_soft_dice", "aux_2_soft_dice"} <= result.keys()
    assert all(output.grad is not None for output in outputs)
    assert sum(criterion.output_weights) == pytest.approx(1.0)
