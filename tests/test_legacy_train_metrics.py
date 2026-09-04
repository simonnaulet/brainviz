import pytest
import torch

from brainviz.train import macro_dice_from_subject_totals


def test_macro_dice_is_computed_per_subject_before_averaging() -> None:
    # Sujet 1 parfait sur la classe testée, sujet 2 entièrement manqué. Une
    # agrégation globale pondérerait abusivement le plus grand second sujet.
    subject_totals = [
        (torch.tensor([10]), torch.tensor([20])),
        (torch.tensor([0]), torch.tensor([100])),
    ]

    result = macro_dice_from_subject_totals(subject_totals)

    assert result.item() == pytest.approx(0.5)


def test_macro_dice_rejects_empty_validation_set() -> None:
    with pytest.raises(ValueError, match="au moins un sujet"):
        macro_dice_from_subject_totals([])
