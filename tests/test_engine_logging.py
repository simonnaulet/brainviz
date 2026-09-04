import copy

import torch

from brainviz.training.engine import _validation_summary, restore_rng_state, rng_state


METRICS = {
    "mean_dice": 0.9218622554,
    "csf_dice": 0.9535029066,
    "gm_dice": 0.9172089169,
    "wm_dice": 0.8948749427,
}


def test_validation_summary_reports_new_best_and_gain() -> None:
    summary = _validation_summary("triplane", 120, METRICS, previous_best=0.9209747385)

    assert summary == (
        "validation triplane epoch 120: mean_dice=0.92186 "
        "CSF=0.95350 GM=0.91721 WM=0.89487 "
        "NEW_BEST previous=0.92097 gain=+0.00089"
    )


def test_validation_summary_reports_best_when_score_does_not_improve() -> None:
    summary = _validation_summary("triplane", 130, METRICS, previous_best=0.925)

    assert summary.endswith("best=0.92500 delta=-0.00314")


def test_axial_validation_summary_has_no_best_marker() -> None:
    summary = _validation_summary("axial", 121, METRICS)

    assert summary == (
        "validation axial epoch 121: mean_dice=0.92186 "
        "CSF=0.95350 GM=0.91721 WM=0.89487"
    )


def test_restore_rng_state_normalizes_torch_state_to_cpu_byte_tensor() -> None:
    state = rng_state()
    expected_torch_state = state["torch"].clone()
    state_with_wrong_dtype = copy.deepcopy(state)
    state_with_wrong_dtype["torch"] = state["torch"].to(torch.int16)

    restore_rng_state(state_with_wrong_dtype)

    assert torch.equal(torch.get_rng_state(), expected_torch_state)
