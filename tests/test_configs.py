from pathlib import Path

from brainviz.config import load_config
from brainviz.training.engine import configured_validation_planes


def test_validation_planes_follow_experiment_presets():
    root = Path("configs/experiments")
    for name in (
        "a_tiny_2d.toml",
        "b_tiny_25d.toml",
        "cprime_single_branch.toml",
        "c_reparameterized.toml",
        "d0_axial_coords.toml",
    ):
        assert configured_validation_planes(load_config(root / name)) == (0,)
    assert configured_validation_planes(load_config(root / "d_triplane.toml")) == (0, 1, 2)


def test_main_config_enables_measured_training_optimizations():
    config = load_config("configs/rep_slicemix.toml")
    assert config["data"]["num_workers"] == 6
    assert config["training"]["fused_adamw"] is True
    assert config["training"]["cudnn_benchmark"] is False
