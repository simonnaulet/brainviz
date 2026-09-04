from pathlib import Path

from brainviz.config import build_model, load_config
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
    assert config["validation"]["axial_every"] == 5
    assert config["validation"]["triplane_every"] == 10
    assert config["validation"]["batch_size"] == 32


def test_architecture_ablation_presets_are_independent():
    root = Path("configs/experiments")
    b1 = load_config(root / "b1_large_kernel.toml")
    b2 = load_config(root / "b2_deep_supervision.toml")
    b3 = load_config(root / "b3_large_kernel_deep_supervision.toml")
    b4 = load_config(root / "b4_deeper_bottleneck.toml")

    assert b1["model"]["bottleneck_kernel_size"] == 7
    assert b1["model"]["deep_supervision"] is False
    assert b2["model"]["bottleneck_kernel_size"] == 3
    assert b2["model"]["deep_supervision"] is True
    assert b3["model"]["bottleneck_kernel_size"] == 7
    assert b3["model"]["deep_supervision"] is True
    assert b4["model"]["depths"] == [2, 2, 2, 2]
    assert b4["model"]["bottleneck_kernel_size"] == 3
    assert b4["model"]["deep_supervision"] is False
    assert build_model(b3).deep_supervision
    assert len(
        {
            b1["training"]["output_dir"],
            b2["training"]["output_dir"],
            b3["training"]["output_dir"],
            b4["training"]["output_dir"],
        }
    ) == 4


def test_d1_only_ablation_changes_only_sampling_and_output_directory():
    baseline = load_config("configs/rep_slicemix.toml")
    b5 = load_config("configs/experiments/b5_d1_only.toml")

    assert baseline["sampling"]["d1_probability"] == 0.75
    assert b5["sampling"]["d1_probability"] == 1.0
    assert b5["model"] == baseline["model"]
    assert b5["loss"] == baseline["loss"]
    assert b5["training"] | {"output_dir": baseline["training"]["output_dir"]} == baseline["training"]
