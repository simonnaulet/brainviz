"""Chargement des configurations TOML de Rep-SliceMix."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

from brainviz.models import CompactUNet, TinyUNet2D, TriPlaneRepSliceMixNet, UNet3D21D

Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if "extends" in config:
        parent = Path(config.pop("extends"))
        if not parent.is_absolute():
            parent = path.parent / parent
        base = load_config(parent)
        base.pop("_config_path", None)
        config = _deep_merge(base, config)
    config["_config_path"] = str(path.resolve())
    return config


def _deep_merge(base: Config, override: Config) -> Config:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_splits(path: str | Path) -> list[dict[str, list[str]]]:
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold, split in enumerate(splits):
        overlap = set(split["train"]) & set(split["val"])
        if overlap:
            raise ValueError(f"fold {fold}: fuite train/val {sorted(overlap)}")
    return splits


def build_model(config: Config):
    model = config["model"]
    architecture = model.get("architecture", "rep_slicemix")
    if architecture == "compact_unet":
        return CompactUNet(
            in_channels=int(model.get("in_channels", 6)),
            num_classes=int(model.get("num_classes", 4)),
            base_channels=int(model.get("base_channels", 16)),
            depth=int(model.get("depth", 3)),
        )
    if architecture == "tiny_unet_2d":
        return TinyUNet2D(input_mode=model.get("input_mode", "central"), num_classes=int(model.get("num_classes", 4)))
    if architecture == "unet_3d_21d":
        return UNet3D21D(
            in_channels=int(model.get("in_channels", 6)),
            num_classes=int(model.get("num_classes", 4)),
            widths=tuple(model.get("widths", (16, 32, 64, 128))),
        )
    if architecture != "rep_slicemix":
        raise ValueError(f"architecture inconnue: {architecture}")
    return TriPlaneRepSliceMixNet(
        in_channels=int(model.get("in_channels", 6)),
        num_classes=int(model.get("num_classes", 4)),
        widths=tuple(model.get("widths", (24, 48, 96, 192))),
        depths=tuple(model.get("depths", (2, 2, 2, 1))),
        mlp_ratio=int(model.get("mlp_ratio", 2)),
        multi_branch=bool(model.get("multi_branch", True)),
        film=bool(model.get("film", True)),
        down3_mode=str(model.get("down3_mode", "dense")),
        bottleneck_kernel_size=int(model.get("bottleneck_kernel_size", 3)),
        deep_supervision=bool(model.get("deep_supervision", False)),
        input_indices=tuple(model.get("input_indices", range(int(model.get("in_channels", 6))))),
    )
