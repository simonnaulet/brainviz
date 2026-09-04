#!/usr/bin/env python3
"""Micro-benchmark du plan nnU-Net 2D sans lancer d'entraînement de données.

Le réseau et la loss de nnU-Net sont exécutés sur des tenseurs synthétiques avec
la taille de patch et le batch planifiés. Aucun checkpoint ni résultat de modèle
n'est créé. L'extrapolation porte sur les 1000 epochs du trainer standard.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

EPOCHS = 1000
TRAIN_STEPS_PER_EPOCH = 250
VAL_STEPS_PER_EPOCH = 50


def synchronize() -> None:
    torch.cuda.synchronize()


def format_duration(seconds: float) -> str:
    hours = seconds / 3600
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} jours"


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans",
        type=Path,
        default=project_root / "nnunet" / "nnUNet_preprocessed" / "Dataset501_iSeg2017" / "nnUNetPlans.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible")
    if args.warmup < 1 or args.steps < 1:
        raise ValueError("--warmup et --steps doivent être positifs")

    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    config = plans["configurations"]["2d"]
    architecture = config["architecture"]
    patch_size = tuple(config["patch_size"])
    batch_size = int(config["batch_size"])
    input_channels = 2
    output_channels = 4

    device = torch.device("cuda:0")
    torch.manual_seed(12345)
    torch.backends.cudnn.benchmark = True
    network = get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture["_kw_requires_import"],
        input_channels,
        output_channels,
        deep_supervision=True,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())

    data = torch.randn((batch_size, input_channels, *patch_size), device=device)
    with torch.no_grad(), torch.autocast("cuda", enabled=True):
        eager_outputs = network(data)
    targets = [
        torch.randint(0, output_channels, (batch_size, 1, *output.shape[2:]), device=device)
        for output in eager_outputs
    ]
    del eager_outputs

    loss = DC_and_CE_loss(
        {"batch_dice": bool(config["batch_dice"]), "smooth": 1e-5, "do_bg": False, "ddp": False},
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    weights = np.array([1 / (2**index) for index in range(len(targets))], dtype=float)
    weights[-1] = 0
    loss = DeepSupervisionWrapper(loss, weights / weights.sum())

    compile_enabled = not args.no_compile
    compile_seconds = 0.0
    if compile_enabled:
        compile_start = time.perf_counter()
        network = torch.compile(network)

    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True)
    scaler = torch.GradScaler("cuda")

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True):
            outputs = network(data)
            value = loss(outputs, targets)
        scaler.scale(value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

    for index in range(args.warmup):
        train_step()
        synchronize()
        if index == 0 and compile_enabled:
            compile_seconds = time.perf_counter() - compile_start

    torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    synchronize()
    train_seconds = (time.perf_counter() - start) / args.steps
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    network.eval()
    # torch.compile construit un graphe distinct pour eval/no_grad. Ne pas inclure
    # cette compilation ponctuelle dans le débit de validation extrapolé.
    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.autocast("cuda", enabled=True):
                outputs = network(data)
                _ = loss(outputs, targets)
    synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            with torch.autocast("cuda", enabled=True):
                outputs = network(data)
                _ = loss(outputs, targets)
    synchronize()
    val_seconds = (time.perf_counter() - start) / args.steps

    compute_seconds = (
        EPOCHS * TRAIN_STEPS_PER_EPOCH * train_seconds
        + EPOCHS * VAL_STEPS_PER_EPOCH * val_seconds
    )
    # Les tenseurs synthétiques n'incluent ni augmentation/transfert CPU, ni I/O,
    # checkpoints et validation finale. Cette marge est volontairement explicite.
    low_seconds = compute_seconds * 1.15 + compile_seconds
    high_seconds = compute_seconds * 1.50 + compile_seconds

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Réseau: {architecture['network_class_name'].split('.')[-1]}")
    print(f"Paramètres: {parameter_count:,}")
    print(f"Patch: {patch_size}; batch: {batch_size}")
    print(f"torch.compile: {compile_enabled}; compilation initiale: {compile_seconds:.1f} s")
    print(f"Étape train synthétique: {train_seconds:.4f} s")
    print(f"Étape validation synthétique: {val_seconds:.4f} s")
    print(f"VRAM PyTorch maximale mesurée: {peak_memory_gib:.2f} Gio")
    print(f"Calcul pur extrapolé: {format_duration(compute_seconds)}")
    print(f"Estimation réaliste par fold: {format_duration(low_seconds)} à {format_duration(high_seconds)}")
    print(f"Estimation 5 folds séquentiels: {format_duration(low_seconds * 5)} à {format_duration(high_seconds * 5)}")
    print("Aucun entraînement de données ni checkpoint n'a été créé.")
    return 0


if __name__ == "__main__":
    # Garder le cache de compilation local au projet si rien n'est configuré.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".torchinductor"))
    raise SystemExit(main())
