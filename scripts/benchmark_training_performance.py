#!/usr/bin/env python3
"""Micro-campagne reproductible des optimisations de la boucle d'entraînement."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from brainviz.config import build_model, load_config, load_splits
from brainviz.data.triplane import RandomPlaneBatchSampler, SliceStackDataset, collate_slice_stacks
from brainviz.training.engine import ModelEMA, optimizer_groups, seed_everything
from brainviz.training.losses import CompositeSegmentationLoss


class LegacyModelEMA(ModelEMA):
    """EMA scalaire antérieure, conservée uniquement comme témoin du benchmark."""

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[key].copy_(value)


def make_loader(config: dict, iterations: int, workers: int, *, pin_memory: bool) -> tuple[DataLoader, RandomPlaneBatchSampler]:
    split = load_splits(config["data"]["splits"])[0]
    root = Path(config["data"]["preprocessed_dir"])
    paths = [root / f"{case_id}.npz" for case_id in split["train"]]
    dataset = SliceStackDataset(paths, augment=True, cache_size=int(config["data"].get("cache_size", 8)))
    sampling = config["sampling"]
    sampler = RandomPlaneBatchSampler(
        dataset,
        batch_size=int(sampling["batch_size"]),
        iterations=iterations,
        seed=int(config["training"].get("seed", 12345)),
        d1_probability=float(sampling["d1_probability"]),
        brain_probability=float(sampling["brain_probability"]),
        planes=tuple(sampling.get("planes", (0, 1, 2))),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_slice_stacks,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=pin_memory,
    )
    return loader, sampler


def benchmark_loader(config: dict, workers: int, warmup: int, steps: int) -> dict:
    loader, sampler = make_loader(config, warmup + steps, workers, pin_memory=True)
    sampler.set_epoch(0)
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    start = time.perf_counter()
    pixels = 0
    for _ in range(steps):
        batch = next(iterator)
        pixels += batch["image"].numel()
    elapsed = time.perf_counter() - start
    del iterator, loader
    gc.collect()
    return {
        "workers": workers,
        "steps": steps,
        "seconds": elapsed,
        "milliseconds_per_batch": elapsed * 1000 / steps,
        "megapixels_per_second": pixels / elapsed / 1e6,
    }


VARIANTS = (
    {"name": "baseline", "deferred_metrics": False, "foreach_ema": False, "fused_adamw": False, "cudnn_benchmark": False, "dtype": "float16"},
    {"name": "deferred_metrics", "deferred_metrics": True, "foreach_ema": False, "fused_adamw": False, "cudnn_benchmark": False, "dtype": "float16"},
    {"name": "foreach_ema", "deferred_metrics": False, "foreach_ema": True, "fused_adamw": False, "cudnn_benchmark": False, "dtype": "float16"},
    {"name": "fused_adamw", "deferred_metrics": False, "foreach_ema": False, "fused_adamw": True, "cudnn_benchmark": False, "dtype": "float16"},
    {"name": "cudnn_benchmark", "deferred_metrics": False, "foreach_ema": False, "fused_adamw": False, "cudnn_benchmark": True, "dtype": "float16"},
    {"name": "optimized_eager_w4", "deferred_metrics": True, "foreach_ema": True, "fused_adamw": True, "cudnn_benchmark": True, "dtype": "float16", "workers": 4},
    {"name": "optimized_eager_w6", "deferred_metrics": True, "foreach_ema": True, "fused_adamw": True, "cudnn_benchmark": True, "dtype": "float16", "workers": 6},
    {"name": "optimized_eager_w6_no_benchmark", "deferred_metrics": True, "foreach_ema": True, "fused_adamw": True, "cudnn_benchmark": False, "dtype": "float16", "workers": 6},
    {"name": "optimized_bf16", "deferred_metrics": True, "foreach_ema": True, "fused_adamw": True, "cudnn_benchmark": True, "dtype": "bfloat16", "workers": 6},
)


def benchmark_variant(config: dict, variant: dict, warmup: int, steps: int, workers: int, device: torch.device) -> dict:
    seed_everything(int(config["training"].get("seed", 12345)))
    torch.backends.cudnn.benchmark = variant["cudnn_benchmark"]
    loader, sampler = make_loader(config, warmup + steps, workers, pin_memory=True)
    sampler.set_epoch(0)
    iterator = iter(loader)
    model = build_model(config).to(device).train()
    criterion = CompositeSegmentationLoss(**config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(config["training"]["weight_decay"])),
        lr=float(config["training"]["learning_rate"]),
        fused=variant["fused_adamw"],
    )
    dtype = torch.float16 if variant["dtype"] == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    ema_class = ModelEMA if variant["foreach_ema"] else LegacyModelEMA
    ema = ema_class(model, float(config["training"]["ema_decay"]))
    log_every = int(config["training"].get("log_every", 10))

    def step(batch, index: int, writer: SummaryWriter | None) -> dict[str, Tensor]:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        plane = batch["plane"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            losses = criterion(model(image, plane), target, valid)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        if not variant["deferred_metrics"] or index % log_every == 0:
            scalars = {key: float(value.detach()) for key, value in losses.items()}
            if writer is not None:
                for key, scalar in scalars.items():
                    writer.add_scalar(f"train/{key}", scalar, index)
        return losses

    with tempfile.TemporaryDirectory(prefix="brainviz-perf-") as log_dir:
        writer = SummaryWriter(log_dir)
        startup = time.perf_counter()
        losses = None
        for index in range(1, warmup + 1):
            losses = step(next(iterator), index, writer)
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - startup
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        measured_losses = []
        for offset in range(1, steps + 1):
            losses = step(next(iterator), warmup + offset, writer)
            measured_losses.append(losses["loss"].detach())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        mean_loss = float(torch.stack(measured_losses).mean())
        writer.close()
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20
    del losses, measured_losses, ema, scaler, optimizer, criterion, model, iterator, loader
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **variant,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "warmup_seconds": warmup_seconds,
        "seconds": elapsed,
        "milliseconds_per_iteration": elapsed * 1000 / steps,
        "iterations_per_second": steps / elapsed,
        "mean_loss": mean_loss,
        "peak_memory_mib": peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rep_slicemix.toml"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--loader-steps", type=int, default=100)
    parser.add_argument("--skip-loader", action="store_true")
    parser.add_argument("--variants", nargs="*", choices=[variant["name"] for variant in VARIANTS])
    parser.add_argument("--output", type=Path, default=Path("artifacts/rep_slicemix/performance_benchmark.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est requis pour ce benchmark")
    device = torch.device("cuda")
    config = load_config(args.config)

    loader_results = []
    if not args.skip_loader:
        for workers in (0, 2, 4, 6):
            print(f"DataLoader workers={workers}", flush=True)
            result = benchmark_loader(config, workers, args.warmup, args.loader_steps)
            loader_results.append(result)
            print(f"  {result['milliseconds_per_batch']:.2f} ms/batch", flush=True)

    training_results = []
    partial_payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "config": str(args.config),
        "loader": loader_results,
        "training": training_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(partial_payload, indent=2) + "\n", encoding="utf-8")
    selected_variants = [variant for variant in VARIANTS if not args.variants or variant["name"] in args.variants]
    for variant in selected_variants:
        print(f"Training variant={variant['name']}", flush=True)
        try:
            workers = int(variant.get("workers", args.workers))
            result = benchmark_variant(config, variant, args.warmup, args.steps, workers, device)
        except Exception as error:
            result = {**variant, "error": f"{type(error).__name__}: {error}"}
        training_results.append(result)
        args.output.write_text(json.dumps(partial_payload, indent=2) + "\n", encoding="utf-8")
        if "error" in result:
            print(f"  ERREUR: {result['error']}", flush=True)
        else:
            print(
                f"  {result['milliseconds_per_iteration']:.2f} ms/it, "
                f"loss={result['mean_loss']:.5f}, VRAM={result['peak_memory_mib']:.0f} MiB, "
                f"warmup={result['warmup_seconds']:.1f}s",
                flush=True,
            )

    baseline = next(
        (result for result in training_results if result["name"] == "baseline" and "error" not in result),
        None,
    )
    if baseline is not None:
        for result in training_results:
            if "error" not in result:
                result["speedup_vs_baseline"] = baseline["milliseconds_per_iteration"] / result["milliseconds_per_iteration"]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "config": str(args.config),
        "loader": loader_results,
        "training": training_results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Rapport: {args.output}", flush=True)


if __name__ == "__main__":
    main()
