"""Boucle d'entraînement reproductible de Rep-SliceMix."""

from __future__ import annotations

import copy
from datetime import datetime
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from brainviz.config import build_model, load_splits
from brainviz.data.triplane import (
    Random3DPatchBatchSampler,
    RandomPlaneBatchSampler,
    SliceStackDataset,
    VolumePatchDataset,
    collate_slice_stacks,
)
from brainviz.inference import predict_preprocessed, predict_preprocessed_3d
from brainviz.training.losses import CompositeSegmentationLoss
from brainviz.training.metrics import segmentation_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def model_copy(self, model: nn.Module) -> nn.Module:
        result = copy.deepcopy(model).eval()
        result.load_state_dict(self.shadow)
        return result

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.shadow = {key: value.detach().clone() for key, value in state.items()}


def optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = parameter.ndim <= 1 or name.endswith("bias") or "embedding" in name or "layer_scale" in name
        (no_decay if excluded else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validation_summary(
    name: str,
    epoch: int,
    metrics: dict[str, Any],
    *,
    previous_best: float | None = None,
) -> str:
    """Construit une ligne lisible sans perdre le détail conservé dans metrics.jsonl."""
    mean_dice = float(metrics["mean_dice"])
    summary = (
        f"validation {name} epoch {epoch}: mean_dice={mean_dice:.5f} "
        f"CSF={float(metrics['csf_dice']):.5f} "
        f"GM={float(metrics['gm_dice']):.5f} "
        f"WM={float(metrics['wm_dice']):.5f}"
    )
    if previous_best is None:
        return summary
    if mean_dice > previous_best:
        if math.isfinite(previous_best):
            return f"{summary} NEW_BEST previous={previous_best:.5f} gain={mean_dice - previous_best:+.5f}"
        return f"{summary} NEW_BEST"
    return f"{summary} best={previous_best:.5f} delta={mean_dice - previous_best:+.5f}"


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location=device)`` déplace aussi ces tenseurs sur
    # CUDA. Les générateurs PyTorch exigent néanmoins des ByteTensor CPU.
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [
            cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {
            "name": properties.name,
            "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else list(torch.cuda.get_device_capability(device)),
            "total_memory_mib": properties.total_memory / 2**20,
        }
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True, timeout=2
        )
        info["git_commit"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


def _case_paths(directory: Path, case_ids: list[str]) -> list[Path]:
    paths = [directory / f"{case}.npz" for case in case_ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sujets non prétraités: {missing}")
    return paths


def configured_validation_planes(config: dict) -> tuple[int, ...]:
    return tuple(config["validation"].get("planes", config["sampling"].get("planes", (0, 1, 2))))


def validate_volumes(
    model: nn.Module,
    paths: list[Path],
    *,
    planes: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    per_subject = []
    for path in paths:
        probabilities, metadata = predict_preprocessed(
            model, path, planes=planes, batch_size=batch_size, device=device, amp=amp
        )
        with np.load(path, allow_pickle=False) as archive:
            target = archive["label"]
        per_subject.append(segmentation_metrics(probabilities.argmax(axis=0), target, metadata["spacing"]))
    keys = per_subject[0]
    result = {key: float(np.mean([metrics[key] for metrics in per_subject])) for key in keys}
    result["subjects"] = per_subject  # type: ignore[assignment]
    return result


def validate_volumes_3d(
    model: nn.Module,
    paths: list[Path],
    *,
    patch_size: int | tuple[int, int, int],
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    per_subject = []
    for path in paths:
        probabilities, metadata = predict_preprocessed_3d(model, path, patch_size=patch_size, device=device, amp=amp)
        with np.load(path, allow_pickle=False) as archive:
            target = archive["label"]
        per_subject.append(segmentation_metrics(probabilities.argmax(axis=0), target, metadata["spacing"]))
    keys = per_subject[0]
    result = {key: float(np.mean([metrics[key] for metrics in per_subject])) for key in keys}
    result["subjects"] = per_subject  # type: ignore[assignment]
    return result


def train_fold(
    config: dict,
    fold: int | str,
    *,
    resume: str | Path | None = None,
    smoke: bool = False,
    device_name: str = "auto",
) -> Path:
    """Entraîne un fold. ``smoke`` limite le run à deux itérations sans validation volumique."""
    training = config["training"]
    data_cfg, sampling = config["data"], config["sampling"]
    fold_index = int(fold) if fold != "all" else 0
    seed = int(training.get("seed", 12345)) + fold_index
    seed_everything(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    amp = bool(training.get("amp", True)) and device.type == "cuda"

    splits = load_splits(data_cfg["splits"])
    if fold == "all":
        all_cases = sorted(set(splits[0]["train"]) | set(splits[0]["val"]))
        split = {"train": all_cases, "val": []}
    else:
        fold = int(fold)
        if not 0 <= fold < len(splits):
            raise ValueError(f"fold invalide: {fold}")
        split = splits[fold]
    preprocessed = Path(data_cfg["preprocessed_dir"])
    train_paths = _case_paths(preprocessed, split["train"])
    val_paths = _case_paths(preprocessed, split["val"]) if split["val"] else []
    architecture = config["model"].get("architecture", "rep_slicemix")
    validation_planes = configured_validation_planes(config)
    iterations = 2 if smoke else int(sampling["iterations_per_epoch"])
    if architecture == "unet_3d_21d":
        patch_cfg = config["patch3d"]
        configured_patch = patch_cfg.get("patch_size", 96)
        patch_size = 32 if smoke else configured_patch
        dataset = VolumePatchDataset(
            train_paths, patch_size, cache_size=int(data_cfg.get("cache_size", 8)), augment=True
        )
        batch_size = min(1, int(patch_cfg.get("batch_size", 1))) if smoke else int(patch_cfg.get("batch_size", 1))
        sampler = Random3DPatchBatchSampler(
            dataset, batch_size=batch_size, iterations=iterations, seed=seed, brain_probability=float(sampling["brain_probability"])
        )
        collate_fn = None
    else:
        dataset = SliceStackDataset(train_paths, augment=True, cache_size=int(data_cfg.get("cache_size", 2)))
        batch_size = min(2, int(sampling["batch_size"])) if smoke else int(sampling["batch_size"])
        sampler = RandomPlaneBatchSampler(
            dataset,
            batch_size=batch_size,
            iterations=iterations,
            seed=seed,
            d1_probability=float(sampling["d1_probability"]),
            brain_probability=float(sampling["brain_probability"]),
            planes=tuple(sampling.get("planes", (0, 1, 2))),
        )
        collate_fn = collate_slice_stacks
    workers = 0 if smoke else int(data_cfg.get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    criterion = CompositeSegmentationLoss(**config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(training["weight_decay"])), lr=float(training["learning_rate"])
    )
    epochs = 1 if smoke else int(training["epochs"])
    total_steps = epochs * iterations
    warmup_steps = round(total_steps * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lambda(step, total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model, float(training["ema_decay"]))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if resume is None:
        run_dir = Path(training["output_dir"]) / f"fold{fold}-{timestamp}{'-smoke' if smoke else ''}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps(environment_info(device), indent=2) + "\n", encoding="utf-8"
        )
    else:
        run_dir = Path(resume).resolve().parent
    metrics_path = run_dir / "metrics.jsonl"
    start_epoch, global_step, best_dice = 0, 0, -float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_dice = float(checkpoint["best_dice"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tensorboard")
    except ImportError:
        writer = None

    run_start = time.perf_counter()
    run_start_step = global_step
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        running: dict[str, float] = {}
        for iteration, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            plane = batch.get("plane")
            if plane is not None:
                plane = plane.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(image) if architecture == "unet_3d_21d" else model(image, plane)
                losses = criterion(logits, target, valid)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            global_step += 1
            for key, value in losses.items():
                scalar = float(value.detach())
                running[key] = running.get(key, 0.0) + scalar
                if writer is not None:
                    writer.add_scalar(f"train/{key}", scalar, global_step)
            log_every = int(training.get("log_every", 25))
            if iteration % log_every == 0 or iteration == iterations:
                elapsed = time.perf_counter() - run_start
                completed = max(1, global_step - run_start_step)
                seconds_per_iteration = elapsed / completed
                remaining = max(0, epochs * iterations - global_step) * seconds_per_iteration
                print(
                    f"epoch {epoch + 1}/{epochs} iter {iteration}/{iterations} "
                    f"loss={float(losses['loss'].detach()):.4f} {seconds_per_iteration:.3f}s/it "
                    f"ETA={remaining / 60:.1f}min",
                    flush=True,
                )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"train_{key}": value / iterations for key, value in running.items()},
        }

        is_3d = architecture == "unet_3d_21d"
        full_validation = bool(val_paths) and not smoke and ((epoch + 1) % int(config["validation"]["triplane_every"]) == 0 or epoch + 1 == epochs)
        axial_validation = (
            not is_3d
            and not full_validation
            and bool(val_paths)
            and not smoke
            and (epoch + 1) % int(config["validation"]["axial_every"]) == 0
        )
        if axial_validation or full_validation:
            ema_model = ema.model_copy(model).to(device)
            if axial_validation:
                axial = validate_volumes(
                    ema_model, val_paths, planes=(0,), batch_size=int(config["validation"]["batch_size"]), device=device, amp=amp
                )
                record["val_axial"] = axial
                print(_validation_summary("axial", epoch + 1, axial), flush=True)
            if full_validation:
                previous_best = best_dice
                if is_3d:
                    triplane = validate_volumes_3d(
                        ema_model, val_paths, patch_size=config["patch3d"]["patch_size"], device=device, amp=amp
                    )
                else:
                    triplane = validate_volumes(
                        ema_model,
                        val_paths,
                        planes=validation_planes,
                        batch_size=int(config["validation"]["batch_size"]),
                        device=device,
                        amp=amp,
                    )
                record["val_triplane"] = triplane
                print(
                    _validation_summary("triplane", epoch + 1, triplane, previous_best=previous_best),
                    flush=True,
                )
                best_dice = max(best_dice, float(triplane["mean_dice"]))
            del ema_model
        payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_dice": best_dice,
            "config": config,
            "rng_state": rng_state(),
        }
        _atomic_save(payload, run_dir / "checkpoint_last.pt")
        if full_validation and float(record["val_triplane"]["mean_dice"]) >= best_dice:
            _atomic_save(payload, run_dir / "checkpoint_best_triplane.pt")
        with metrics_path.open("a", encoding="utf-8") as stream:
            record["epoch_seconds"] = time.perf_counter() - epoch_start
            stream.write(json.dumps(record, allow_nan=True) + "\n")
        if writer is not None:
            for prefix in ("val_axial", "val_triplane"):
                if prefix in record:
                    writer.add_scalar(f"{prefix}/mean_dice", record[prefix]["mean_dice"], epoch + 1)
            writer.flush()
    if writer is not None:
        writer.close()
    return run_dir
