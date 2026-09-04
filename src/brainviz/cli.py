"""Commandes de la filière Rep-SliceMix."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import statistics

import nibabel as nib
import numpy as np
import torch

from brainviz.config import build_model, load_config
from brainviz.data.triplane import LABEL_MAP, preprocess_subject, restore_prediction
from brainviz.inference import parse_planes, predict_preprocessed, predict_preprocessed_3d
from brainviz.models import count_parameters
from brainviz.training.engine import train_fold
from brainviz.training.metrics import segmentation_metrics


def case_id(subject: Path) -> str:
    try:
        number = int(subject.name.removeprefix("subject-"))
    except ValueError as error:
        raise ValueError(f"nom de sujet invalide: {subject.name}") from error
    return f"iseg_{number:03d}"


def command_preprocess(args: argparse.Namespace) -> None:
    subjects = sorted(args.input_dir.glob("subject-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not subjects:
        raise FileNotFoundError(f"aucun sujet dans {args.input_dir}")
    for subject in subjects:
        output = args.output_dir / f"{case_id(subject)}.npz"
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} existe déjà; utiliser --force pour l'écraser")
        print(f"{subject.name} -> {output}")
        preprocess_subject(subject, output, margin=args.margin, multiple=args.multiple, require_label=not args.unlabeled)


def command_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model = build_model(config).eval()
    training_parameters = count_parameters(model)
    reparameterizable = hasattr(model, "reparameterize")
    deployed = model.reparameterize(inplace=False) if reparameterizable else model
    deployed_parameters = count_parameters(deployed)
    architecture = config["model"].get("architecture", "rep_slicemix")
    if architecture == "unet_3d_21d":
        size = args.patch_3d
        x = torch.randn(args.batch_size, model.in_channels, size, size, size)
        with torch.no_grad():
            y = model(x)
        error = 0.0
    else:
        x = torch.randn(args.batch_size, model.in_channels, 5, args.height, args.width)
        plane = torch.zeros(args.batch_size, dtype=torch.long)
        with torch.no_grad():
            y = model(x, plane)
            error = (y - deployed(x, plane)).abs().max().item()
    result = {
        "input": list(x.shape), "output": list(y.shape),
        "training_parameters": training_parameters,
        "deployed_parameters": deployed_parameters,
        "reparameterization_max_error": error,
    }
    if args.flops:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler:
            with torch.no_grad():
                deployed(x) if architecture == "unet_3d_21d" else deployed(x, plane)
        result["profiled_forward_flops"] = int(sum(event.flops or 0 for event in profiler.key_averages()))
    print(json.dumps(result, indent=2))


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    fold = args.fold if args.fold == "all" else int(args.fold)
    run = train_fold(config, fold, resume=args.resume, smoke=args.smoke, device_name=args.device)
    print(f"Run terminé: {run}")


def _model_from_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).eval()
    reparameterized = bool(checkpoint.get("reparameterized", False))
    if reparameterized:
        model.reparameterize()
        state = checkpoint["model"]
    else:
        state = checkpoint.get("ema", checkpoint["model"])
    model.load_state_dict(state)
    return model.to(device), config


def command_export(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model, config = _model_from_checkpoint(args.checkpoint, device)
    reparameterized = hasattr(model, "reparameterize")
    if reparameterized:
        model.eval().reparameterize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "reparameterized": reparameterized}, args.output)
    operation = "fusionné" if reparameterized else "exporté"
    print(f"Modèle {operation} ({count_parameters(model):,} paramètres): {args.output}")


def command_predict(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = _model_from_checkpoint(args.checkpoint, device)
    if config["model"].get("architecture") == "unet_3d_21d":
        probabilities, metadata = predict_preprocessed_3d(
            model, args.subject, patch_size=config["patch3d"]["patch_size"], device=device, amp=not args.no_amp
        )
    else:
        planes = tuple(config["sampling"].get("planes", (0, 1, 2))) if args.planes == "config" else parse_planes(args.planes)
        probabilities, metadata = predict_preprocessed(
            model,
            args.subject,
            planes=planes,
            slice_spacings=args.slice_spacings,
            batch_size=args.batch_size,
            device=device,
            amp=not args.no_amp,
        )
    output = restore_prediction(probabilities.argmax(axis=0).astype(np.uint8), metadata, raw_labels=not args.contiguous_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, args.output)
    print(f"Prédiction écrite: {args.output} ({time.perf_counter() - start:.2f} s/volume)")


def command_evaluate(args: argparse.Namespace) -> None:
    prediction_image, target_image = nib.squeeze_image(nib.load(args.prediction)), nib.squeeze_image(nib.load(args.target))
    if prediction_image.shape != target_image.shape or not np.allclose(prediction_image.affine, target_image.affine):
        raise ValueError("prédiction et cible n'ont pas la même géométrie")
    prediction = np.rint(np.asarray(prediction_image.dataobj)).astype(np.int16)
    target = np.rint(np.asarray(target_image.dataobj)).astype(np.int16)
    if prediction.max() > 3:
        prediction = np.vectorize(LABEL_MAP.__getitem__)(prediction)
    if target.max() > 3:
        target = np.vectorize(LABEL_MAP.__getitem__)(target)
    print(json.dumps(segmentation_metrics(prediction, target, target_image.header.get_zooms()[:3]), indent=2))


def positive_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("liste d'entiers attendue, par exemple 1 ou 1,2") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("les espacements doivent être strictement positifs")
    return result


def command_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).eval().to(device)
    if args.deployed and hasattr(model, "reparameterize"):
        model.reparameterize()
    architecture = config["model"].get("architecture", "rep_slicemix")
    if architecture == "unet_3d_21d":
        size = args.patch_3d
        x = torch.randn(args.batch_size, model.in_channels, size, size, size, device=device)
        forward = lambda: model(x)
    else:
        x = torch.randn(args.batch_size, model.in_channels, 5, args.height, args.width, device=device)
        plane = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        forward = lambda: model(x, plane)
    for _ in range(args.warmup):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            forward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iterations):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            forward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.iterations
    result = {"device": str(device), "milliseconds_per_batch": elapsed * 1000, "parameters": count_parameters(model)}
    if device.type == "cuda":
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(result, indent=2))


def command_probe_3d(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("probe-3d nécessite un GPU CUDA visible")
    config = load_config(args.config)
    if config["model"].get("architecture") != "unet_3d_21d":
        raise ValueError("utiliser la configuration de l'expérience E")
    shapes = [(int(size),) * 3 for size in config["patch3d"]["candidates"]]
    if args.include_full_crop:
        preprocessed = Path(config["data"]["preprocessed_dir"])
        subject_shapes = []
        for path in preprocessed.glob("iseg_*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                x, y, z = archive["brain_mask"].shape
            subject_shapes.append((z, x, y))
        if not subject_shapes:
            raise FileNotFoundError(f"aucun sujet prétraité dans {preprocessed}")
        full_shape = tuple(int(np.ceil(max(values) / 8) * 8) for values in zip(*subject_shapes, strict=True))
        if full_shape not in shapes:
            shapes.append(full_shape)
    results = []
    for shape in shapes:
        model = optimizer = x = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(config).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(args.batch_size, model.in_channels, *shape, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            results.append({"patch": list(shape), "fits": True, "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20})
            del model, optimizer, x, loss
        except torch.OutOfMemoryError:
            results.append({"patch": list(shape), "fits": False})
            model = optimizer = x = loss = None
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps({"device": torch.cuda.get_device_name(), "batch_size": args.batch_size, "results": results}, indent=2))


def command_summarize_cv(args: argparse.Namespace) -> None:
    epochs = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epochs.append(int(checkpoint["epoch"]) + 1)
    print(json.dumps({"best_epochs": epochs, "final_epochs_median": round(statistics.median(epochs))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brainviz-repslice")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = Path("configs/rep_slicemix.toml")

    preprocess = subparsers.add_parser("preprocess", help="prétraite un dossier iSeg")
    preprocess.add_argument("--input-dir", type=Path, default=Path("dataset/train"))
    preprocess.add_argument("--output-dir", type=Path, default=Path("artifacts/rep_slicemix/preprocessed"))
    preprocess.add_argument("--margin", type=int, default=8)
    preprocess.add_argument("--multiple", type=int, default=16)
    preprocess.add_argument("--unlabeled", action="store_true")
    preprocess.add_argument("--force", action="store_true")
    preprocess.set_defaults(function=command_preprocess)

    inspect = subparsers.add_parser("inspect", help="shapes, paramètres et équivalence")
    inspect.add_argument("--config", type=Path, default=default_config)
    inspect.add_argument("--batch-size", type=int, default=2)
    inspect.add_argument("--height", type=int, default=160)
    inspect.add_argument("--width", type=int, default=128)
    inspect.add_argument("--patch-3d", type=int, default=64)
    inspect.add_argument("--flops", action="store_true", help="profile les FLOPs du modèle déployé")
    inspect.set_defaults(function=command_inspect)

    train = subparsers.add_parser("train", help="entraîne un fold")
    train.add_argument("--config", type=Path, default=default_config)
    train.add_argument("--fold", required=True, help="0..4 ou all pour le réentraînement final")
    train.add_argument("--epochs", type=int, help="surcharge la durée, notamment pour --fold all")
    train.add_argument("--resume", type=Path)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--device", default="auto")
    train.set_defaults(function=command_train)

    export = subparsers.add_parser("export", help="exporte les poids EMA fusionnés")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.set_defaults(function=command_export)

    predict = subparsers.add_parser("predict", help="prédit un sujet prétraité")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("subject", type=Path)
    predict.add_argument("output", type=Path)
    predict.add_argument("--planes", default="config", help="config, all, axial, coronal ou sagittal séparés par des virgules")
    predict.add_argument(
        "--slice-spacings",
        type=positive_int_list,
        default=(1,),
        help="espacements de coupes moyennés, par exemple 1 ou 1,2 (défaut: 1)",
    )
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--no-amp", action="store_true")
    predict.add_argument("--contiguous-labels", action="store_true")
    predict.set_defaults(function=command_predict)

    evaluate = subparsers.add_parser("evaluate", help="calcule Dice/HD95/ASD")
    evaluate.add_argument("prediction", type=Path)
    evaluate.add_argument("target", type=Path)
    evaluate.set_defaults(function=command_evaluate)

    benchmark = subparsers.add_parser("benchmark", help="mesure la latence d'un batch")
    benchmark.add_argument("--config", type=Path, default=default_config)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--batch-size", type=int, default=16)
    benchmark.add_argument("--height", type=int, default=160)
    benchmark.add_argument("--width", type=int, default=128)
    benchmark.add_argument("--patch-3d", type=int, default=64)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--iterations", type=int, default=20)
    benchmark.add_argument("--deployed", action="store_true")
    benchmark.set_defaults(function=command_benchmark)

    probe = subparsers.add_parser("probe-3d", help="cherche le plus grand patch 3D tenant en VRAM")
    probe.add_argument("--config", type=Path, default=Path("configs/experiments/e_unet_3d.toml"))
    probe.add_argument("--batch-size", type=int, default=1)
    probe.add_argument("--include-full-crop", action=argparse.BooleanOptionalAction, default=True)
    probe.set_defaults(function=command_probe_3d)

    summarize = subparsers.add_parser("summarize-cv", help="calcule la durée médiane du réentraînement final")
    summarize.add_argument("checkpoints", type=Path, nargs="+")
    summarize.set_defaults(function=command_summarize_cv)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
