#!/usr/bin/env python3
"""Compare les axes et espacements de coupes sans réentraîner Rep-SliceMix."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from brainviz.config import build_model, load_splits
from brainviz.data.triplane import PLANE_NAMES
from brainviz.inference import predict_preprocessed
from brainviz.training.metrics import segmentation_metrics


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).eval()
    if checkpoint.get("reparameterized", False):
        if not hasattr(model, "reparameterize"):
            raise ValueError("checkpoint fusionné mais modèle non reparamétrable")
        model.reparameterize()
        state = checkpoint["model"]
    else:
        state = checkpoint.get("ema", checkpoint["model"])
    model.load_state_dict(state)
    return model.to(device), config


def timed_view(model, path: Path, plane: int, spacing: int, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    probabilities, metadata = predict_preprocessed(
        model,
        path,
        planes=(plane,),
        slice_spacings=(spacing,),
        batch_size=args.batch_size,
        device=device,
        amp=not args.no_amp,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return probabilities, metadata, time.perf_counter() - start


def ensemble_name(planes: tuple[int, ...], spacings: tuple[int, ...]) -> str:
    plane_name = "+".join(PLANE_NAMES[plane] for plane in planes)
    spacing_name = "".join(f"d{spacing}" for spacing in spacings)
    return f"{plane_name}_{spacing_name}"


def mean_metrics(per_subject: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([subject[key] for subject in per_subject]))
        for key in per_subject[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/rep_slicemix/runs/fold0-20260903-233524/checkpoint_best_triplane.pt"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rep_slicemix/evaluations/fold0_tta.json"),
    )
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, config = load_model(args.checkpoint, device)
    split = load_splits(config["data"]["splits"])[args.fold]
    preprocessed = Path(config["data"]["preprocessed_dir"])
    paths = [preprocessed / f"{case_id}.npz" for case_id in split["val"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sujets de validation manquants: {missing}")

    # Un passage non chronométré absorbe l'initialisation CUDA/cuDNN.
    warmup, _, _ = timed_view(model, paths[0], 0, 1, args, device)
    del warmup

    subjects = []
    for path in paths:
        print(f"Inférence des six vues: {path.stem}", flush=True)
        views = {}
        timings = {}
        metadata = None
        for plane in range(3):
            for spacing in (1, 2):
                probabilities, metadata, elapsed = timed_view(model, path, plane, spacing, args, device)
                views[(plane, spacing)] = probabilities
                timings[(plane, spacing)] = elapsed
                print(f"  {PLANE_NAMES[plane]} d{spacing}: {elapsed:.3f}s", flush=True)
        assert metadata is not None
        with np.load(path, allow_pickle=False) as archive:
            if "label" not in archive.files:
                raise ValueError(f"{path}: label requis pour l'évaluation quantitative")
            target = archive["label"]
        subjects.append(
            {
                "id": path.stem,
                "target": target,
                "spacing": metadata["spacing"],
                "views": views,
                "timings": timings,
            }
        )

    results = []
    plane_sets = [subset for size in range(1, 4) for subset in combinations(range(3), size)]
    for spacings in ((1,), (2,), (1, 2)):
        for planes in plane_sets:
            keys = [(plane, spacing) for plane in planes for spacing in spacings]
            subject_metrics = []
            subject_seconds = []
            subject_records = []
            for subject in subjects:
                ensemble = np.zeros_like(subject["views"][keys[0]])
                for key in keys:
                    ensemble += subject["views"][key] / len(keys)
                metrics = segmentation_metrics(ensemble.argmax(axis=0), subject["target"], subject["spacing"])
                elapsed = float(sum(subject["timings"][key] for key in keys))
                subject_metrics.append(metrics)
                subject_seconds.append(elapsed)
                subject_records.append({"id": subject["id"], "seconds": elapsed, **metrics})
            results.append(
                {
                    "name": ensemble_name(planes, spacings),
                    "planes": [PLANE_NAMES[plane] for plane in planes],
                    "slice_spacings": list(spacings),
                    "views": len(keys),
                    "seconds_per_volume": float(np.mean(subject_seconds)),
                    **mean_metrics(subject_metrics),
                    "subjects": subject_records,
                }
            )

    by_name = {result["name"]: result for result in results}
    reference = by_name["axial_d1"]
    current = by_name["axial+coronal+sagittal_d1"]
    for result in results:
        result["dice_gain_vs_axial_d1"] = result["mean_dice"] - reference["mean_dice"]
        result["dice_gain_vs_current_triplane_d1"] = result["mean_dice"] - current["mean_dice"]
        result["cost_vs_axial_d1"] = result["seconds_per_volume"] / reference["seconds_per_volume"]

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "fold": args.fold,
        "validation_subjects": split["val"],
        "device": str(device),
        "amp": not args.no_amp and device.type == "cuda",
        "note": "Métriques sur le hold-out du fold; le test iSeg officiel n'a pas de labels locaux.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("\nConfiguration                              vues  Dice     CSF      GM       WM       s/vol", flush=True)
    for result in sorted(results, key=lambda item: (item["views"], item["name"])):
        print(
            f"{result['name']:<42} {result['views']:>2}  {result['mean_dice']:.5f} "
            f"{result['csf_dice']:.5f}  {result['gm_dice']:.5f}  {result['wm_dice']:.5f}  "
            f"{result['seconds_per_volume']:.3f}",
            flush=True,
        )
    print(f"\nRapport: {args.output}", flush=True)


if __name__ == "__main__":
    main()
