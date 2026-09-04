"""Pipeline exploratoire historique du CompactUNet sur iSeg-2017.

Ce module est conservé pour reproduire le rapport initial dans ``report/``. Il
n'utilise ni les folds, ni le preprocessing, ni l'ensemble des métriques du
pipeline principal et ses scores ne doivent donc pas être comparés directement
à Rep-SliceMix. Pour une comparaison contrôlée, utiliser :

    brainviz-repslice train --config configs/experiments/compact_unet_fair.toml --fold 0

Split par sujet (pas par tranche) pour éviter toute fuite entre train et validation :
avec 10 sujets, on réserve les 2 derniers (subject-9, subject-10) pour la validation.

Usage historique :
    uv run python -m brainviz.train [--epochs 50] [--modality T1T2] [--base-channels 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from brainviz.data.dataset import CLASS_NAMES, BrainSliceDataset
from brainviz.models import CompactUNet

VAL_SUBJECTS = {"subject-9", "subject-10"}


def dice_intersection_union(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    """Retourne les intersections et unions à accumuler avant le Dice.

    Calculer un Dice dans chaque batch puis moyenner surévalue les classes
    absentes de certains batches. L'intersection et l'union doivent être
    accumulées sur le volume complet de chaque sujet.
    """
    prediction = logits.argmax(dim=1)
    intersections, unions = [], []
    for label in range(num_classes):
        predicted = prediction == label
        expected = target == label
        intersections.append((predicted & expected).sum())
        unions.append(predicted.sum() + expected.sum())
    return torch.stack(intersections), torch.stack(unions)


def macro_dice_from_subject_totals(
    subject_totals: list[tuple[Tensor, Tensor]],
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Calcule le Dice par sujet, puis la moyenne macro par classe."""
    if not subject_totals:
        raise ValueError("au moins un sujet de validation est requis")
    subject_dices = [
        (2 * intersection.float() + eps) / (union.float() + eps)
        for intersection, union in subject_totals
    ]
    return torch.stack(subject_dices).mean(dim=0)


def split_datasets(root_dir, axis, modality, min_foreground_ratio):
    """Construit les datasets train/val en excluant VAL_SUBJECTS de train et inversement.

    BrainSliceDataset ne filtre pas par sujet nativement : on charge tout le split
    "train" du dataset iSeg-2017 puis on masque les tranches par subject_ids.
    """
    full = BrainSliceDataset(
        root_dir, axis=axis, modality=modality, scaling="padding", min_foreground_ratio=min_foreground_ratio
    )
    train_indices = [
        index
        for index, subject in enumerate(full.subject_ids)
        if subject not in VAL_SUBJECTS
    ]
    validation_datasets = [
        Subset(
            full,
            [
                index
                for index, current_subject in enumerate(full.subject_ids)
                if current_subject == subject
            ],
        )
        for subject in sorted(VAL_SUBJECTS)
    ]
    if not train_indices or any(not len(dataset) for dataset in validation_datasets):
        raise ValueError("split Compact U-Net incomplet")
    return (
        Subset(full, train_indices),
        validation_datasets,
        full.num_classes,
        full.num_channels,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds, validation_datasets, num_classes, num_channels = split_datasets(
        args.data_dir, args.axis, args.modality, args.min_foreground_ratio
    )
    validation_size = sum(len(dataset) for dataset in validation_datasets)
    print(
        f"train: {len(train_ds)} tranches, val: {validation_size} tranches, "
        f"canaux entrée: {num_channels}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for dataset in validation_datasets
    ]

    model = CompactUNet(
        in_channels=num_channels, num_classes=num_classes, base_channels=args.base_channels, depth=args.depth
    ).to(device)
    n_params = model.num_parameters()
    print(f"modèle: {n_params:,} paramètres")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_mean_dice = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        subject_totals: list[tuple[Tensor, Tensor]] = []
        with torch.no_grad():
            for validation_loader in validation_loaders:
                intersection_sum = torch.zeros(num_classes)
                union_sum = torch.zeros(num_classes)
                for x, y in validation_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss += criterion(logits, y).item() * x.size(0)
                    intersection, union = dice_intersection_union(
                        logits,
                        y,
                        num_classes,
                    )
                    intersection_sum += intersection.cpu()
                    union_sum += union.cpu()
                subject_totals.append((intersection_sum, union_sum))
        val_loss /= validation_size
        dice_per_c = macro_dice_from_subject_totals(subject_totals).tolist()
        mean_dice_fg = sum(dice_per_c[1:]) / (num_classes - 1)

        elapsed = time.time() - t0
        dice_str = ", ".join(f"{name}={d:.4f}" for name, d in zip(CLASS_NAMES, dice_per_c))
        print(
            f"epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"| dice: {dice_str} | mean_dice_fg={mean_dice_fg:.4f} | {elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice_per_class": dict(zip(CLASS_NAMES, dice_per_c)),
                "mean_dice_fg": mean_dice_fg,
                "elapsed_s": elapsed,
            }
        )

        if mean_dice_fg > best_mean_dice:
            best_mean_dice = mean_dice_fg
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    summary = {
        "modality": args.modality,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "num_parameters": n_params,
        "best_mean_dice_fg": best_mean_dice,
        "dice_per_param_x1e6": best_mean_dice / (n_params / 1e6),
        "history": history,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmeilleur mean_dice_fg: {best_mean_dice:.4f} | dice/1M params: {summary['dice_per_param_x1e6']:.4f}")
    print(f"résumé sauvegardé dans {out_dir / 'run_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="dataset/train", help="dossier du split d'entraînement")
    parser.add_argument("--out-dir", default="runs/compact_unet", help="dossier de sortie (poids + logs)")
    parser.add_argument("--axis", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--modality", default="T1T2", choices=("T1", "T2", "T1T2"))
    parser.add_argument("--min-foreground-ratio", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
