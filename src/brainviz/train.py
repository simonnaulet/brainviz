"""Entraînement du CompactUNet sur iSeg-2017.

Split par sujet (pas par tranche) pour éviter toute fuite entre train et validation :
avec 10 sujets, on réserve les 2 derniers (subject-9, subject-10) pour la validation.

Usage:
    uv run python -m brainviz.train [--epochs 50] [--modality T1T2] [--base-channels 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brainviz.data.dataset import CLASS_NAMES, BrainSliceDataset
from brainviz.models import CompactUNet

VAL_SUBJECTS = {"subject-9", "subject-10"}


def dice_per_class(logits, target, num_classes, eps=1e-6):
    """Dice par classe sur un batch.

    Args:
        logits (torch.Tensor): (B, C, H, W), logits bruts du modèle.
        target (torch.Tensor): (B, H, W), indices de classe 0..C-1.
        num_classes (int): nombre de classes C.
        eps (float): terme de lissage au numérateur/dénominateur.

    Returns:
        torch.Tensor: (C,), score Dice par classe, moyenné sur le batch.
    """
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dices.append((2 * intersection + eps) / (union + eps))
    return torch.stack(dices)


def split_datasets(root_dir, axis, modality, min_foreground_ratio):
    """Construit les datasets train/val en excluant VAL_SUBJECTS de train et inversement.

    BrainSliceDataset ne filtre pas par sujet nativement : on charge tout le split
    "train" du dataset iSeg-2017 puis on masque les tranches par subject_ids.
    """
    full = BrainSliceDataset(
        root_dir, axis=axis, modality=modality, scaling="padding", min_foreground_ratio=min_foreground_ratio
    )
    is_val = torch.tensor([sid in VAL_SUBJECTS for sid in full.subject_ids])

    train_ds = torch.utils.data.Subset(full, torch.nonzero(~is_val).squeeze(1).tolist())
    val_ds = torch.utils.data.Subset(full, torch.nonzero(is_val).squeeze(1).tolist())
    return train_ds, val_ds, full.num_classes, full.num_channels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds, val_ds, num_classes, num_channels = split_datasets(
        args.data_dir, args.axis, args.modality, args.min_foreground_ratio
    )
    print(f"train: {len(train_ds)} tranches, val: {len(val_ds)} tranches, canaux entrée: {num_channels}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
        dice_sum = torch.zeros(num_classes)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                dice_sum += dice_per_class(logits, y, num_classes).cpu() * x.size(0)
        val_loss /= len(val_ds)
        dice_per_c = (dice_sum / len(val_ds)).tolist()
        mean_dice_fg = sum(dice_per_c[1:]) / (num_classes - 1)  # sans le fond, comme la métrique du challenge

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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