#!/usr/bin/env python3
"""Crée le split 5-fold déterministe de la baseline nnU-Net iSeg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=project_root / "nnunet" / "nnUNet_raw" / "Dataset501_iSeg2017",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=project_root / "configs" / "splits_iseg.json",
        help="source de vérité partagée avec Rep-SliceMix",
    )
    parser.add_argument(
        "--preprocessed-dataset",
        type=Path,
        default=project_root / "nnunet" / "nnUNet_preprocessed" / "Dataset501_iSeg2017",
    )
    args = parser.parse_args()

    cases = sorted(path.name.removesuffix(".nii.gz") for path in (args.raw_dataset / "labelsTr").glob("*.nii.gz"))
    if len(cases) < 5:
        raise ValueError(f"Au moins 5 sujets sont requis pour le 5-fold, trouvé : {len(cases)}")

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    for fold, split in enumerate(splits):
        if set(split["train"]) & set(split["val"]):
            raise ValueError(f"fold {fold}: fuite entre train et val")
        if set(split["train"]) | set(split["val"]) != set(cases):
            raise ValueError(f"fold {fold}: les cas ne correspondent pas au dataset nnU-Net")
    output = args.preprocessed_dataset / "splits_final.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(splits, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"{output} existe avec un split différent; suppression manuelle requise")
    output.write_text(serialized, encoding="utf-8")

    print(f"Split écrit dans {output}")
    for fold, split in enumerate(splits):
        print(f"  fold {fold}: train={split['train']} val={split['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
