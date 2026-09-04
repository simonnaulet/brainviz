#!/usr/bin/env python3
"""Convertit iSeg-2017 vers le format de dataset nnU-Net v2.

Les volumes Analyze extraits dans ``dataset/`` sont convertis en NIfTI. T1 et
T2 deviennent les canaux 0000 et 0001. Les labels iSeg non contigus
0/10/150/250 sont remappés vers 0/1/2/3.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ID = 501
DATASET_NAME = "iSeg2017"
LABEL_MAP = {0: 0, 10: 1, 150: 2, 250: 3}


def subject_id(path: Path) -> int:
    """Extrait l'identifiant numérique d'un dossier ``subject-N``."""
    try:
        return int(path.name.removeprefix("subject-"))
    except ValueError as exc:
        raise ValueError(f"Nom de sujet invalide : {path.name}") from exc


def load_volume(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Charge un volume Analyze et retire uniquement ses dimensions singleton."""
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj).squeeze()
    if data.ndim != 3:
        raise ValueError(f"{path}: volume 3D attendu, forme obtenue {data.shape}")
    return image, data


def save_nifti(data: np.ndarray, reference: nib.spatialimages.SpatialImage, output: Path) -> None:
    """Écrit un NIfTI 3D en conservant l'affine et l'espacement de la référence."""
    image = nib.Nifti1Image(data, reference.affine)
    image.header.set_zooms(reference.header.get_zooms()[:3])
    image.header.set_xyzt_units("mm")
    nib.save(image, str(output))


def validate_geometry(named_images: dict[str, nib.spatialimages.SpatialImage], subject: Path) -> None:
    """Vérifie l'alignement spatial des modalités et du masque d'un sujet."""
    reference_name, reference = next(iter(named_images.items()))
    reference_shape = tuple(int(v) for v in reference.shape[:3])
    for name, image in named_images.items():
        shape = tuple(int(v) for v in image.shape[:3])
        if shape != reference_shape or not np.allclose(image.affine, reference.affine):
            raise ValueError(
                f"{subject}: géométrie de {name} différente de {reference_name} "
                f"({shape} contre {reference_shape})"
            )


def convert_subject(subject: Path, images_dir: Path, labels_dir: Path | None) -> None:
    """Convertit les deux modalités et, si demandé, le masque d'un sujet."""
    sid = subject_id(subject)
    case = f"iseg_{sid:03d}"

    t1_image, t1 = load_volume(subject / "T1.img")
    t2_image, t2 = load_volume(subject / "T2.img")
    named_images = {"T1": t1_image, "T2": t2_image}

    label_image = None
    label = None
    if labels_dir is not None:
        label_image, label = load_volume(subject / "label.img")
        named_images["label"] = label_image

    validate_geometry(named_images, subject)
    save_nifti(t1.astype(np.int16, copy=False), t1_image, images_dir / f"{case}_0000.nii.gz")
    save_nifti(t2.astype(np.int16, copy=False), t2_image, images_dir / f"{case}_0001.nii.gz")

    if labels_dir is not None and label_image is not None and label is not None:
        values = {int(value) for value in np.unique(label)}
        unexpected = values - LABEL_MAP.keys()
        if unexpected:
            raise ValueError(f"{subject}: labels iSeg inconnus {sorted(unexpected)}")
        remapped = np.zeros(label.shape, dtype=np.uint8)
        for source, target in LABEL_MAP.items():
            remapped[label == source] = target
        save_nifti(remapped, label_image, labels_dir / f"{case}.nii.gz")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=project_root / "dataset")
    parser.add_argument("--nnunet-root", type=Path, default=project_root / "nnunet")
    parser.add_argument("--dataset-id", type=int, default=DATASET_ID)
    parser.add_argument("--force", action="store_true", help="recrée le dataset nnU-Net s'il existe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_root = args.input_dir / "train"
    test_root = args.input_dir / "test"
    train_subjects = sorted((p for p in train_root.glob("subject-*") if p.is_dir()), key=subject_id)
    test_subjects = sorted((p for p in test_root.glob("subject-*") if p.is_dir()), key=subject_id)
    if not train_subjects:
        raise FileNotFoundError(f"Aucun sujet d'entraînement dans {train_root}")

    dataset_dir = args.nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{DATASET_NAME}"
    if dataset_dir.exists():
        if not args.force:
            raise FileExistsError(f"{dataset_dir} existe déjà (utiliser --force pour le recréer)")
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    images_ts = dataset_dir / "imagesTs"
    labels_tr = dataset_dir / "labelsTr"
    for folder in (images_tr, images_ts, labels_tr):
        folder.mkdir(parents=True, exist_ok=True)

    for subject in train_subjects:
        print(f"Conversion train/{subject.name}")
        convert_subject(subject, images_tr, labels_tr)
    for subject in test_subjects:
        print(f"Conversion test/{subject.name}")
        convert_subject(subject, images_ts, None)

    metadata = {
        "channel_names": {"0": "T1", "1": "T2"},
        "labels": {
            "background": 0,
            "CSF": 1,
            "gray_matter": 2,
            "white_matter": 3,
        },
        "numTraining": len(train_subjects),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset nnU-Net créé : {dataset_dir}")
    print(f"  {len(train_subjects)} sujets annotés, {len(test_subjects)} sujets de test, 2 canaux")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
