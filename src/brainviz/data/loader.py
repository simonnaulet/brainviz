from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

_SCALING_MODES = ("padding", "bilinear")


def _load_volume(path):
    """Charge un volume Analyze 7.5 (.hdr/.img) en array numpy 3D (H, W, D).

    Args:
        path (Path): chemin du fichier .img à charger.

    Returns:
        np.ndarray: volume 3D, dimensions singleton retirées.
    """
    return np.asarray(nib.load(str(path)).dataobj).squeeze()


def _scale_slices(slices, target_size, scaling, interp_mode):
    """Met à l'échelle un lot de tranches vers une taille carrée commune.

    Args:
        slices (torch.Tensor): tranches (N, C, H, W).
        target_size (int): taille cible (target_size, target_size).
        scaling (str): "padding" (zero-padding centré) ou "bilinear" (interpolation).
        interp_mode (str): mode passé à F.interpolate quand scaling="bilinear"
            ("bilinear" pour des données continues, "nearest" pour un label catégoriel).

    Returns:
        torch.Tensor: tranches redimensionnées (N, C, target_size, target_size).
    """
    h, w = slices.shape[-2:]
    if scaling == "padding":
        pad_h, pad_w = target_size - h, target_size - w
        top, left = pad_h // 2, pad_w // 2
        return F.pad(slices, (left, pad_w - left, top, pad_h - top))
    align_corners = False if interp_mode == "bilinear" else None
    return F.interpolate(slices, size=(target_size, target_size), mode=interp_mode, align_corners=align_corners)


def extract_data_slices(subject_path, axes = [0, 1, 2], scaling="padding"):
    """Extrait les tranches T1/T2 et label d'un sujet du dataset iSeg-2017.

    Charge les volumes T1, T2 et label du sujet, puis extrait les tranches 2D selon
    les axes demandés (les tranches de tailles différentes selon l'axe sont mises à
    l'échelle du plus grand côté du volume pour pouvoir toutes être empilées).

    Args:
        subject_path (str | Path): dossier du sujet (ex. "dataset/train/subject-1"),
            doit contenir T1.img, T2.img et label.img.
        axes (list[int]): axes (0, 1 et/ou 2) selon lesquels extraire les tranches.
            Par défaut les trois axes.
        scaling (str): mise à l'échelle des tranches, "padding" (défaut) ou "bilinear".

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - data : tranches T1+T2, forme (N, 2, S, S).
            - label : tranches de label (valeurs brutes 0/10/150/250), forme (N, 1, S, S).
            N est la somme des tranches sur tous les axes demandés, S la taille cible.

    Raises:
        ValueError: subject_path n'est pas un dossier, scaling est invalide, ou
            T1/T2/label sont manquants dans subject_path.
    """
    subject_path = Path(subject_path)
    if not subject_path.is_dir():
        raise ValueError("subject_path must be a directory")
    if scaling not in _SCALING_MODES:
        raise ValueError(f"scaling must be one of {_SCALING_MODES}")

    paths = {p.stem: p for p in subject_path.iterdir() if p.is_file() and p.suffix == ".img"}
    if not {"T1", "T2", "label"} <= paths.keys():
        raise ValueError("subject_path must contain T1, T2 and label files")

    t1 = _load_volume(paths["T1"])
    t2 = _load_volume(paths["T2"])
    label = _load_volume(paths["label"])

    # les 3 axes du volume donnent des tranches de tailles différentes ;
    # on met tout à l'échelle du plus grand côté du volume pour pouvoir empiler.
    target_size = max(t1.shape)

    data_slices = []
    label_slices = []
    for axis in axes:
        t1_axis = np.moveaxis(t1, axis, 0)
        t2_axis = np.moveaxis(t2, axis, 0)
        data_axis = np.stack([t1_axis, t2_axis], axis=1)  # (N, 2, H, W)
        data_slices.append(_scale_slices(torch.from_numpy(data_axis).float(), target_size, scaling, "bilinear"))

        label_axis = np.moveaxis(label, axis, 0)[:, None]  # (N, 1, H, W)
        # le label est catégoriel : le lisser en bilinéaire créerait des classes inexistantes.
        label_slices.append(_scale_slices(torch.from_numpy(label_axis).float(), target_size, scaling, "nearest"))

    data_tensor = torch.cat(data_slices, dim=0)
    label_tensor = torch.cat(label_slices, dim=0)

    return data_tensor, label_tensor
