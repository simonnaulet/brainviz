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


def _brain_bbox(*volumes, margin):
    """Bounding box 3D du cerveau, à partir des voxels non nuls d'un ou plusieurs volumes.

    Le dataset iSeg-2017 est déjà skull-strippé : le fond vaut exactement 0 sur T1 et T2,
    et coïncide voxel à voxel avec le fond du label (vérifié sur les 10 sujets d'entraînement).
    Un simple seuillage d'intensité suffit donc à isoler le cerveau, sans risque de rogner
    du vrai tissu.

    Args:
        *volumes (np.ndarray): volumes 3D de même forme (T1, T2) ; un voxel appartient au
            cerveau s'il est non nul dans au moins un des volumes.
        margin (int): marge de sécurité ajoutée de chaque côté de la bbox, en voxels
            (tolère un sujet légèrement atypique, ex. en inférence sur le split test).

    Returns:
        tuple[np.ndarray, np.ndarray]: bornes (lo, hi) inclusives par axe, clampées aux
        dimensions du volume.
    """
    mask = np.zeros(volumes[0].shape, dtype=bool)
    for volume in volumes:
        mask |= volume > 0
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin, np.array(volumes[0].shape) - 1)
    return lo, hi


def _crop_to_bbox(volume, lo, hi):
    """Découpe volume selon les bornes (lo, hi) inclusives par axe (voir _brain_bbox)."""
    return volume[tuple(slice(l, h + 1) for l, h in zip(lo, hi))]


def _round_up(value, multiple):
    """Arrondit value au multiple supérieur (ou égal) de multiple."""
    return -(-value // multiple) * multiple


def compute_crop_size(subject_path, margin=4):
    """Taille de canvas (carré, multiple de 8) requise pour le crop cerveau d'un sujet.

    Ne charge que T1/T2 (pas le label) pour calculer la bbox, sans extraire les tranches.
    Comme chaque sujet a une bbox de taille différente, une cohorte entière doit être
    cropée sur une taille de canvas commune (le max de compute_crop_size sur tous les
    sujets) pour que les tranches restent empilables — voir BrainSliceDataset.

    Args:
        subject_path (str | Path): dossier du sujet, doit contenir T1.img et T2.img.
        margin (int): voir _brain_bbox.

    Returns:
        int: plus grand côté de la bbox (avec marge), arrondi au multiple de 8 supérieur.
    """
    subject_path = Path(subject_path)
    paths = {p.stem: p for p in subject_path.iterdir() if p.is_file() and p.suffix == ".img"}
    t1 = _load_volume(paths["T1"])
    t2 = _load_volume(paths["T2"])
    lo, hi = _brain_bbox(t1, t2, margin=margin)
    return _round_up(int((hi - lo + 1).max()), 8)


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


def extract_data_slices(subject_path, axes = [0, 1, 2], scaling="padding", crop=False, crop_margin=4, target_size=None):
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
        crop (bool): si True, découpe le volume à la bounding box 3D du cerveau (voxels
            T1 ou T2 non nuls, cf. _brain_bbox) avant extraction des tranches, pour
            réduire le fond envoyé au modèle. Défaut False (comportement inchangé).
        crop_margin (int): marge de sécurité (en voxels) autour de la bbox quand
            crop=True. Sans effet si crop=False.
        target_size (int | None): taille de canvas à imposer, au lieu de la calculer à
            partir de ce seul sujet. Nécessaire avec crop=True : chaque sujet a une bbox
            de taille différente, donc pour empiler plusieurs sujets (BrainSliceDataset)
            il faut leur imposer la même taille commune (voir compute_crop_size). Sans
            effet si scaling="bilinear" au-delà de fixer la taille de sortie ; avec
            "padding", doit être >= la taille du volume (cropé ou non) sous peine de
            padding négatif.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - data : tranches T1+T2, forme (N, 2, S, S).
            - label : tranches de label (valeurs brutes 0/10/150/250), forme (N, 1, S, S).
            N est la somme des tranches sur tous les axes demandés, S = target_size si
            fourni, sinon le plus grand côté du volume (éventuellement cropé, arrondi à
            un multiple de 8 si crop=True pour rester compatible avec la profondeur par
            défaut du CompactUNet).

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

    if crop:
        lo, hi = _brain_bbox(t1, t2, margin=crop_margin)
        t1 = _crop_to_bbox(t1, lo, hi)
        t2 = _crop_to_bbox(t2, lo, hi)
        label = _crop_to_bbox(label, lo, hi)

    if target_size is None:
        # les 3 axes du volume donnent des tranches de tailles différentes ;
        # on met tout à l'échelle du plus grand côté du volume pour pouvoir empiler.
        target_size = max(t1.shape)
        if crop:
            target_size = _round_up(target_size, 8)

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
