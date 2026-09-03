from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from brainviz.data.loader import extract_data_slices

# valeurs de label du dataset iSeg-2017 : 0=fond, 10=LCR, 150=matière grise, 250=matière blanche
LABEL_VALUES = (0, 10, 150, 250)
# noms de classe alignés sur LABEL_VALUES ; après remappage, l'indice i correspond à LABEL_VALUES[i]
CLASS_NAMES = ("fond", "LCR", "matière grise", "matière blanche")


class BrainSliceDataset(Dataset):
    """Dataset PyTorch de tranches 2D (T1 ou T2) et de leur masque de segmentation.

    Charge tous les sujets d'un split au moment de la construction, extrait leurs
    tranches sur l'axe demandé et remappe le label vers des indices de classe
    contigus 0..num_classes-1, directement utilisables avec nn.CrossEntropyLoss.

    Args:
        root_dir (str | Path): dossier d'un split (ex. "dataset/train"), contenant
            un sous-dossier par sujet.
        axis (int): axe (0, 1 ou 2) selon lequel les tranches sont extraites.
        modality (str): "T1" ou "T2", la modalité IRM à utiliser comme entrée du modèle.
        scaling (str): mise à l'échelle des tranches, transmise à extract_data_slices
            ("padding" ou "bilinear").
        min_foreground_ratio (float): seuil de filtrage des tranches quasi vides, entre
            0 et 1. Une tranche est gardée si la fraction de ses pixels de label non-fond
            (label != 0) est >= min_foreground_ratio. 0.0 (défaut) ne filtre rien ; utile
            pour comparer l'apport des tranches peu/pas segmentées à l'entraînement.

    Raises:
        ValueError: axis, modality ou min_foreground_ratio invalide, root_dir n'est pas
            un dossier, ou root_dir ne contient aucun sous-dossier de sujet.

    Attributes:
        foreground_ratio (torch.Tensor): fraction de pixels non-fond par tranche gardée, (N,).
        subject_ids (np.ndarray): nom du sujet (ex. "subject-3") de chaque tranche gardée, (N,).
        num_slices_before_filter (int): nombre de tranches avant filtrage par min_foreground_ratio.
    """

    def __init__(self, root_dir, axis=2, modality="T1", scaling="padding", min_foreground_ratio=0.0):
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if modality not in ("T1", "T2"):
            raise ValueError("modality must be 'T1' or 'T2'")
        if not 0.0 <= min_foreground_ratio <= 1.0:
            raise ValueError("min_foreground_ratio must be between 0 and 1")

        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise ValueError("root_dir must be a directory")
        self.axis = axis
        self.modality = modality
        self.scaling = scaling
        self.min_foreground_ratio = min_foreground_ratio

        subject_dirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
        if not subject_dirs:
            raise ValueError(f"no subject directory found in {self.root_dir}")

        channel = 0 if modality == "T1" else 1
        value_to_class = {value: idx for idx, value in enumerate(LABEL_VALUES)}

        data_slices = []
        label_slices = []
        subject_ids = []
        for subject_dir in subject_dirs:
            data, label = extract_data_slices(subject_dir, axes=[axis], scaling=scaling)
            data_slices.append(data[:, channel:channel + 1])  # (N, 1, H, W)
            label_slices.append(label)
            subject_ids.extend([subject_dir.name] * data.shape[0])
        subject_ids = np.array(subject_ids)

        data = torch.cat(data_slices, dim=0)
        labels = torch.cat(label_slices, dim=0).squeeze(1).long()  # (N, H, W), valeurs brutes
        # on remplace les valeurs brutes du label par des indices de classe contigus 0..C-1
        # pour être directement utilisable avec nn.CrossEntropyLoss.
        class_labels = torch.zeros_like(labels)
        for value, idx in value_to_class.items():
            class_labels[labels == value] = idx

        self.num_slices_before_filter = data.shape[0]
        # fraction de pixels de label non-fond par tranche ; sert au filtrage ci-dessous
        # et à l'analyse de la distribution des tranches (cf. notebooks/test_data.ipynb).
        self.foreground_ratio = (class_labels != 0).float().mean(dim=(1, 2))
        if min_foreground_ratio > 0.0:
            keep = self.foreground_ratio >= min_foreground_ratio
            data, class_labels = data[keep], class_labels[keep]
            subject_ids = subject_ids[keep.numpy()]
            self.foreground_ratio = self.foreground_ratio[keep]

        self.data = data
        self.labels = class_labels
        self.subject_ids = subject_ids

    @property
    def num_classes(self):
        """int: nombre de classes de segmentation (len(LABEL_VALUES))."""
        return len(LABEL_VALUES)

    def __len__(self):
        """int: nombre total de tranches du dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Renvoie une tranche et son label.

        Args:
            idx (int): indice de la tranche.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: image (1, H, W) et label (H, W) d'indices
            de classe 0..num_classes-1.
        """
        return self.data[idx], self.labels[idx]


def get_dataloader(
    root_dir,
    axis=2,
    modality="T1",
    scaling="padding",
    min_foreground_ratio=0.0,
    batch_size=32,
    shuffle=True,
    **dataloader_kwargs,
):
    """Construit un DataLoader PyTorch sur un BrainSliceDataset.

    Args:
        root_dir (str | Path): voir BrainSliceDataset.
        axis (int): voir BrainSliceDataset.
        modality (str): voir BrainSliceDataset.
        scaling (str): voir BrainSliceDataset.
        min_foreground_ratio (float): voir BrainSliceDataset.
        batch_size (int): taille de batch.
        shuffle (bool): mélange les tranches à chaque epoch.
        **dataloader_kwargs: arguments supplémentaires passés à torch.utils.data.DataLoader.

    Returns:
        torch.utils.data.DataLoader: dataloader sur
        BrainSliceDataset(root_dir, axis, modality, scaling, min_foreground_ratio).
    """
    dataset = BrainSliceDataset(
        root_dir, axis=axis, modality=modality, scaling=scaling, min_foreground_ratio=min_foreground_ratio
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
