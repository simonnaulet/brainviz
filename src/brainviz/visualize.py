import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from skimage import measure

from brainviz.data.dataset import CLASS_NAMES


def _to_numpy_2d(array):
    """Convertit un tenseur/array (H, W) ou (1, H, W) en array numpy 2D (H, W).

    Args:
        array (torch.Tensor | np.ndarray): image ou label à convertir.

    Returns:
        np.ndarray: array 2D (H, W).
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 3:
        array = array.squeeze(0)  # (1, H, W) -> (H, W)
    return array


def show_slice(image, label=None, title=None, modality=None, figsize=(8, 4), class_names=CLASS_NAMES):
    """Affiche côte à côte une tranche IRM (T1/T2) et son label de segmentation, légendé par classe.

    Args:
        image (torch.Tensor | np.ndarray): tranche (H, W) ou (1, H, W).
        label (torch.Tensor | np.ndarray | None): label (H, W) ou (1, H, W) d'indices de
            classe 0..len(class_names)-1 (format produit par BrainSliceDataset, pas les
            valeurs brutes 0/10/150/250). Si None, seule l'image est affichée.
        title (str | None): titre du panneau image. Par défaut, modality ou "image".
        modality (str | None): "T1" ou "T2", affiché dans le titre du panneau image
            si title n'est pas fourni.
        figsize (tuple[float, float]): taille de la figure matplotlib.
        class_names (tuple[str, ...]): noms de classe pour la légende, alignés sur les
            indices du label.

    Returns:
        matplotlib.figure.Figure: la figure créée.
    """
    image = _to_numpy_2d(image)
    image_title = title if title is not None else (modality or "image")

    if label is None:
        fig, ax = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
        ax.imshow(image, cmap="gray")
        ax.set_title(image_title)
        ax.axis("off")
        plt.close(fig)  # évite le double affichage Jupyter (auto-display inline + Out[n] du retour)
        return fig

    label = _to_numpy_2d(label)
    n_classes = len(class_names)
    cmap = plt.get_cmap("viridis", n_classes)

    fig, (ax_image, ax_label) = plt.subplots(1, 2, figsize=figsize)
    ax_image.imshow(image, cmap="gray")
    ax_image.set_title(image_title)
    ax_image.axis("off")

    # vmin/vmax fixes : la couleur de chaque classe reste la même d'une tranche à l'autre,
    # même si toutes les classes ne sont pas présentes sur celle-ci.
    # interpolation="nearest" : le label est catégoriel, pas de flou entre classes.
    ax_label.imshow(label, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_classes - 1)
    ax_label.set_title("label")
    ax_label.axis("off")

    handles = [Patch(color=cmap(i), label=name) for i, name in enumerate(class_names)]
    ax_label.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    plt.close(fig)  # évite le double affichage Jupyter (auto-display inline + Out[n] du retour)
    return fig


def find_nonempty_indices(dataset, min_ratio=0.2):
    """Cherche les tranches de dataset suffisamment "remplies" pour être représentatives.

    Beaucoup de tranches (bords du volume, hors du cerveau, ou juste en bordure) sont
    vides ou quasi vides ; ce filtre les écarte pour ne garder que celles qui valent
    la peine d'être affichées.

    Args:
        dataset (Dataset): dataset dont chaque élément est (image, label).
        min_ratio (float): un indice est gardé si son nombre de pixels de label non
            nuls atteint au moins min_ratio fois le maximum observé sur tout dataset.

    Returns:
        list[int]: indices des tranches suffisamment remplies.

    Raises:
        ValueError: dataset ne contient aucune tranche non vide.
    """
    scores = [int((dataset[i][1] != 0).sum()) for i in range(len(dataset))]
    best = max(scores)
    if best == 0:
        raise ValueError("dataset does not contain any non-empty slice")
    return [i for i, score in enumerate(scores) if score >= min_ratio * best]


def show_sample(dataset, idx=None):
    """Affiche une tranche aléatoire (non vide) de dataset et son label.

    Args:
        dataset (Dataset): dataset dont chaque élément est (image, label), par exemple
            un BrainSliceDataset.
        idx (int | None): indice de la tranche à afficher. Si None, tire au sort parmi
            les tranches suffisamment remplies (voir find_nonempty_indices) plutôt que
            de risquer un exemple vide ou quasi vide.

    Returns:
        matplotlib.figure.Figure: la figure créée par show_slice. Le titre indique la
        modalité (T1/T2) quand dataset l'expose (cas de BrainSliceDataset).
    """
    if idx is None:
        idx = random.choice(find_nonempty_indices(dataset))
    image, label = dataset[idx]
    modality = getattr(dataset, "modality", None)
    title = f"{modality} (idx={idx})" if modality else f"idx={idx}"
    return show_slice(image, label, title=title)


def get_patient_volume(dataset, subject_id):
    """Reconstruit le volume 3D des labels d'un patient à partir d'un BrainSliceDataset.

    Empile dans l'ordre d'origine du volume les tranches du patient demandé, extraites
    sur l'axe unique de dataset (dataset.axis). Nécessite un dataset non filtré
    (min_foreground_ratio=0.0) : filtrer des tranches romprait la continuité du volume.

    Args:
        dataset (BrainSliceDataset): dataset exposant subject_ids et labels.
        subject_id (str): nom du sujet (ex. "subject-1").

    Returns:
        torch.Tensor: volume (D, H, W) d'indices de classe 0..dataset.num_classes-1.

    Raises:
        ValueError: subject_id absent de dataset, ou dataset est filtré
            (min_foreground_ratio > 0).
    """
    if getattr(dataset, "min_foreground_ratio", 0.0) > 0.0:
        raise ValueError("dataset must be unfiltered (min_foreground_ratio=0.0) to reconstruct a full volume")
    mask = dataset.subject_ids == subject_id
    if not mask.any():
        raise ValueError(f"subject_id {subject_id!r} not found in dataset")
    return dataset.labels[mask]


def _marching_cubes_meshes(volume, class_names, classes, step_size=1):
    """Calcule le mesh marching-cubes (verts, faces, couleur) de chaque classe présente.

    Args:
        volume (np.ndarray): volume (D, H, W) d'indices de classe.
        class_names (tuple[str, ...]): noms de classe, alignés sur les indices du volume.
        classes (list[int] | None): indices de classe à reconstruire. Par défaut, toutes
            les classes non nulles présentes dans volume.
        step_size (int): pas de marching_cubes ; >1 sous-échantillonne le mesh (moins de
            sommets/faces, rendu moins fin mais bien plus léger, utile en interactif).

    Returns:
        list[tuple[int, np.ndarray, np.ndarray, tuple]]: (class_idx, verts, faces, couleur RGBA)
        pour chaque classe non vide.
    """
    n_classes = len(class_names)
    cmap = plt.get_cmap("viridis", n_classes)
    if classes is None:
        classes = sorted(c for c in np.unique(volume) if c != 0)

    meshes = []
    for class_idx in classes:
        mask = volume == class_idx
        if not mask.any():
            continue  # classe absente de ce volume (ex. label partiel ou prédiction manquée)
        # level=0.5 : isosurface entre 0 (hors classe) et 1 (dans la classe) sur le masque binaire.
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5, step_size=step_size)
        meshes.append((class_idx, verts, faces, cmap(class_idx)))
    return meshes


def plot_3d_segmentation_interactive(volume, class_names=CLASS_NAMES, classes=None, opacity=0.4, step_size=2):
    """Reconstruit et affiche en 3D (marching cubes) les surfaces de segmentation d'un volume.

    Rendu avec plotly : dans un notebook Jupyter, la figure retournée s'affiche comme un
    widget qu'on peut tourner, zoomer et déplacer à la souris.

    Fonctionne aussi bien sur un label de vérité terrain (y, ex. get_patient_volume) que
    sur une prédiction de modèle (y_hat, déjà réduite à des indices de classe via argmax) :
    la fonction ne dépend que de la forme et des valeurs du volume, pas de sa provenance.

    Args:
        volume (torch.Tensor | np.ndarray): volume (D, H, W) d'indices de classe
            0..len(class_names)-1 (0 = fond, exclu de la reconstruction).
        class_names (tuple[str, ...]): noms de classe, alignés sur les indices du volume.
        classes (list[int] | None): indices de classe à reconstruire. Par défaut, toutes
            les classes non nulles présentes dans volume.
        opacity (float): transparence des surfaces (0-1).
        step_size (int): voir _marching_cubes_meshes ; 2 (défaut) allège nettement le mesh
            (~4x moins de sommets qu'en pleine résolution) pour rester fluide dans le
            navigateur. Monter à 1 pour plus de détail si ça reste fluide côté client.

    Returns:
        plotly.graph_objects.Figure: la figure interactive (un mesh par classe, légendé).
    """
    import plotly.graph_objects as go
    from matplotlib.colors import to_hex

    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)

    fig = go.Figure()
    for class_idx, verts, faces, color in _marching_cubes_meshes(volume, class_names, classes, step_size):
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=to_hex(color),
            opacity=opacity,
            name=class_names[class_idx],
            showlegend=True,
        ))

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(title="classe"),
    )
    return fig
