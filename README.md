# brainviz

Segmentation d'IRM cérébrales sur le dataset [iSeg-2017](https://iseg2017.web.unc.edu/) (T1/T2 → LCR / matière grise / matière blanche). Projet étudiant (hackathon SCIA 2026).

## Structure

```
scripts/prepare_dataset.py   # extrait les archives iSeg-2017 vers dataset/
src/brainviz/
  data/loader.py              # extract_data_slices : sujet -> tenseurs de tranches T1/T2 + label
  data/dataset.py              # BrainSliceDataset (torch Dataset) + get_dataloader
  visualize.py                 # show_slice / show_sample : affichage image + label légendé
notebooks/test_data.ipynb    # notebook d'exploration
dataset/                     # généré par prepare_dataset.py, pas versionné
archives/                    # archives iSeg-2017 brutes, pas versionné
```

## Setup

```bash
uv sync
```

Installe les dépendances, dont PyTorch (CPU/NVIDIA par défaut). Sur GPU AMD, utiliser
la variante ROCm à la place :

```bash
uv sync --no-group cuda --group rocm
```

⚠️ `uv run` resynchronise sur le groupe par défaut (`cuda`) et écrase les wheels ROCm.
Sur une machine AMD, préfixer les commandes avec les mêmes groupes :

```bash
uv run --no-group cuda --group rocm python -c "import torch; print(torch.__version__)"
```

## Dataset

Placer les deux archives (`iSeg-2017-Training.zip`, `iSeg-2017-Testing.zip`) dans `archives/`, puis :

```bash
uv run python scripts/prepare_dataset.py
```

Génère `dataset/train/subject-<n>/{T1,T2,label}.{hdr,img}` (sujets 1-10) et
`dataset/test/subject-<n>/{T1,T2}.{hdr,img}` (sujets 11-23, sans label).

## Usage rapide

```python
from brainviz.data.dataset import get_dataloader
from brainviz.visualize import show_sample

loader = get_dataloader("dataset/train", axis=2, modality="T1", batch_size=32)
x, y = next(iter(loader))          # x: (B, 1, H, W), y: (B, H, W) indices de classe

show_sample(loader.dataset)        # affiche une tranche + label, avec légende
```

## Baseline nnU-Net v2 2D

Préparer et préprocesser la baseline sans lancer l'entraînement :

```bash
scripts/setup_nnunet_baseline.sh
```

Le setup utilise T1 et T2, vérifie l'intégrité du dataset, planifie uniquement
le preprocessing 2D et crée un split 5-fold au niveau sujet. Le réseau obtenu
est un `PlainConvUNet` 2D (`patch=160x128`, `batch=41`).

Une fois prêt, le fold 0 se lance explicitement avec :

```bash
scripts/train_nnunet_baseline.sh 0
```

Voir [docs/nnunet_baseline.md](docs/nnunet_baseline.md) pour la configuration,
les splits et l'estimation de durée sur la RTX 5070 Ti.

## TriPlane Rep-SliceMix-Net

La seconde baseline est un réseau 2.5D compact qui traite cinq coupes T1/T2 dans
les trois plans. Le choix 2.5D vise le **champ de vision d'un plan complet à sa
résolution native**, et non simplement une économie face à un CNN 3D. Ce contexte
anatomique global peut aider durant la phase isointense, lorsque le contraste local
GM/WM est faible. Un U-Net 3D à blocs depthwise (2+1)D reste implémenté comme
concurrent expérimental direct.

Les branches structurelles sont inspirées de
[RepVGG](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)
et de [Diverse Branch Block](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.html).
Elles sont fusionnées en convolutions depthwise uniques pour le déploiement.

Préparer les dix sujets annotés :

```bash
uv run --group dev brainviz-repslice preprocess
```

Inspecter le réseau sans entraînement :

```bash
uv run --group dev brainviz-repslice inspect
```

Faire uniquement le test d'intégration de deux itérations :

```bash
uv run --group dev brainviz-repslice train --fold 0 --smoke
```

Le vrai entraînement reste une commande explicite :

```bash
uv run --group dev brainviz-repslice train --fold 0
```

Le modèle principal possède 548 516 paramètres pendant l'entraînement et
543 380 après fusion. Voir [docs/rep_slicemix.md](docs/rep_slicemix.md) pour les
formats, l'inférence, les checkpoints et les ablations.
