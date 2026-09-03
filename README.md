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
