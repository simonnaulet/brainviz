# BrainViz — segmentation cérébrale iSeg-2017

BrainViz compare des modèles 2D et 2.5D pour segmenter les IRM T1/T2 de
nourrissons en CSF, matière grise (GM) et matière blanche (WM). Le modèle
principal est **TriPlane Rep-SliceMix-Net**, un U-Net 2.5D compact qui partage
ses poids entre les plans axial, coronal et sagittal.

Le dépôt contient le pipeline complet : préparation d'iSeg-2017, preprocessing
sans information issue des labels, entraînement reproductible, reprise sur
checkpoint, validation volumique, ablations, export reparamétré et inférence
NIfTI.

## Résultat principal

Le meilleur run terminé est la baseline B0 sur le fold 0. Le checkpoint EMA de
l'epoch 200 obtient :

| Modèle | Paramètres déployés | CSF | GM | WM | Dice moyen |
|---|---:|---:|---:|---:|---:|
| Rep-SliceMix B0 | 543 380 | 0,95388 | 0,91830 | 0,89686 | **0,92302** |

Ce résultat est mesuré sur `iseg_001` et `iseg_008`, jamais vus pendant
l'entraînement. Il ne constitue ni une cross-validation complète, ni un score
officiel sur le test caché iSeg. B5 (`d=1` uniquement) atteint 0,92242 à
l'epoch 100, soit +0,00145 face à B0 au même point, mais doit encore être
confirmée sur un autre fold avant de remplacer la configuration principale.

La synthèse chiffrée et les décisions sont dans
[docs/experiments.md](docs/experiments.md). La model card détaillée est dans
[docs/model_card_rep_slicemix.md](docs/model_card_rep_slicemix.md).

## Méthode

Rep-SliceMix reçoit cinq coupes et six canaux par coupe : T1, T2, coordonnées
anatomiques RAS X/Y/Z et masque cérébral image-based. Il prédit la coupe
centrale. L'encodeur mélange l'information spatiale et inter-coupes avec des
convolutions depthwise séparées, tandis que FiLM et SlicePool conditionnent le
traitement sur le plan anatomique. Le décodeur est entièrement 2D.

Les branches `kernel principal + 1x1 + identité` de chaque RepDW sont fusionnées
exactement pour le déploiement. Le modèle passe de 548 516 paramètres pendant
l'entraînement à 543 380 après `reparameterize()`.

## Installation

Le projet utilise Python 3.13 et [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run python -c "import torch; ok=torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.get_device_name() if ok else 'CPU')"
```

Le groupe CUDA/NVIDIA est installé par défaut. Pour une machine AMD ROCm :

```bash
uv sync --no-group cuda --group rocm --group dev
```

## Données et preprocessing

Placer `iSeg-2017-Training.zip` et `iSeg-2017-Testing.zip` dans `archives/`, puis
exécuter :

```bash
uv run python scripts/prepare_dataset.py
uv run brainviz-repslice preprocess
```

Le preprocessing réoriente les volumes en RAS, construit le masque depuis T1/T2,
normalise les intensités dans ce masque, croppe, padde et conserve les métadonnées
nécessaires au retour dans la géométrie native. Les labels ne déterminent ni le
crop ni les statistiques de normalisation.

## Entraînement reproductible

Vérifier d'abord le modèle et l'équivalence de la reparamétrisation :

```bash
uv run brainviz-repslice inspect
uv run brainviz-repslice train --fold 0 --smoke
```

Lancer ensuite un fold complet :

```bash
uv run brainviz-repslice train --fold 0
```

Une reprise continue dans le dossier existant sans écraser les anciens runs :

```bash
uv run brainviz-repslice train --fold 0 \
  --resume artifacts/rep_slicemix/runs/<run>/checkpoint_last.pt
```

`--stop-after-epoch N` permet un screening court tout en conservant le scheduler
cosine prévu sur 200 epochs. Les splits sont fixés par sujet dans
`configs/splits_iseg.json`; le fold 0 utilise huit sujets train et deux sujets
validation. Il n'y a pas d'early stopping : `checkpoint_best_triplane.pt` garde
le meilleur Dice volumique de l'EMA, tandis que `checkpoint_last.pt` permet la
reprise exacte.

## Export, prédiction et évaluation

```bash
uv run brainviz-repslice export checkpoint_best_triplane.pt model_deployed.pt
uv run brainviz-repslice predict model_deployed.pt \
  artifacts/rep_slicemix/preprocessed/iseg_011.npz prediction.nii.gz
```

Le mode qualité moyenne les trois plans avec `d=1`. Si la latence compte,
coronal+sagittal conserve presque tout le score observé sur le fold 0 :

```bash
uv run brainviz-repslice predict model_deployed.pt <subject.npz> prediction.nii.gz \
  --planes coronal,sagittal --slice-spacings 1
```

Pour un sujet annoté :

```bash
uv run brainviz-repslice evaluate prediction.nii.gz dataset/train/subject-1/label.img
```

## Baselines et ablations

- `configs/rep_slicemix.toml` : configuration B0 gelée et recommandée.
- `configs/experiments/` : Compact U-Net, variantes A–E et ablations B1–B5.
- [docs/experiments.md](docs/experiments.md) : tableau maître des résultats et
  statut de chaque hypothèse.
- [docs/tta_ablation.md](docs/tta_ablation.md) : coût et qualité des plans et de
  `d=1/d=2`.
- [docs/training_performance.md](docs/training_performance.md) : micro-benchmarks
  de la boucle et options retenues/rejetées.
- [docs/nnunet_baseline.md](docs/nnunet_baseline.md) : baseline nnU-Net v2 2D.

Les chemins sous `artifacts/`, `dataset/`, `archives/` et `nnunet/` sont ignorés
par Git. Chaque configuration d'ablation possède son propre `output_dir` pour
éviter tout écrasement accidentel.

## Structure du dépôt

```text
configs/                 configurations et splits déterministes
docs/                    protocole, model card et rapports d'ablation
scripts/                 préparation, benchmarks et évaluation TTA
src/brainviz/data/       preprocessing et samplers 2D/2.5D/3D
src/brainviz/models/     Rep-SliceMix et modèles de comparaison
src/brainviz/training/   moteur, losses et métriques
src/brainviz/inference.py
tests/                   tests unitaires et d'intégration
report/                  archive du prototype Compact U-Net initial
```

## Vérifications

```bash
uv run --group dev pytest -q
```

Les tests couvrent notamment les permutations des trois plans, l'absence de
fuite liée aux labels, le padding dynamique, les losses, la reprise du RNG et
l'équivalence avant/après reparamétrisation à `1e-4` près.
