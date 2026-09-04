# Baseline nnU-Net v2 2D pour iSeg-2017

## Choix du réseau

La baseline utilise le `PlainConvUNet` 2D standard planifié automatiquement par
nnU-Net v2.8.1. C'est le meilleur premier essai pour ce projet : seulement 10
sujets annotés, délai de hackathon, et objectif explicite de traiter des coupes
2D sans contexte voisin.

Les presets ResEnc sont la recommandation moderne de nnU-Net pour rechercher la
meilleure performance. Le preset L recommandé comme défaut demande environ 24 Go
de VRAM et ne tient pas sur la RTX 5070 Ti 16 Go. ResEnc M (9 à 11 Go annoncés)
pourra être testé ensuite, mais il coûte davantage en calcul et ne remplace pas
la baseline standard.

Documentation officielle :

- [format des datasets](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) ;
- [presets ResEnc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md).

## Configuration obtenue par le planner

- Dataset : `Dataset501_iSeg2017`.
- Entrées : T1 (`0000`) et T2 (`0001`).
- Sorties : fond, LCR, matière grise, matière blanche.
- Architecture : `PlainConvUNet` avec convolutions 2D et 6 niveaux.
- Paramètres : 20 621 364.
- Patch : `160 x 128`.
- Batch size : `41`.
- Normalisation : Z-score T1 et T2 dans le masque non nul.
- Entraînement standard : 1000 epochs, 250 itérations train et 50 validation
  par epoch.
- Validation : 5-fold déterministe, soit 8 sujets train et 2 sujets validation
  par fold. Une coupe d'un sujet ne peut donc pas fuiter dans l'autre split.

Le fichier de plans contient aussi une proposition 3D créée automatiquement par
le planner global. Elle n'est ni préprocessée ni utilisée : les commandes du
projet sélectionnent exclusivement la configuration `2d`.

## Setup reproductible

Depuis la racine du projet :

```bash
scripts/setup_nnunet_baseline.sh
```

Le script :

1. extrait les archives si nécessaire ;
2. convertit Analyze vers NIfTI et remappe les labels `0/10/150/250` en
   `0/1/2/3` ;
3. vérifie l'intégrité du dataset ;
4. calcule le fingerprint et le plan ;
5. préprocesse uniquement la configuration 2D ;
6. écrit le split 5-fold reproductible.

Les artefacts sont placés sous `nnunet/` et ignorés par Git. Pour utiliser les
commandes nnU-Net manuellement :

```bash
source scripts/nnunet_env.sh
```

## Entraînement — à lancer explicitement

Fold 0, recommandé pour obtenir rapidement une première mesure :

```bash
scripts/train_nnunet_baseline.sh 0
```

Les autres folds utilisent la même commande avec `1`, `2`, `3` ou `4`. La RTX
5070 Ti est sélectionnée par défaut. Pour utiliser la RTX 3080 :

```bash
CUDA_VISIBLE_DEVICES=1 scripts/train_nnunet_baseline.sh 0
```

## Estimation sur la RTX 5070 Ti

Le micro-benchmark reproduit le réseau planifié, le batch, la précision mixte,
la loss Dice + entropie croisée, le backward, le clipping et l'optimiseur SGD,
avec des tenseurs synthétiques. Il ne lance pas l'entraînement des images et ne
crée aucun checkpoint.

Mesure obtenue avec PyTorch 2.14.0 CUDA 13.0 :

- étape train : environ 0,036 s ;
- étape validation : environ 0,008 s ;
- calcul GPU pur extrapolé : environ 2,6 h par fold ;
- estimation réaliste avec augmentation, transferts, I/O et checkpoints :
  **3,0 à 3,9 h par fold** ;
- 5 folds séquentiels : **15 à 20 h**.

Ce sont des ordres de grandeur. Pour la baseline de hackathon, commencer par le
fold 0. Les cinq folds servent ensuite à obtenir une évaluation croisée plus
robuste et, éventuellement, un ensemble.

Le benchmark peut être reproduit sans entraînement :

```bash
scripts/benchmark_nnunet_baseline.py --warmup 5 --steps 30
```
