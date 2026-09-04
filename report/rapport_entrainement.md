# Rapport d'entraînement — iSeg-2017, CompactUNet

> **Archive exploratoire.** Ce rapport précède le protocole commun du projet :
> autre split, évaluation initialement agrégée par batch de coupes et autre
> machine. Son Dice 0,925 n'est pas comparable au Dice volumique des expériences
> Rep-SliceMix. La
> réévaluation contrôlée du Compact U-Net est consignée dans
> [`docs/experiments.md`](../docs/experiments.md) et atteint 0,89776 à l'epoch 90.

**Challenge :** MICCAI iSeg-2017 — segmentation de la substance blanche, substance grise et liquide céphalo-rachidien (LCR) sur IRM T1/T2 de nourrissons de 6 mois.
**Contrainte du hackathon :** qualité de segmentation *rapportée au nombre de paramètres* — viser un modèle frugal, pas le plus gros réseau possible.
**Métrique :** Dice moyen sur les 3 classes de premier plan (LCR, substance grise, substance blanche), fond exclu — c'est la métrique utilisée par le challenge iSeg-2017 lui-même.

---

## 1. Setup

- **Modèle :** `CompactUNet` — U-Net à convolutions *depthwise-separable* (au lieu de convolutions classiques), pour réduire drastiquement le nombre de paramètres à profondeur égale.
- **Données :** 10 sujets annotés (`dataset/train`), split par **sujet** (pas par tranche) pour éviter toute fuite entre train/validation — sujets 9 et 10 réservés à la validation, 8 sujets pour l'entraînement.
- **Entrée :** tranches axiales 2D, T1 + T2 + ratio T1/T2 normalisé (3 canaux).
- **Environnement :** RTX 3070 (GPU), entraînement exécuté nativement sous Windows (voir note technique en fin de rapport).

## 2. Résultats

### Run 1 — configuration de référence, 2 epochs (métrique biaisée)

| Paramètre | Valeur |
|---|---|
| base_channels | 16 |
| depth | 3 |
| Nombre de paramètres | 99 503 |

| Epoch | train_loss | val_loss | Dice LCR | Dice S. grise | Dice S. blanche | **Dice moyen (fg)** |
|---|---|---|---|---|---|---|
| 1 | 0.466 | 0.173 | 0.625 | 0.852 | 0.563 | 0.680 |
| 2 | 0.115 | 0.078 | 0.847 | 0.872 | 0.594 | **0.771** |

Objectif de ce run : valider que le pipeline (données → GPU → modèle → métrique) fonctionne de bout en bout. Résultat encourageant dès 2 epochs, mais la substance blanche reste nettement en retard (contraste le plus faible à cet âge, cf. contexte du challenge).

### Run 2 — même configuration, 20 epochs (métrique biaisée)

| Paramètre | Valeur |
|---|---|
| base_channels | 16 |
| depth | 3 |
| Nombre de paramètres | 99 503 |

**Résultat rapporté à l'époque : epoch 18 — Dice moyen (fg) = 0,925. Cette
valeur est surestimée et ne doit plus être citée comme un résultat valide.**

| Epoch | train_loss | val_loss | Dice LCR | Dice S. grise | Dice S. blanche | **Dice moyen (fg)** |
|---|---|---|---|---|---|---|
| 1 | 1.003 | 0.447 | 0.853 | 0.858 | 0.653 | 0.788 |
| 5 | 0.039 | 0.047 | 0.905 | 0.800 | 0.793 | 0.833 |
| 10 | 0.022 | 0.020 | 0.958 | 0.928 | 0.844 | 0.910 |
| 15 | 0.018 | 0.018 | 0.961 | 0.934 | 0.873 | 0.923 |
| **18** | **0.017** | **0.017** | **0.960** | **0.937** | **0.878** | **0.925** |
| 20 | 0.016 | 0.019 | 0.945 | 0.923 | 0.851 | 0.906 |

Le ratio d'efficacité de 9,30 points de Dice par million de paramètres calculé
à partir de cette valeur est invalide pour la même raison.

### Correctif du Dice

Le Dice était calculé indépendamment sur chaque batch de coupes, puis moyenné.
Lorsqu'une classe était absente de la cible et de la prédiction d'un batch, le
lissage lui attribuait un Dice égal à 1. Les nombreuses coupes sans CSF, GM ou WM
gonflaient donc artificiellement le score.

Le pipeline historique accumule désormais intersection et union sur le volume
complet de chaque sujet, calcule le Dice du sujet, puis effectue une moyenne
macro entre sujets. Un test de régression vérifie que des sujets de tailles
différentes ont le même poids.

Un run corrigé de 10 epochs, antérieur à cette dernière amélioration macro,
atteignait `0,835`. Il confirmait que `0,925` était une surestimation importante,
mais reste une expérience historique non comparable au protocole principal.

### Interprétation corrigée

- Les Dice des runs 1 et 2 ne permettent aucune conclusion quantitative sur la
  convergence ou l'early stopping.
- La WM reste qualitativement la classe la plus difficile.
- La comparaison contrôlée actuelle du Compact U-Net utilise le fold 0, la loss
  et la validation volumique Rep-SliceMix. Elle atteint `0,89776` à l'epoch 90;
  voir [`docs/experiments.md`](../docs/experiments.md).

## 3. Expérience de crop sur le pipeline historique

Une expérience CPU avec seed fixe a comparé l'ancien canvas `256×256` à un crop
image-based `160×160` avec une marge de quatre voxels :

| Variante | Coupes train/val | Meilleur Dice indicatif | Temps médian/epoch |
|---|---:|---:|---:|
| sans crop | 2 048 / 512 | 0,8224 | 110,9 s |
| crop marge 4 | 874 / 216 | 0,8409 | 14,25 s |

Cette expérience montre clairement l'intérêt d'éliminer le fond massif pour ce
loader. Elle ne mesure toutefois pas un gain GPU isolé : le crop diminue à la
fois la surface des images et le nombre de batches par epoch. Les scores sont
issus d'un seul run par condition et de l'ancienne agrégation globale des deux
sujets.

Le pipeline Rep-SliceMix applique déjà un crop plus robuste après
canonicalisation et utilise un nombre fixe d'itérations. Aucun code de crop du
loader historique n'est donc repris dans le pipeline principal.

## 4. Note technique — environnement d'exécution

Le premier essai sous WSL2 a provoqué un crash irrécupérable du service WSL (`Wsl/Service/E_UNEXPECTED`), lié à la couche de passthrough GPU (`dxgkrnl`) sous charge CUDA soutenue — aucun résultat exploitable n'en est sorti. L'entraînement a été relancé **nativement sous Windows** (venv Python + `uv`, PyTorch avec roue CUDA `cu128`), ce qui a résolu le problème et donné les résultats ci-dessus sans instabilité.

---
*Rapport historique mis à jour le 2026-09-04 ; résultats principaux dans
`docs/experiments.md`.*
