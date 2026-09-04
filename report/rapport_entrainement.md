# Rapport d'entraînement — iSeg-2017, CompactUNet

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

### Run 1 — configuration de référence, 2 epochs (test rapide initial)

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

### Run 2 — même configuration, 20 epochs

| Paramètre | Valeur |
|---|---|
| base_channels | 16 |
| depth | 3 |
| Nombre de paramètres | 99 503 |

**Meilleur résultat : epoch 18 — Dice moyen (fg) = 0.925**

| Epoch | train_loss | val_loss | Dice LCR | Dice S. grise | Dice S. blanche | **Dice moyen (fg)** |
|---|---|---|---|---|---|---|
| 1 | 1.003 | 0.447 | 0.853 | 0.858 | 0.653 | 0.788 |
| 5 | 0.039 | 0.047 | 0.905 | 0.800 | 0.793 | 0.833 |
| 10 | 0.022 | 0.020 | 0.958 | 0.928 | 0.844 | 0.910 |
| 15 | 0.018 | 0.018 | 0.961 | 0.934 | 0.873 | 0.923 |
| **18** | **0.017** | **0.017** | **0.960** | **0.937** | **0.878** | **0.925** |
| 20 | 0.016 | 0.019 | 0.945 | 0.923 | 0.851 | 0.906 |

**Ratio d'efficacité : 9,30 points de Dice pour 1M de paramètres** (99 503 paramètres au total — un modèle très frugal).

### Observations clés

- **Gain massif entre 2 et 18 epochs** : Dice moyen 0.771 → 0.925 (+0.154). Le run de 2 epochs sous-estimait largement le potentiel du modèle — il n'avait simplement pas eu le temps de converger.
- **La substance blanche reste la classe la plus difficile** (0.878 au mieux, contre 0.960 pour le LCR) — cohérent avec le contexte du challenge : à 6 mois, le contraste substance grise/blanche est au plus bas de toute la première année de vie.
- **Léger surapprentissage après l'epoch 18** : le Dice de validation redescend légèrement (0.925 → 0.906 à l'epoch 20) pendant que la train_loss continue de baisser. Un arrêt anticipé (*early stopping*) autour de l'epoch 15-18 serait pertinent pour la suite.
- **Vitesse d'entraînement** : ~31s/epoch sur RTX 3070, soit ~10 minutes pour les 20 epochs — largement compatible avec des itérations rapides en hackathon.

## 3. Prochaines pistes testées / à tester

*(section à compléter avec les runs suivants : variante plus frugale `base_channels=8`, variante plus large `base_channels=24`, filtrage des tranches vides via `min_foreground_ratio`)*

## 4. Note technique — environnement d'exécution

Le premier essai sous WSL2 a provoqué un crash irrécupérable du service WSL (`Wsl/Service/E_UNEXPECTED`), lié à la couche de passthrough GPU (`dxgkrnl`) sous charge CUDA soutenue — aucun résultat exploitable n'en est sorti. L'entraînement a été relancé **nativement sous Windows** (venv Python + `uv`, PyTorch avec roue CUDA `cu128`), ce qui a résolu le problème et donné les résultats ci-dessus sans instabilité.

---
*Rapport généré le 2026-09-04 — à compléter au fil des runs suivants.*
