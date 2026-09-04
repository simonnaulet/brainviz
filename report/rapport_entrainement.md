# Rapport historique — Compact U-Net

> **Archive exploratoire.** Ce rapport précède le protocole commun du projet :
> autre split, anciennes métriques et plusieurs machines. Les résultats
> comparables à Rep-SliceMix sont dans
> [`docs/experiments.md`](../docs/experiments.md).

## Protocole historique

- Dataset : iSeg-2017, sujets 1–8 en entraînement et 9–10 en validation.
- Entrée : coupes axiales, T1 + T2 + ratio T1/T2.
- Modèle : Compact U-Net depthwise-separable, 99 503 paramètres dans la version
  initiale.
- Loss : entropie croisée.
- Matériel selon les runs : RTX 3070 sous Windows ou CPU Linux.

Ce protocole diffère du pipeline actuel par le split, le preprocessing, la loss,
les augmentations et l'absence d'inférence tri-plan. Ses scores ne doivent pas
être comparés directement à ceux de Rep-SliceMix.

## Runs initiaux et métrique invalide

Les deux premiers runs calculaient un Dice séparément dans chaque batch, puis
moyennaient ces valeurs. Lorsqu'une classe était absente de la cible et de la
prédiction, le lissage attribuait un Dice égal à 1. Les nombreuses coupes sans
CSF, GM ou WM gonflaient donc artificiellement le résultat.

Le score `0,925` annoncé à l'epoch 18 et le ratio d'efficacité qui en découlait
sont invalides. Ils sont conservés dans les anciens JSON uniquement pour
documenter l'origine du problème.

## Correctif du Dice

Le pipeline historique accumule désormais intersection et union sur toutes les
coupes d'un sujet, calcule son Dice volumique, puis effectue une moyenne macro
entre sujets. Un test de régression vérifie que deux sujets de tailles
différentes conservent le même poids.

Un run corrigé antérieur à la moyenne macro atteignait `0,835` après 10 epochs.
Il confirme la surestimation initiale, mais ne constitue toujours pas une
comparaison contrôlée avec Rep-SliceMix.

## Expérience de crop historique

Le commit distant `eccf059` compare sur CPU l'ancien canvas `256×256` à un crop
image-based `160×160`, avec seed fixe et marge de quatre voxels :

| Variante | Coupes train/val | Meilleur Dice indicatif | Temps médian/epoch |
|---|---:|---:|---:|
| sans crop | 2 048 / 512 | 0,8224 | 110,9 s |
| crop marge 4 | 874 / 216 | 0,8409 | 14,25 s |

Le crop accélère fortement ce loader puisqu'il réduit simultanément la surface
des images et le nombre de batches par epoch. Il ne s'agit donc pas d'un facteur
×7,6 transposable au pipeline principal, dont le nombre d'itérations est fixe.
Les scores proviennent d'un seul run par condition et de l'ancienne agrégation
globale des deux sujets.

Les résumés bruts sont conservés dans `baseline_nocrop_summary.json` et
`crop_summary.json`. Les figures peuvent être régénérées avec `make_plots.py`.

## Comparaison contrôlée actuelle

Le Compact U-Net a ensuite été entraîné avec le preprocessing, le fold 0, le
sampler et la validation volumique du pipeline Rep-SliceMix :

```bash
brainviz-repslice train \
  --config configs/experiments/compact_unet_fair.toml \
  --fold 0
```

Il atteint `0,89776` à l'epoch 90 contre `0,92036` pour Rep-SliceMix B0 au même
point. Ce run contrôlé est celui à utiliser dans les présentations.

## Conclusion

- Le score historique `0,925` ne doit plus être cité.
- Le crop est une bonne amélioration du loader historique.
- Rep-SliceMix possède déjà un crop image-based plus robuste, après
  canonicalisation, avec restauration de la géométrie native.
- Les résultats de référence et leurs limites restent centralisés dans
  [`docs/experiments.md`](../docs/experiments.md).
