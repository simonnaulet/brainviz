# Ablation TTA : plans et espacements de coupes

Ce document conserve le protocole et les conclusions de l'ablation TTA du
TriPlane Rep-SliceMix-Net. Il doit servir de checklist lors des prochains folds,
des changements d'architecture et de la préparation d'une soumission.

## Expérience de référence

- Date : 4 septembre 2026.
- Checkpoint : `artifacts/rep_slicemix/runs/fold0-20260903-233524/checkpoint_best_triplane.pt`.
- Fold : 0.
- Entraînement : `iseg_002` à `iseg_007`, `iseg_009`, `iseg_010`.
- Hold-out : `iseg_001`, `iseg_008`.
- Métrique : Dice volumique calculé séparément par sujet et par classe,
  puis moyenné sur CSF, GM et WM.
- AMP fp16, batch d'inférence 16, RTX 5070 Ti.
- Une vue est un couple `(plan, espacement)` ; son coût est une inférence du
  volume complet.

Les sujets officiels iSeg 11–23 ne possèdent pas de labels dans le dataset local.
Ils permettent de produire une soumission et de mesurer la latence, mais pas de
calculer localement un Dice. Les ablations doivent donc être choisies sur les
hold-outs des folds, jamais sur un pseudo-score du test officiel.

## Résultats du fold 0

| Configuration | Vues | CSF | GM | WM | Dice moyen | s/volume |
|---|---:|---:|---:|---:|---:|---:|
| axial `d1` | 1 | 0,93990 | 0,89608 | 0,86301 | 0,89966 | 0,957 |
| axial `d2` | 1 | 0,93687 | 0,89160 | 0,85863 | 0,89570 | 0,983 |
| coronal `d1` | 1 | 0,94323 | 0,90654 | 0,88734 | 0,91237 | 0,954 |
| coronal `d2` | 1 | 0,94038 | 0,89896 | 0,87555 | 0,90496 | 0,651 |
| sagittal `d1` | 1 | 0,94810 | 0,90598 | 0,88757 | 0,91389 | 0,651 |
| sagittal `d2` | 1 | 0,94327 | 0,89901 | 0,88065 | 0,90764 | 0,633 |
| axial + coronal `d1` | 2 | 0,95088 | 0,91281 | 0,88726 | 0,91699 | 1,911 |
| axial + sagittal `d1` | 2 | 0,95105 | 0,91273 | 0,88930 | 0,91770 | 1,607 |
| **coronal + sagittal `d1`** | **2** | **0,95182** | **0,91681** | **0,89943** | **0,92269** | **1,605** |
| axial + coronal + sagittal `d1` | 3 | 0,95388 | 0,91830 | 0,89686 | **0,92302** | 2,562 |
| axial + coronal + sagittal `d2` | 3 | 0,95252 | 0,91368 | 0,89006 | 0,91875 | 2,268 |
| coronal + sagittal `d1+d2` | 4 | 0,95328 | 0,91738 | 0,89887 | **0,92318** | 2,889 |
| axial + coronal + sagittal `d1+d2` | 6 | 0,95491 | 0,91836 | 0,89597 | 0,92308 | 4,829 |

Les temps sont indicatifs : seulement deux volumes ont été chronométrés et
les formes diffèrent selon le sujet et le plan. Le nombre de vues est la mesure de
coût la plus robuste. Le rapport JSON complet, avec les 21 combinaisons et le
détail par sujet, est écrit dans
`artifacts/rep_slicemix/evaluations/fold0_tta.json`.

## Conclusions à conserver

1. Le multi-plan est utile : le tri-plan `d1` gagne `+0,02335` de Dice par rapport
   à l'axial `d1` seul.
2. L'axial est le plan individuel le moins performant sur ce fold.
3. Coronal + sagittal `d1` conserve presque tout le gain du tri-plan : seulement
   `-0,00033` de Dice pour deux vues au lieu de trois.
4. Ajouter l'axial à coronal + sagittal coûte une vue entière pour un gain trop
   faible sur ce fold.
5. `d2` seul est systématiquement moins bon que `d1`. En tri-plan, la baisse vaut
   `-0,00426` Dice.
6. Moyenner `d1+d2` sur les trois plans double le nombre de vues pour seulement
   `+0,00007` Dice. Ce TTA n'est pas rentable.
7. Le meilleur score brut, coronal + sagittal `d1+d2`, ne gagne que `+0,00016`
   par rapport au tri-plan `d1`. Cet écart n'est pas significatif avec deux sujets.

## Modes recommandés

| Mode | Plans | Espacements | Usage |
|---|---|---|---|
| qualité de référence | axial, coronal, sagittal | `d1` | score principal et comparaison historique |
| efficient | coronal, sagittal | `d1` | choix recommandé si la latence compte |
| rapide | sagittal | `d1` | prévisualisation ou contrainte forte de temps |
| expérimental | coronal, sagittal | `d1+d2` | à confirmer sur les autres folds avant usage |

Ne pas activer `d1+d2` par défaut. Une amélioration inférieure à `0,001` sur
deux sujets doit être considérée comme non concluante jusqu'à confirmation sur
les cinq folds.

## Reproduire l'ablation

Le script calcule les six vues élémentaires `3 plans × 2 espacements`, puis
recombine leurs probabilités sans refaire les passages GPU :

```bash
.venv/bin/python scripts/evaluate_repslicemix_tta.py \
  --checkpoint artifacts/rep_slicemix/runs/<run>/checkpoint_best_triplane.pt \
  --fold 0 \
  --device cuda \
  --output artifacts/rep_slicemix/evaluations/fold0_tta.json
```

Pour les folds suivants, modifier simultanément `--fold`, le checkpoint et le nom
du fichier de sortie. Ne jamais réutiliser le checkpoint d'un autre fold.

Exemples d'inférence sur un sujet prétraité :

```bash
# Qualité de référence : tri-plan d1
.venv/bin/brainviz-repslice predict <checkpoint.pt> <subject.npz> <prediction.nii.gz> \
  --planes all --slice-spacings 1 --device cuda

# Mode efficient : coronal + sagittal d1
.venv/bin/brainviz-repslice predict <checkpoint.pt> <subject.npz> <prediction.nii.gz> \
  --planes coronal,sagittal --slice-spacings 1 --device cuda

# Variante d1+d2, uniquement pour une ablation
.venv/bin/brainviz-repslice predict <checkpoint.pt> <subject.npz> <prediction.nii.gz> \
  --planes coronal,sagittal --slice-spacings 1,2 --device cuda
```

## Checklist des prochains tests

- Utiliser `checkpoint_best_triplane.pt` et ses poids EMA.
- Vérifier que le split du checkpoint correspond bien au fold évalué.
- Conserver `d1` comme référence.
- Calculer les métriques sur les volumes complets, sujet par sujet.
- Rapporter CSF, GM et WM, pas seulement leur moyenne.
- Rapporter le nombre de vues en plus de la latence mesurée.
- Comparer au minimum axial seul, chaque plan seul, coronal+sagittal et tri-plan.
- Confirmer toute petite différence sur les autres folds avant de changer le mode
  d'inférence par défaut.
- Pour le test officiel non annoté, produire les NIfTI puis utiliser le serveur de
  soumission ; un désaccord entre TTA n'est pas une mesure de qualité.
