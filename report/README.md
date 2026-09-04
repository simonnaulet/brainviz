# Archive du prototype Compact U-Net

Ce dossier conserve le rapport et les figures produits avant l'adoption du
protocole commun. Ils utilisent un split `subject-9/subject-10`, des coupes
axiales et, pour les premiers runs, un Dice biaisé par une agrégation par batch.
Ils ne doivent pas servir à comparer directement les architectures du pipeline
actuel.

Le calcul historique a été corrigé dans `brainviz.train` : accumulation sur le
volume de chaque sujet, puis moyenne macro entre sujets. Les valeurs déjà
produites avec l'ancienne formule restent invalides et ne sont pas réécrites.

La comparaison contrôlée utilise le même fold, le même preprocessing, la même
loss et la même validation volumique que Rep-SliceMix :

```bash
brainviz-repslice train \
  --config configs/experiments/compact_unet_fair.toml \
  --fold 0
```

Les résultats comparables et leurs limites sont dans
[`docs/experiments.md`](../docs/experiments.md).
