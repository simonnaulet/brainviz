# Ablation architecture — grand noyau et supervision profonde

## Objectif

Cette première vague teste deux changements compatibles, peu coûteux et
attribuables séparément. La baseline B0 reste la configuration
`configs/rep_slicemix.toml`, dont le fold 0 a atteint un Dice tri-plan de
`0,923015` à l'epoch 200.

| Variante | Grand noyau | Supervision profonde | Configuration |
|---|---:|---:|---|
| B0 | non | non | `configs/rep_slicemix.toml` |
| B1 | oui | non | `configs/experiments/b1_large_kernel.toml` |
| B2 | non | oui | `configs/experiments/b2_deep_supervision.toml` |
| B3 | oui | oui | `configs/experiments/b3_large_kernel_deep_supervision.toml` |

B1 et B2 doivent être évaluées avant B3. B3 ne sera lancée que si les deux
changements sont au moins neutres, afin de ne pas masquer l'origine d'un gain
ou d'une régression.

## B1 — grand noyau reparamétrable au bottleneck

Le RepDW 2D du bottleneck passe de `3x3 + 1x1 + identité` à
`7x7 + 3x3 + 1x1 + identité`. Chaque branche possède sa BatchNorm pendant
l'entraînement. `reparameterize()` centre et fusionne les quatre branches dans
une seule convolution depthwise 7x7 avec biais.

Le changement reste limité à la résolution H/16. Il augmente donc le champ
réceptif avec un faible coût en calcul et environ 7 700 paramètres déployés
supplémentaires. Les blocs du décodeur et des étages haute résolution restent
en 3x3.

## B2 — supervision profonde

Pendant l'entraînement uniquement, deux têtes 1x1 produisent des logits à H/2
et H/4. Les sorties sont pondérées selon `1, 0.5, 0.25`, puis les poids sont
normalisés. La sortie principale conserve la loss complète
`Dice + CE + BoundaryCE`; les sorties auxiliaires utilisent Dice + CE, car la
largeur de frontière en pixels ne conserve pas sa signification après
downsampling.

En mode évaluation, seul le tenseur pleine résolution est calculé. Lors de la
reparamétrisation/export, les têtes auxiliaires sont retirées : elles
n'ajoutent aucun paramètre ni aucune latence au modèle déployé.

Les métriques `main_loss`, `aux_1_loss`, `aux_2_loss`,
`aux_1_soft_dice` et `aux_2_soft_dice` sont écrites dans `metrics.jsonl` et
TensorBoard avec les métriques existantes.

## Protocole rapide sans modifier le scheduler

La courbe B0 du fold 0 donne les repères suivants :

| Epoch | Dice tri-plan B0 |
|---:|---:|
| 10 | 0,732148 |
| 20 | 0,886879 |
| 30 | 0,908442 |
| 50 | 0,915374 |
| 100 | 0,920975 |
| 200 | 0,923015 |

Toutes les variantes gardent `training.epochs = 200`. L'option
`--stop-after-epoch` limite seulement le lancement courant : elle ne comprime
ni le warmup ni le cosine schedule. Le checkpoint de l'epoch 30 peut ainsi être
repris jusqu'à 50, puis jusqu'à 200.

1. Faire un smoke test de B1 et B2.
2. Entraîner B1 et B2 jusqu'à l'epoch 30 sur le fold 0.
3. Rejeter une régression claire; prolonger un résultat prometteur ou ambigu à
   50 epochs.
4. Lancer B3 jusqu'à 30 epochs seulement si B1 et B2 sont non négatives.
5. Reprendre le meilleur candidat jusqu'à 200 epochs, puis confirmer sur
   d'autres folds.

Comparer en priorité les validations tri-plan des epochs 10, 20 et 30, le Dice
par classe et le temps par epoch. Un écart absolu inférieur à 0,002 sur les deux
sujets du fold 0 est ambigu et ne justifie pas une conclusion avant l'epoch 50
ou un second fold.

## Résultats

| Variante | Dice e20 | Dice e30 | Dice e50 | CSF/GM/WM à la dernière epoch | s/epoch moyen e1–30 | Paramètres déployés | Décision |
|---|---:|---:|---:|---|---:|---:|---|
| B0 | 0,886879 | 0,908442 | 0,915374 | 0,949743 / 0,910250 / 0,886129 | 47,96 avant optimisations | 543 380 | référence historique |
| B1 | 0,889900 | 0,908344 | 0,914202 | 0,949018 / 0,908968 / 0,884620 | 44,57 à e30 | 551 060 | neutre à légèrement défavorable |
| B2 | 0,879943 | 0,897228 | — | 0,945988 / 0,894533 / 0,851162 | 45,68 | 543 380 | rejetée |
| B3 | — | — | — | — | — | 551 060 | ne pas lancer à ce stade |
| B0 contrôle actuel | 0,887147 | 0,908371 | — | 0,946078 / 0,902141 / 0,876895 | 45,76 | 543 380 | confirme B0 historique |

### Analyse après screening et prolongation — 4 septembre 2026

B1 était à `-0,000098` de B0 sur le Dice moyen à l'epoch 30. Elle échangeait
alors environ 0,002 point de CSF et de GM contre `+0,003627` de WM, ce qui a
justifié sa prolongation jusqu'à l'epoch 50.

À l'epoch 50, B1 atteint `0,914202` contre `0,915374` pour B0, soit
`-0,001172`. Elle est désormais légèrement derrière sur chaque classe :
CSF `-0,000725`, GM `-0,001282` et WM `-0,001509`. Sur les deux sujets, les
écarts de Dice moyen sont respectivement `-0,002437` et `+0,000093`. Le gain WM
précoce n'a donc pas persisté. B1 est classée neutre à légèrement défavorable et
ne doit pas être prolongée à 200 epochs en l'état.

B2 est à `-0,011214` de B0, avec une baisse WM de `-0,025739`. La régression est
présente sur les deux sujets et tout au long du screening (epochs 10, 20 et 30).
La supervision profonde avec les poids normalisés `1/0.5/0.25` est donc rejetée.
B3 ne doit pas être lancée, car elle incorporerait un changement déjà nettement
défavorable.

La référence B0 historique a été entraînée avant les optimisations de boucle :
4 workers et AdamW non fusionné, contre 6 workers et AdamW fusionné pour B1/B2.
Le sampler reste reproductible, mais une différence de l'ordre de `1e-3` ne peut
pas être attribuée au grand noyau avec certitude dans ces conditions. B1 n'ayant
pas dépassé B0 à l'epoch 50, elle n'est pas prolongée. Avant la prochaine vague
d'architecture, produire une B0 de contrôle avec le code courant jusqu'à l'epoch
30. Cette référence sera réutilisable; ne la prolonger à 50 que si son score à 30
diffère de plus de 0,001 de la référence historique `0,908442`.

Le contrôle B0 actuel a ensuite obtenu `0,908371` à l'epoch 30, seulement
`-0,000071` sous la référence historique. Les écarts par classe sont tous
inférieurs à 0,0002. Il n'est donc pas nécessaire de le prolonger à 50 : la
reproductibilité de B0 est confirmée et B1 est définitivement classée neutre à
légèrement défavorable.

Ne pas utiliser `--epochs 30` pour ce screening : cette option redéfinit
l'horizon du scheduler et produit une expérience différente.

## Commandes

```bash
# Vérifications rapides
brainviz-repslice inspect --config configs/experiments/b1_large_kernel.toml
brainviz-repslice train --config configs/experiments/b1_large_kernel.toml --fold 0 --smoke
brainviz-repslice train --config configs/experiments/b2_deep_supervision.toml --fold 0 --smoke

# Screening jusqu'à l'epoch 30
brainviz-repslice train --config configs/experiments/b1_large_kernel.toml --fold 0 --stop-after-epoch 30
brainviz-repslice train --config configs/experiments/b2_deep_supervision.toml --fold 0 --stop-after-epoch 30

# Prolongation d'un run existant jusqu'à l'epoch 50
brainviz-repslice train --config configs/experiments/b1_large_kernel.toml --fold 0 \
  --resume artifacts/rep_slicemix/ablations/b1_large_kernel/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt \
  --stop-after-epoch 50

# Puis entraînement jusqu'à l'horizon prévu de 200 epochs
brainviz-repslice train --config configs/experiments/b1_large_kernel.toml --fold 0 \
  --resume artifacts/rep_slicemix/ablations/b1_large_kernel/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt
```

Remplacer la configuration et le répertoire par B2 ou B3 selon la variante.
Chaque variante écrit dans son propre répertoire et ne peut donc pas écraser
les runs B0 existants.
