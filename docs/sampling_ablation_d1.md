# Ablation B5 — entraînement avec d=1 uniquement

## Hypothèse

La baseline B0 tire 75 % de stacks espacés de `d=1` et 25 % de stacks espacés
de `d=2`. L'inférence retenue utilise uniquement `d=1`, car l'ablation TTA a
montré que l'ajout de `d=2` n'améliore pas le score. B5 teste si supprimer ce
décalage entre entraînement et inférence améliore l'apprentissage.

La seule différence de configuration est :

```toml
[sampling]
d1_probability = 1.0
```

Architecture, nombre de paramètres, loss, augmentations, graine, scheduler et
validation restent identiques à B0. À chaque requête, les indices sont donc
toujours `[z-2, z-1, z, z+1, z+2]`.

## Références fold 0

| Epoch | Dice tri-plan B0 actuelle |
|---:|---:|
| 20 | 0,887147 |
| 30 | 0,908371 |
| 50 | 0,915374 (run historique, trajectoire reproduite) |
| 100 | 0,920975 (run historique) |

À l'epoch 30 :

- B5 au-dessus de `0,910371` : prometteur, prolonger à 50 ;
- entre `0,906371` et `0,910371` : ambigu, examiner les classes et la
  trajectoire ;
- sous `0,906371` : rejeter.

Comme B5 n'ajoute aucun paramètre ni coût d'inférence, un petit gain cohérent
est plus intéressant que pour B1 ou B4. Un écart inférieur à 0,002 sur ce seul
fold doit toutefois être confirmé à l'epoch 50 ou sur un autre fold.

## Résultat du screening à 30 epochs

| Variante | Dice e20 | Dice e30 | CSF e30 | GM e30 | WM e30 | Loss train e30 | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 contrôle | 0,887147 | 0,908371 | 0,946078 | 0,902141 | 0,876895 | 0,132213 | 45,76 |
| B5 d=1 | 0,889537 | 0,907323 | 0,946206 | 0,901402 | 0,874361 | 0,127771 | 45,54 |

B5 est meilleure à l'epoch 20 (`+0,002390`), puis légèrement derrière à
l'epoch 30 (`-0,001048`). À e30, CSF est inchangé (`+0,000128`), GM baisse de
`-0,000739` et WM de `-0,002534`. Les deux sujets baissent respectivement de
`-0,001655` et `-0,000442` en Dice moyen.

La loss d'entraînement plus basse alors que la validation n'augmente pas indique
que les stacks d=1 sont plus faciles à ajuster, sans gain de généralisation
démontré. Les 25 % de stacks d=2 de B0 pourraient agir comme une augmentation ou
une régularisation. Comme B5 n'a aucun surcoût et avait un avantage à e20, le run
est prolongé à 50 epochs avant décision.

À e50, comparer à B0=`0,915374` : un gain doit être accompagné d'une WM au moins
stable. Un résultat inférieur à environ `0,9144` rejettera B5; un résultat dans
la zone `0,9144–0,9164` sera considéré neutre et ne justifiera pas encore un
changement de valeur par défaut.

## Résultat à 50 epochs

| Variante | Dice moyen | CSF | GM | WM | s/epoch moyen |
|---|---:|---:|---:|---:|---:|
| B0 | 0,915374 | 0,949743 | 0,910250 | 0,886129 | 47,96 avant optimisations |
| B5 d=1 | 0,917169 | 0,952118 | 0,912084 | 0,887306 | 45,49 |

B5 gagne `+0,001795` en moyenne, avec CSF `+0,002375`, GM `+0,001834` et WM
`+0,001177`. Le gain apparaît sur les deux sujets : `+0,001088` et
`+0,002503`. C'est le premier changement de la campagne qui améliore toutes les
classes et tous les sujets sans ajouter de paramètres ni de latence.

B5 est donc promue pour une prolongation à 100 epochs sur le même fold. La
configuration par défaut ne sera modifiée qu'après persistance du gain à e100,
puis confirmation sur au moins un autre fold.

À e100, comparer à B0=`0,920975` : viser au moins `0,921975`, avec des gains
répartis. Un résultat entre `0,919975` et `0,921975` restera ambigu; sous
`0,919975`, B5 sera rejetée.

## Résultat à 100 epochs

| Variante | Dice moyen | CSF | GM | WM |
|---|---:|---:|---:|---:|
| B0 | 0,920975 | 0,952864 | 0,916295 | 0,893766 |
| B5 d=1 | 0,922420 | 0,954610 | 0,917568 | 0,895082 |

B5 gagne `+0,001445` en moyenne, avec CSF `+0,001746`, GM `+0,001273` et WM
`+0,001316`. Les deux sujets gagnent respectivement `+0,001903` et `+0,000988`.
Le gain n'est pas un pic isolé : entre les epochs 50 et 100, les validations
tri-plan B5 restent devant B0 de `+0,001297` à `+0,001795`.

B5 franchit donc le seuil de promotion fixé avant l'expérience. Elle devient la
candidate principale, sans encore remplacer B0 par défaut : la prochaine étape
est une comparaison appariée B0/B5 sur le fold 1, au moins jusqu'à l'epoch 50.
Cette confirmation inter-fold est plus informative qu'une prolongation immédiate
de B5 à 200 sur les mêmes deux sujets.

## Commandes

Screening à 30 epochs sans modifier l'horizon du scheduler :

```bash
brainviz-repslice train \
  --config configs/experiments/b5_d1_only.toml \
  --fold 0 \
  --stop-after-epoch 30
```

Si le résultat mérite une prolongation :

```bash
brainviz-repslice train \
  --config configs/experiments/b5_d1_only.toml \
  --fold 0 \
  --resume artifacts/rep_slicemix/ablations/b5_d1_only/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt \
  --stop-after-epoch 50
```

Les sorties sont isolées dans `artifacts/rep_slicemix/ablations/b5_d1_only/`.

Prolongation du run retenu jusqu'à l'epoch 100 :

```bash
brainviz-repslice train \
  --config configs/experiments/b5_d1_only.toml \
  --fold 0 \
  --resume artifacts/rep_slicemix/ablations/b5_d1_only/fold0-20260904-175827/checkpoint_last.pt \
  --stop-after-epoch 100
```
