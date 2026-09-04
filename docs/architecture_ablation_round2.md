# Ablation architecture B4 — bottleneck plus profond

## Hypothèse

La première vague a montré qu'un noyau 7x7 au bottleneck est neutre à
légèrement défavorable et que la supervision profonde testée est défavorable.
B4 teste donc une seule nouvelle hypothèse : le modèle bénéficierait de plus de
capacité non linéaire à basse résolution, plutôt que d'un champ réceptif plus
large.

La seule différence par rapport à B0 est :

```toml
[model]
depths = [2, 2, 2, 2]
bottleneck_kernel_size = 3
deep_supervision = false
```

Un second `Block2D(192)` standard est ajouté à H/16. Les autres étages, la loss,
le sampler, le scheduler et la validation restent identiques.

| Variante | Paramètres entraînement | Paramètres déployés |
|---|---:|---:|
| B0 | 548 516 | 543 380 |
| B4 | 700 196 | 693 908 |

L'augmentation est de 151 680 paramètres à l'entraînement et 150 528 après
fusion. Le calcul supplémentaire reste concentré à H/16.

## Référence et règle de décision

La B0 de contrôle exécutée avec le code actuel atteint sur le fold 0 :

| Epoch | Dice tri-plan |
|---:|---:|
| 20 | 0,887147 |
| 30 | 0,908371 |

La référence historique à l'epoch 50 est `0,915374`; sa trajectoire à l'epoch
30 a été reproduite à 0,000071 près par la B0 actuelle.

Pour B4 à l'epoch 30 :

- au-dessus de `0,910371` : prometteur, prolonger à 50 ;
- entre `0,906371` et `0,910371` : ambigu, examiner les classes et la
  trajectoire avant de prolonger ;
- sous `0,906371` : rejeter.

Un résultat proche de B0 ne suffit pas à retenir B4 : le modèle déployé est
environ 28 % plus gros. Le gain doit être visible sur plusieurs classes et ne
pas reposer sur un seul sujet.

## Résultat du screening à 30 epochs

| Variante | Dice e20 | Dice e30 | CSF e30 | GM e30 | WM e30 | s/epoch moyen |
|---|---:|---:|---:|---:|---:|---:|
| B0 contrôle | 0,887147 | 0,908371 | 0,946078 | 0,902141 | 0,876895 | 45,76 |
| B4 | 0,843894 | 0,892482 | 0,937627 | 0,887895 | 0,851922 | 46,39 |

B4 est à `-0,015889` de B0 à l'epoch 30 et reste derrière sur les trois
classes. Son surcoût temporel mesuré n'est toutefois que de 1,4 %, car le bloc
supplémentaire travaille à H/16.

Le score seul déclencherait un rejet selon la règle initiale. La trajectoire
justifie néanmoins une prolongation exceptionnelle à 50 epochs : B4 gagne
`+0,048588` entre les epochs 20 et 30, contre `+0,021224` pour B0, et sa courbe
axiale reste nettement plus pentue entre les epochs 21 et 29. Cette prolongation
sert uniquement à déterminer si B4 converge plus tard; elle ne préjuge pas d'un
gain final.

À l'epoch 50, comparer à B0=`0,915374` :

- au-dessus de `0,917374` avec un gain réparti sur les classes : prometteur ;
- entre `0,913374` et `0,917374` : neutre, rejeter par défaut à cause des 28 %
  de paramètres supplémentaires ;
- sous `0,913374` : rejeter.

## Résultat à 50 epochs

| Variante | Dice moyen | CSF | GM | WM | Loss entraînement |
|---|---:|---:|---:|---:|---:|
| B0 | 0,915374 | 0,949743 | 0,910250 | 0,886129 | 0,123058 |
| B4 | 0,914276 | 0,948627 | 0,908889 | 0,885311 | 0,123012 |

B4 termine `-0,001098` sous B0 et reste derrière sur les trois classes : CSF
`-0,001116`, GM `-0,001361` et WM `-0,000818`. Les écarts de Dice moyen sur les
deux sujets sont `-0,001756` et `-0,000440`. La loss d'entraînement est
pratiquement identique à celle de B0, ce qui ne révèle aucun bénéfice de capacité
malgré 28 % de paramètres déployés supplémentaires.

B4 est donc rejetée et ne doit pas être prolongée à 200 epochs. La conclusion de
cette vague est que la capacité du bottleneck B0 n'est pas le facteur limitant
mis en évidence par ces expériences.

## Commandes

Smoke test optionnel :

```bash
brainviz-repslice train \
  --config configs/experiments/b4_deeper_bottleneck.toml \
  --fold 0 \
  --smoke
```

Screening comparable jusqu'à l'epoch 30 :

```bash
brainviz-repslice train \
  --config configs/experiments/b4_deeper_bottleneck.toml \
  --fold 0 \
  --stop-after-epoch 30
```

La limite d'exécution ne modifie pas l'horizon de 200 epochs du scheduler. Si
B4 est prometteur, reprendre le même run jusqu'à l'epoch 50 :

```bash
brainviz-repslice train \
  --config configs/experiments/b4_deeper_bottleneck.toml \
  --fold 0 \
  --resume artifacts/rep_slicemix/ablations/b4_deeper_bottleneck/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt \
  --stop-after-epoch 50
```

Les checkpoints B4 sont écrits dans un répertoire distinct et ne peuvent pas
écraser B0, B1 ou B2.
