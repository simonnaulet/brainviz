# Optimisation des performances d'entraînement

Ce document conserve la campagne de micro-benchmarks effectuée le 4 septembre
2026 sur la RTX 5070 Ti. Les essais utilisent de vrais batches augmentés, la loss
complète, le backward, le gradient clipping, AdamW et l'EMA du Rep-SliceMix-Net.

Les changements retenus ne modifient ni l'architecture, ni les données tirées, ni
la loss. Ils visent uniquement l'overhead de la boucle d'entraînement.

## Pipeline de données

Mesure de 50 batches après 10 batches d'échauffement :

| Workers | ms/batch CPU |
|---:|---:|
| 0 | 346,84 |
| 2 | 238,54 |
| 4 | 169,54 |
| 6 | **162,19** |

Six workers sont conservés. Le gain face à quatre workers reste modeste, environ
4 %, mais il se retrouve dans la boucle GPU complète. Augmenter davantage n'est
pas recommandé sur le Ryzen 5600X sans nouvelle mesure.

## Optimisations isolées

Mesure courte de 100 itérations, FP16, batch 16 :

| Variante | ms/it | Gain approximatif |
|---|---:|---:|
| baseline | 189,38 | — |
| métriques différées | 185,37 | 2,2 % |
| EMA `foreach` | 179,65 | 5,4 % |
| AdamW fusionné | 175,16 | 8,1 % |
| cuDNN benchmark | 186,65 | 1,5 % |
| combinaison eager, 4 workers | 173,87 | 8,9 % |
| combinaison eager, 6 workers | 167,99 | 12,7 % |
| combinaison BF16, 6 workers | 240,54 | **-27,0 %** |

Les gains isolés ne sont pas additifs, car plusieurs réduisent le même temps
d'attente CPU ou le nombre de petits lancements CUDA.

## Mesure finale des options retenues

Une seconde mesure appariée utilise 20 itérations d'échauffement et 200
itérations chronométrées :

| Configuration | ms/it | Loss moyenne | VRAM allouée |
|---|---:|---:|---:|
| boucle d'origine | 182,71 | 0,60264 | 1 975 MiB |
| boucle retenue | **163,64** | **0,60264** | 1 975 MiB |

Le gain apparié est de **11,7 %**. Sur 50 000 itérations, cela représente
environ 16 minutes économisées sur la boucle chaude, hors validation et
checkpoints.

Rapports bruts ignorés par Git :

- `artifacts/rep_slicemix/performance_benchmark.json` ;
- `artifacts/rep_slicemix/performance_benchmark_final.json`.

## Changements retenus

1. Les valeurs CUDA des losses restent sur GPU pendant l'epoch. Elles ne sont
   converties en scalaires Python et envoyées à TensorBoard que tous les
   `log_every` steps.
2. L'EMA utilise une opération `torch._foreach_lerp_` pour tous les tenseurs
   flottants et une `torch._foreach_copy_` pour les buffers entiers.
3. AdamW utilise son implémentation CUDA fusionnée. Lors d'une reprise, le réglage
   est réappliqué après le chargement de l'état d'un ancien optimiseur.
4. Le DataLoader utilise six workers persistants, avec `pin_memory` et transferts
   non bloquants déjà présents.
5. FP16 et le `GradScaler` sont conservés.

Les options actives sont enregistrées dans `configs/rep_slicemix.toml` :

```toml
[data]
num_workers = 6

[training]
amp = true
fused_adamw = true
cudnn_benchmark = false
log_every = 10
```

## Options rejetées

### BF16

BF16 a produit une loss comparable, mais environ 27 % de débit en moins sur ce
modèle convolutionnel. FP16 reste donc le choix par défaut.

### cuDNN benchmark

Le gain stabilisé n'était que d'environ 1,5 %. Dans un processus neuf, la
recherche d'algorithmes a ajouté environ 20 secondes à un smoke test de deux
itérations. Les tailles spatiales variant selon le sujet et le plan, l'option est
laissée configurable dans le moteur mais désactivée dans la configuration.

### `torch.compile`

La variante `mode="reduce-overhead", dynamic=True` a rencontré de longues
recompilations sur les formes variables, puis WSL a crashé pendant cet essai. La
cause directe ne peut pas être prouvée après redémarrage, mais la coïncidence est
forte. Cette variante a été retirée du benchmark et ne doit pas être relancée sur
la configuration actuelle.

Une future tentative ne serait raisonnable qu'après avoir imposé des formes
statiques, dans un environnement isolé et avec un seul processus de test.

### Augmentations GPU et padding global statique

Avec six workers, le pipeline CPU atteint environ 162 ms/batch, proche du temps
GPU complet. Déplacer les augmentations sur GPU ajouterait de la complexité et
pourrait modifier leurs distributions aléatoires. Un padding global augmenterait
le nombre de pixels traités pour gagner seulement sur la compilation, qui est
désormais désactivée. Ces changements ne sont pas retenus.

## Reproduire la mesure sûre

Le script ne contient plus de variante `torch.compile` :

```bash
.venv/bin/python scripts/benchmark_training_performance.py \
  --warmup 20 \
  --steps 200 \
  --skip-loader \
  --variants baseline optimized_eager_w6_no_benchmark \
  --output artifacts/rep_slicemix/performance_benchmark_final.json
```
