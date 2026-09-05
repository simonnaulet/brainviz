# Synthèse des expérimentations

Ce document est la source de vérité courte pour les résultats, les décisions et
les expériences restant à confirmer. Les rapports spécialisés conservent le
détail des protocoles et des trajectoires.

## Protocole commun

- Dataset : iSeg-2017, dix sujets annotés T1/T2.
- Fold 0 : train `002–007`, `009`, `010`; validation `001`, `008`.
- Mesure : Dice volumique sur CSF/GM/WM, moyenné par sujet puis par classe.
- Sélection : poids EMA au meilleur Dice tri-plan, toutes les dix epochs.
- Entraînement : 250 itérations/epoch, batch 16, FP16, AdamW et scheduler cosine
  prévu sur 200 epochs.
- Sauf indication contraire, toutes les variantes utilisent la même graine, les
  mêmes splits, augmentations, loss et données prétraitées.

Les différences inférieures à 0,002 sur deux sujets sont considérées comme
ambiguës tant qu'elles ne persistent pas dans le temps ou sur un autre fold.

## Résultats architecture et échantillonnage

| ID | Changement face à B0 | Paramètres déployés | e30 | e50 | e100 | e200 | Décision |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | référence, 75 % `d=1` | 543 380 | 0,90844 | 0,91537 | 0,92097 | **0,92302** | modèle terminé de référence |
| B0 contrôle | même code, boucle optimisée | 543 380 | 0,90837 | — | — | — | reproduction confirmée |
| B1 | bottleneck `7x7+3x3+1x1+id` | 551 060 | 0,90834 | 0,91420 | — | — | rejeté : neutre et plus gros |
| B2 | supervision profonde | 543 380 | 0,89723 | — | — | — | rejeté : régression nette |
| B3 | B1 + B2 | 551 060 | — | — | — | — | non lancé, car B2 est défavorable |
| B4 | deux blocs au bottleneck | 693 908 | 0,89248 | 0,91428 | — | — | rejeté : aucun gain pour +28 % |
| B5 | 100 % `d=1` | 543 380 | 0,90732 | **0,91717** | **0,92242** | — | prometteur, à confirmer fold 1 |

À l'epoch 100, B5 gagne `+0,00145` face à B0 : CSF `+0,00175`, GM
`+0,00127`, WM `+0,00132`. Le gain apparaît sur les deux sujets. B5 ne remplace
cependant pas encore B0 dans `configs/rep_slicemix.toml`, car il n'existe qu'un
seul fold de comparaison.

Artefacts locaux principaux :

| Run | Répertoire |
|---|---|
| B0 terminé | `artifacts/rep_slicemix/runs/fold0-20260903-233524` |
| B0 contrôle | `artifacts/rep_slicemix/runs/fold0-20260904-163532` |
| B1 | `artifacts/rep_slicemix/ablations/b1_large_kernel/fold0-20260904-151520` |
| B2 | `artifacts/rep_slicemix/ablations/b2_deep_supervision/fold0-20260904-153909` |
| B4 | `artifacts/rep_slicemix/ablations/b4_deeper_bottleneck/fold0-20260904-170442` |
| B5 | `artifacts/rep_slicemix/ablations/b5_d1_only/fold0-20260904-175827` |

Ces chemins sont volontairement ignorés par Git; `config.json`,
`environment.json` et `metrics.jsonl` rendent chaque run auditable localement.

## Comparaison Compact U-Net

Le Compact U-Net fusionné depuis la baseline externe a été réentraîné avec le
même preprocessing, les mêmes splits, le même sampler tri-plan et la même loss.

| Modèle | Paramètres | e30 | e50 | e90 |
|---|---:|---:|---:|---:|
| Compact U-Net | **99 578** | 0,88479 | 0,89236 | 0,89776 |
| Rep-SliceMix B0 | 548 516 entraînement | 0,90844 | 0,91537 | 0,92036 |

Le Compact U-Net est environ 5,5 fois plus petit, mais reste à `-0,02260` de B0
à l'epoch 90. Son run s'arrête à l'epoch 92 : il s'agit donc d'une baseline
d'efficacité utile, pas d'une comparaison finale à convergence égale.

## Ablation d'inférence

| Mode | Vues | Dice fold 0 | Temps indicatif/volume | Décision |
|---|---:|---:|---:|---|
| axial `d1` | 1 | 0,89966 | 0,957 s | aperçu rapide seulement |
| sagittal `d1` | 1 | 0,91389 | 0,651 s | meilleur mode une vue |
| coronal+sagittal `d1` | 2 | 0,92269 | 1,605 s | meilleur compromis |
| tri-plan `d1` | 3 | **0,92302** | 2,562 s | score de référence |
| tri-plan `d1+d2` | 6 | 0,92308 | 4,829 s | rejeté : gain négligeable |

Le multi-plan est déterminant, mais la troisième vue axiale n'ajoute que
`+0,00033` face à coronal+sagittal. `d=2` n'est pas un TTA rentable. Le rapport
complet est dans [tta_ablation.md](tta_ablation.md).

## Ablation des canaux de coordonnées

Le pipeline de référence ajoute déjà trois canaux de coordonnées RAS normalisées
(`coord_x/y/z`, type CoordConv) et un `brain_mask` à T1/T2, soit 6 canaux au total. Cet
apport n'avait jamais été mesuré isolément. La config `d0_axial_coords.toml` (axial
seul, FiLM désactivé, pour ne pas confondre l'apport des coordonnées avec celui du
tri-plan/FiLM) existait déjà mais n'avait jamais été exécutée; `d0prime_no_coords.toml`
a été ajoutée comme config sœur strictement identique sauf `in_channels=3` /
`input_indices=[0,1,5]` (T1, T2, `brain_mask`, coordonnées exclues).

Exécutée sur une machine ROCm, ce qui a révélé et fait corriger deux bugs préexistants
d'autocast fp16 codé en dur (entraînement puis validation), instable sur ce GPU —
détails dans [coord_channel_ablation.md](coord_channel_ablation.md). Seule l'epoch 35
est comparable pour D0 (epochs antérieures invalidées par le bug de validation avant
correctif) ; D0′ est fiable sur toute sa trajectoire.

| ID | Canaux | Dice e35 | CSF | GM | WM | Décision |
|---|---|---:|---:|---:|---:|---|
| D0  | 6 (avec coords) | 0,903407 | 0,938244 | 0,897093 | 0,874884 | référence |
| D0′ | 3 (sans coords) | 0,887936 | 0,938665 | 0,886426 | 0,838717 | rejeté : −0,015471 |

Les coordonnées apportent un gain de `+0,015471` en moyenne (jusqu'à `+0,036166` sur
WM), cohérent sur les deux sujets de validation et très au-dessus du seuil d'ambiguïté
de 0,002. Les 6 canaux restent le défaut de `rep_slicemix.toml`.

Détails, commandes et critère de décision dans
[coord_channel_ablation.md](coord_channel_ablation.md).

## Optimisations de vitesse

La boucle retenue combine six workers persistants, AdamW fusionné, EMA
`foreach`, journalisation différée et AMP FP16. Le benchmark apparié passe de
182,71 à 163,64 ms/it, soit **11,7 %** de gain à loss et VRAM identiques.

BF16, `cudnn.benchmark` et `torch.compile` ne sont pas retenus sur cette machine.
La validation utilise désormais un batch de 32; le micro-benchmark d'un sujet
tri-plan passe de 4,925 s (batch 16) à 2,217 s (batch 32). Voir
[training_performance.md](training_performance.md).

### Marge de crop 4 — test sans réentraînement

Les commits distants `9b0c6bf` (agrégation du Dice) et `eccf059` (crop du loader
historique) ont été évalués le 4 septembre 2026. Après cette revue, le checkpoint
B0 terminé a été évalué sur les deux sujets du fold 0 avec un preprocessing
temporaire à marge 4. Le modèle avait été entraîné avec la marge 8 habituelle.

| Marge | Dice moyen | Temps tri-plan pour 2 sujets |
|---:|---:|---:|
| **8 (référence)** | **0,92302** | 7,85 s |
| 4 | 0,92286 | **6,07 s** |

La différence de Dice vaut seulement `-0,00016`, pour environ 23 % de temps de
validation en moins sur cette mesure échauffée. La marge 4 est donc une option
d'inférence prometteuse, pas un nouveau défaut : elle n'a été testée que sur un
fold et ne doit pas modifier la reproductibilité du checkpoint B0.

Le preprocessing marge 4 est déjà supporté sans changement de code :

```bash
brainviz-repslice preprocess --input-dir dataset/train \
  --output-dir artifacts/rep_slicemix/preprocessed-margin4 --margin 4
```

## Décision de soumission

1. Geler l'architecture B0 et utiliser son checkpoint terminé à l'epoch 200
   comme résultat reproductible actuel.
2. Utiliser le tri-plan `d=1` pour le score principal, ou coronal+sagittal `d=1`
   si la latence est évaluée.
3. Si deux screenings supplémentaires deviennent possibles, comparer B0 et B5
   sur le fold 1 jusqu'à l'epoch 50 avant tout changement de configuration. Avec
   un seul run, conserver B0 : il manquerait le contrôle apparié du fold 1.
4. Ne pas relancer B1, B2, B3 ou B4 sans nouvelle hypothèse précise.
5. Présenter E, le U-Net 3D (2+1)D, comme expérience prévue mais **non exécutée**;
   ne pas lui attribuer de résultat.

Commandes d'une éventuelle confirmation appariée :

```bash
brainviz-repslice train --config configs/rep_slicemix.toml \
  --fold 1 --stop-after-epoch 50
brainviz-repslice train --config configs/experiments/b5_d1_only.toml \
  --fold 1 --stop-after-epoch 50
```

## Limites

- Les chiffres d'architecture proviennent d'un seul fold et de deux sujets de
  validation.
- Aucun score local n'est disponible pour les sujets test officiels 11–23, qui
  n'ont pas de labels.
- Les temps par volume de l'ablation TTA ne portent que sur deux volumes et
  dépendent de leur forme.
- B5 et le Compact U-Net ne sont pas allés jusqu'à 200 epochs.
- La comparaison 3D E et la cross-validation cinq folds restent à réaliser.

## Rapports détaillés

- [architecture_ablation_round1.md](architecture_ablation_round1.md) : B1–B3.
- [architecture_ablation_round2.md](architecture_ablation_round2.md) : B4.
- [sampling_ablation_d1.md](sampling_ablation_d1.md) : B5.
- [coord_channel_ablation.md](coord_channel_ablation.md) : apport des canaux de coordonnées (D0/D0′).
- [tta_ablation.md](tta_ablation.md) : plans et espacements.
- [training_performance.md](training_performance.md) : vitesse d'entraînement.
- [model_card_rep_slicemix.md](model_card_rep_slicemix.md) : modèle retenu.
