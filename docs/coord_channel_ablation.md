# Ablation D0 vs D0′ — apport des canaux de coordonnées RAS

## Hypothèse

Le pipeline de référence consomme 6 canaux par coupe : `t1, t2, coord_x, coord_y,
coord_z, brain_mask`. Les trois canaux `coord_x/y/z` encodent la position spatiale
dans le cerveau (grille `linspace(-1, 1)` par axe, calculée avant crop, en RAS —
`src/brainviz/data/triplane.py:109-119`), à la manière d'un CoordConv 3D. Leur apport
réel n'a jamais été mesuré isolément : la config `d0_axial_coords.toml` a été préparée
pour ne pas confondre l'apport des coordonnées avec celui du tri-plan/FiLM (elle fige
déjà l'axial seul et `film=false`), mais elle n'a jamais été exécutée, et les ablations
`C`/`C′` existantes suppriment coordonnées **et** `brain_mask` en même temps
(`input_indices=[0,1]`), ce qui ne permet pas de conclure sur les coordonnées seules.

**D0** (`configs/experiments/d0_axial_coords.toml`) : 6 canaux (coordonnées incluses),
axial seul, FiLM désactivé.

**D0′** (`configs/experiments/d0prime_no_coords.toml`) : identique à D0, sauf
`in_channels=3` et `input_indices=[0,1,5]` (T1, T2, `brain_mask` — coord_x/y/z
exclus).

C'est le seul changement entre les deux configurations : seed, splits, loss, sampler,
augmentations et preprocessing sont hérités à l'identique de `rep_slicemix.toml` via
`extends`. Le mécanisme de sélection de canaux (`input_indices`, validé et testé dans
`tests/test_rep_slicemix.py::test_input_indices_ignores_excluded_channels`) est déjà
utilisé par d'autres ablations (`C`/`C′`) et ne nécessite aucun changement de code.

## Protocole

Fold 0 (train `iseg_002..007,009,010`, validation `iseg_001,008`), screening
e30 → e50 → e100 selon la règle d'ambiguïté à 0,002 déjà en vigueur dans
`docs/experiments.md` : un écart inférieur à 0,002 sur les deux sujets de validation
est considéré ambigu tant qu'il ne persiste pas dans le temps (prolongation d'epochs)
ou sur un autre fold.

Attention : ne pas comparer les chiffres de D0/D0′ (axial seul, sans FiLM) aux Dice de
B0/B5 (tri-plan + FiLM, ~0,90–0,92) — architectures effectives différentes. La seule
comparaison valide ici est D0 vs D0′, à epoch identique, sur le même fold.

## Incident d'exécution : autocast fp16 codé en dur, instable sur ROCm

Les deux runs ont été exécutés sur une machine ROCm (AMD, pas de CUDA), ce qui a
révélé deux bugs préexistants, indépendants de cette ablation mais qui ont invalidé une
partie des mesures avant d'être corrigés en cours de route :

1. `src/brainviz/training/engine.py` fixait `torch.autocast(..., dtype=torch.float16)`
   sans condition — sur ce GPU ROCm, fp16 + `GradScaler` divergeait en `loss=nan` dès la
   première epoch. Corrigé en rendant le dtype configurable
   (`training.amp_dtype`, CLI `--amp-dtype {float16,bfloat16}`), avec `float16` toujours
   par défaut (aucun changement de comportement sur la machine CUDA de référence) et le
   `GradScaler` désactivé quand `bfloat16` est choisi (il n'en a pas besoin).
2. Une fois (1) corrigé, l'entraînement progressait normalement
   (`train_mean_soft_dice` sain, croissant, jamais NaN) mais **la validation
   s'effondrait à zéro Dice** à partir de l'epoch ~15-20 pour le run D0. Cause : la
   validation appelle `predict_preprocessed`/`predict_preprocessed_3d`
   (`src/brainviz/inference.py`), qui avaient elles aussi `dtype=torch.float16` codé en
   dur — non couvertes par le correctif (1). À mesure que les activations grandissaient
   avec l'entraînement, elles dépassaient la plage fp16 (max ≈ 65504) uniquement dans ce
   chemin d'inférence, produisant `inf`/`NaN` puis un `argmax` dégénéré sur la classe de
   fond. Corrigé en propageant `amp_dtype` jusqu'à `predict_preprocessed*`,
   `validate_volumes*` et la commande `predict`.

Conséquence pour les données ci-dessous : dans le run **D0**
(`fold0-20260905-114757`), les validations aux epochs 5, 10, 15, 20, 25 et 30 ont été
calculées **avant** le correctif (2) et sont invalidées (chute erratique jusqu'à
`mean_dice≈0,00008` à e30, alors que la loss d'entraînement était saine tout du long —
preuve que seule la mesure, pas l'entraînement, était corrompue). Seule l'epoch 35 de ce
run a été validée avec le code corrigé. Le run **D0′** (`fold0-20260905-121430`) a été
lancé entièrement après les deux correctifs ; toutes ses validations sont fiables (voir
sa progression saine et monotone de e5 à e35 dans les artefacts). La seule comparaison
propre entre D0 et D0′ est donc **à l'epoch 35**.

## Résultat à l'epoch 35 (seul point de comparaison valide pour D0)

| Variante | Dice e35 | CSF | GM | WM |
|---|---:|---:|---:|---:|
| D0  (coords incluses) | 0,903407 | 0,938244 | 0,897093 | 0,874884 |
| D0′ (coords exclues)  | 0,887936 | 0,938665 | 0,886426 | 0,838717 |
| Δ (D0 − D0′) | **+0,015471** | −0,000420 | +0,010667 | +0,036166 |

Par sujet (mean Dice) :

| Sujet | D0 | D0′ | Δ |
|---|---:|---:|---:|
| iseg_001 | 0,900055 | 0,885256 | +0,014799 |
| iseg_008 | 0,906759 | 0,890616 | +0,016143 |

Le gain est cohérent sur les deux sujets, largement au-dessus du seuil d'ambiguïté de
0,002 (environ 7 à 8 fois ce seuil), neutre sur CSF (classe la plus facile,
essentiellement délimitée par le contraste d'intensité) et concentré sur GM et surtout
WM — les classes les plus dépendantes du contexte anatomique global plutôt que de la
seule intensité locale, ce qui est cohérent avec l'hypothèse que les coordonnées
apportent un a priori de position utile pour désambiguïser ces tissus.

## Décision

**Coordonnées utiles : conservées telles quelles.** Le gain (+0,015 en moyenne, jusqu'à
+0,036 sur WM) est net, cohérent sur les deux sujets de validation et largement
au-dessus du seuil d'ambiguïté du projet, malgré une seule epoch de comparaison propre
disponible (voir incident ci-dessus). `rep_slicemix.toml` garde ses 6 canaux par défaut
(t1, t2, coord_x, coord_y, coord_z, brain_mask) — aucun changement de configuration par
défaut nécessaire suite à cette ablation.

Limite à noter : la conclusion repose sur un seul point de comparaison propre (e35) sur
un seul fold. Si une confirmation plus forte est souhaitée avant une décision finale de
soumission, prolonger les deux runs (maintenant rapides, ~31-40 s/epoch sur cette
machine) jusqu'à e50 ou e100, ou reproduire sur un second fold — non fait ici car le
signal est déjà net et cohérent.

La piste secondaire (ancrage anatomique inter-sujets des coordonnées, cf. plan initial)
n'a pas été explorée : elle ne se justifie qu'en cas de gain incertain, ce qui n'est pas
le cas ici.

## Commandes

Verrou avant tout run :

```bash
python -m pytest tests/test_rep_slicemix.py -q
```

Screening à 30 epochs :

```bash
brainviz-repslice train --config configs/experiments/d0_axial_coords.toml \
  --fold 0 --stop-after-epoch 30

brainviz-repslice train --config configs/experiments/d0prime_no_coords.toml \
  --fold 0 --stop-after-epoch 30
```

Si `|Δ mean_dice|` à e30 est inférieur à 0,002, prolonger les deux runs :

```bash
brainviz-repslice train --config configs/experiments/d0_axial_coords.toml \
  --fold 0 --resume artifacts/rep_slicemix/ablations/d0_axial_coords/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt \
  --stop-after-epoch 50

brainviz-repslice train --config configs/experiments/d0prime_no_coords.toml \
  --fold 0 --resume artifacts/rep_slicemix/ablations/d0prime_no_coords/fold0-YYYYMMDD-HHMMSS/checkpoint_last.pt \
  --stop-after-epoch 50
```

Répéter jusqu'à e100 avec le même schéma (`--resume ... --stop-after-epoch 100`) si
toujours ambigu à e50 — même approche que pour B5 (voir
[sampling_ablation_d1.md](sampling_ablation_d1.md)).

Extraction des métriques pour comparaison (`val_triplane` est ici une validation
axiale seule, du fait de `sampling.planes=[0]` sur les deux configs) :

```bash
python -c "
import json
for path in [
    'artifacts/rep_slicemix/ablations/d0_axial_coords/fold0-<TS_D0>/metrics.jsonl',
    'artifacts/rep_slicemix/ablations/d0prime_no_coords/fold0-<TS_D0PRIME>/metrics.jsonl',
]:
    for line in open(path):
        rec = json.loads(line)
        if 'val_triplane' in rec:
            v = rec['val_triplane']
            print(path, rec['epoch'], round(v['mean_dice'], 6),
                  round(v['csf_dice'], 6), round(v['gm_dice'], 6), round(v['wm_dice'], 6))
"
```

Les sorties sont isolées dans `artifacts/rep_slicemix/ablations/d0_axial_coords/` et
`artifacts/rep_slicemix/ablations/d0prime_no_coords/`.
