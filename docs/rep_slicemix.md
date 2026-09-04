# TriPlane Rep-SliceMix-Net

## État de l'implémentation

Le pipeline est indépendant du preprocessing nnU-Net, mais utilise exactement le
même split 5-fold par sujet dans `configs/splits_iseg.json`. Les sujets iSeg 11–23
ne sont jamais chargés par l'entraînement ou la sélection d'hyperparamètres.

Le modèle principal reçoit `[B,6,5,H,W]` : T1, T2, coordonnées RAS X/Y/Z et masque
cérébral dérivé des images, sur cinq coupes. Le masque évite notamment que le zéro
des coordonnées dans le padding soit interprété comme le centre anatomique.

Le troisième downsampling reste une convolution dense. Le modèle compte :

- 548 516 paramètres sous sa forme multibranche d'entraînement ;
- 543 380 paramètres après `reparameterize()`.

FiLM et SlicePool ne sont pas fusionnés : contrairement aux branches Conv+BN de
RepDW, leur emplacement dans le graphe résiduel ne permet pas une fusion exacte.

## Prétraitement

```bash
brainviz-repslice preprocess
```

Le traitement réoriente en RAS, vérifie T1/T2/label, construit le masque depuis
T1/T2, normalise dans ce masque, croppe avec une marge de huit voxels et padde à un
multiple de 16. Les coordonnées sont calculées avant le crop dans le repère commun.
Chaque fichier `artifacts/rep_slicemix/preprocessed/iseg_NNN.npz` conserve les
données et les métadonnées nécessaires au retour vers l'affine native.

La commande refuse d'écraser un preprocessing existant. Pour le reconstruire :

```bash
brainviz-repslice preprocess --force
```

`--unlabeled` permet de préparer explicitement un dossier destiné à l'inférence ;
il ne doit pas être utilisé comme source d'entraînement dans le protocole retenu.

## Entraînement

```bash
brainviz-repslice train --config configs/rep_slicemix.toml --fold 0
```

Chaque batch utilise un seul plan, avec des sujets paddés dynamiquement. Le sampler
utilise 75 % de stacks à `d=1`, 25 % à `d=2`, 90 % de coupes cérébrales et 10 % de
coupes de marge. Les transformations géométriques sont communes à toutes les
données ; les augmentations d'intensité sont propres à T1/T2 mais cohérentes entre
les cinq coupes. Les centres candidats sont précalculés et les huit sujets du fold
sont conservés en cache dans chacun des quatre workers. La graine d'augmentation
est portée par chaque requête : une reprise à une frontière d'epoch reproduit les
mêmes données indépendamment de l'ordonnancement des workers.

La loss est `0.6 Dice + 0.3 CE + 0.1 BoundaryCE`. BoundaryCE utilise un poids 3
près de GM/WM, 2 près des autres interfaces et 1 ailleurs. Le padding de collate
est exclu. Il n'y a pas d'early stopping : `checkpoint_best_triplane.pt` est choisi
sur le Dice volumique moyen CSF/GM/WM de l'EMA.

La validation axiale a lieu chaque epoch et la validation complète tous les dix
epochs. Cette dernière suit `sampling.planes` pour les modèles axiaux A–D0 et les
trois plans pour D. Les métriques et durées sont ajoutées à `metrics.jsonl` et
TensorBoard ; `environment.json` conserve les versions, le GPU et le commit Git.
Une reprise continue dans le même dossier :

```bash
brainviz-repslice train --fold 0 --resume artifacts/rep_slicemix/runs/<run>/checkpoint_last.pt
```

Après les cinq folds, calculer l'epoch médian puis réentraîner sur les dix sujets :

```bash
brainviz-repslice summarize-cv <checkpoint_best_fold0.pt> <checkpoint_best_fold1.pt> <checkpoint_best_fold2.pt> <checkpoint_best_fold3.pt> <checkpoint_best_fold4.pt>
brainviz-repslice train --fold all --epochs <epoch_median>
```

## Export et inférence

```bash
brainviz-repslice export checkpoint_best_triplane.pt model_deployed.pt
brainviz-repslice predict model_deployed.pt artifacts/rep_slicemix/preprocessed/iseg_001.npz prediction.nii.gz
```

`predict` moyenne les probabilités axiales, coronales et sagittales, force le fond
hors d'une marge sûre autour du masque image, retire crop/padding et restaure
l'affine et les labels iSeg `0/10/150/250`. La commande affiche aussi la latence
par volume. Le mode rapide est `--planes axial`.

```bash
brainviz-repslice evaluate prediction.nii.gz dataset/train/subject-1/label.img
```

L'évaluation rapporte Dice, HD95 et ASD pour CSF, GM et WM.

L'ablation des plans et des espacements `d1/d2`, ses conclusions et les commandes
à réutiliser sont conservées dans [tta_ablation.md](tta_ablation.md).

## Ablations préparées

Les configurations de `configs/experiments/` fournissent A, B, C′, C, D0, D et E.
D0 est ajouté pour ne pas confondre l'apport des coordonnées avec celui du
tri-plan/FiLM. Le modèle E utilise des blocs 3D depthwise (2+1)D et environ 580 k
paramètres. Son patch doit être choisi sur la machine cible :

```bash
brainviz-repslice probe-3d --config configs/experiments/e_unet_3d.toml
```

Cette commande teste 64³, 80³, 96³ et le plus grand crop non cubique du dataset
avec une vraie passe avant/arrière AdamW. Une taille non cubique `[D,H,W]` peut être
recopiée dans `patch3d.patch_size` si le volume presque complet tient en mémoire.
La campagne recommandée est un screening sur le fold 0, puis cinq folds uniquement
pour les finalistes, après discussion du budget de calcul.

Pour profiler les FLOPs du modèle déployé sur une taille donnée :

```bash
brainviz-repslice inspect --flops --height 160 --width 128
```

## Vérifications

```bash
uv run --group dev pytest -q
```

Les tests couvrent notamment les trois permutations de plans, le padding dynamique,
l'indépendance du preprocessing vis-à-vis du label, les losses 2D/3D et l'équivalence
de reparamétrisation de chaque RepDW et du modèle complet avec une tolérance `1e-4`.
