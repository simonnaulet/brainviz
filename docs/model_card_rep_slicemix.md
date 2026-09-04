# Model card — TriPlane Rep-SliceMix-Net

> Model card du run terminé `fold0-20260903-233524`, relevée le 4 septembre
> 2026. Le meilleur checkpoint tri-plan est celui de l'epoch 200.

## Résumé

TriPlane Rep-SliceMix-Net est un petit U-Net 2.5D conçu pour segmenter les IRM
cérébrales T1/T2 de nourrissons d'iSeg-2017 en quatre classes : fond, CSF, GM et
WM. Il traite cinq coupes à la fois, prédit uniquement la coupe centrale et
partage ses poids entre les plans axial, coronal et sagittal. Le décodeur est
entièrement 2D.

| Propriété | Valeur |
|---|---:|
| Paramètres pendant l'entraînement | 548 516 |
| Paramètres après reparamétrisation | 543 380 |
| Largeurs | `[24, 48, 96, 192]` |
| Profondeurs | `[2, 2, 2, 1]` |
| Entrée | `[B, 6, 5, H, W]` |
| Sortie | `[B, 4, H, W]` |
| Précision d'entraînement | AMP FP16 |

Les six canaux sont `T1`, `T2`, les coordonnées anatomiques normalisées `x/y/z`
et un masque cérébral dérivé des intensités. Les labels ne servent ni au crop ni
à la normalisation.

## Architecture

```text
5 coupes × (T1, T2, x, y, z, masque)                 plan ∈ {A, C, S}
                │                                             │
                ├─ branche centrale 2D 6→16 ─────────────── skip0
                │
                └─ stem 3D 6→24, stride spatial 2
                     │
                 2 × RepSliceMix(24), S=5 ─ FiLM ─ SlicePool ─ skip1
                     │  Down 24→48 : S 5→3, spatial /2
                 2 × RepSliceMix(48), S=3 ─ FiLM ─ SlicePool ─ skip2
                     │  Down 48→96 : S 3→1, spatial /2
                 2 × RepSliceMix(96), S=1 ─ FiLM ─ squeeze ─── skip3
                     │  Conv2d 96→192, spatial /2
                 Block2D(192)
                     │
                 décodeur 192→96→48→24→16 + skips
                     │
                 Conv2d 1×1 : 16→4
```

Un bloc `RepSliceMix` sépare le mélange spatial depthwise `(1×3×3)`, le mélange
inter-coupes depthwise `(3×1×1)` et le mélange de canaux par MLP `1×1×1`. Son
résultat est ajouté au résidu avec un `layer_scale` appris. Lorsque `S=1`, la
convolution inter-coupes est supprimée.

Pendant l'entraînement, chaque `RepDW` somme trois branches indépendamment
normalisées : kernel principal, kernel `1×1` et identité. `reparameterize()` les
fusionne exactement en une seule convolution depthwise avec biais pour le
déploiement. Les convolutions spatiale et inter-coupes restent séparées car un
GELU se trouve entre elles.

`FiLM` adapte les activations au plan anatomique avec un embedding plan-spécifique
gamma/bêta. `SlicePool` apprend, pour chaque plan et chaque canal, la pondération
des coupes avant de produire les skips 2D.

## Entraînement évalué

- Dataset : iSeg-2017, T1 + T2, volumes isotropes 1 mm.
- Fold 0 : 8 sujets train (`002–007`, `009`, `010`) et 2 sujets validation
  (`001`, `008`) ; séparation stricte par sujet.
- Batch 16, 250 itérations par epoch, objectif 200 epochs.
- AdamW, LR initial `3e-4`, weight decay `1e-4`, warmup 5 %, cosine decay.
- Loss : `0.6 SoftDice + 0.3 CE + 0.1 BoundaryCE`.
- EMA des poids avec decay `0.999` ; le checkpoint est sélectionné sur l'EMA.
- Validation volumique tri-plan toutes les 10 epochs.
- Augmentations : flips, rotation ±15°, échelle ±10 %, gamma, bruit gaussien et
  bias field. Les transformations géométriques sont communes aux images,
  coordonnées, masque et cible.

## Résultats

Checkpoint : `checkpoint_best_triplane.pt`, epoch **200**, 50 000 itérations.

| Classe | Dice ↑ | HD95 ↓ | ASD ↓ |
|---|---:|---:|---:|
| CSF | 0,95388 | 1,00 mm | 0,1316 mm |
| GM | 0,91830 | 1,00 mm | 0,3399 mm |
| WM | 0,89686 | 1,00 mm | 0,3853 mm |
| **Moyenne Dice** | **0,92302** | — | — |

Dice moyen par sujet de validation : `0,92123` et `0,92481`.

Ces métriques proviennent uniquement du fold 0 (`iseg_001`, `iseg_008`). Le
checkpoint local est
`artifacts/rep_slicemix/runs/fold0-20260903-233524/checkpoint_best_triplane.pt`.

## Inférence

Pour chaque plan, le réseau parcourt le volume avec des stacks de cinq coupes,
réoriente les probabilités dans le repère canonique, puis moyenne les trois
volumes avant l'argmax. Le mode axial seul est disponible pour réduire la
latence. L'export utilise les poids EMA et fusionne les branches RepDW.

## Limites et interprétation

- Ces résultats proviennent d'un seul fold et de seulement deux sujets de
  validation ; ils ne constituent ni une cross-validation complète ni un score
  officiel sur le test caché iSeg.
- Le score sélectionne indirectement un checkpoint sur ce fold : les résultats
  finaux devront être rapportés sur les cinq folds.
- La WM reste la classe la plus difficile.
- L'inférence tri-plan coûte environ trois passages par volume.
- La reparamétrisation simplifie le graphe déployé mais son gain de latence doit
  être mesuré sur le GPU cible.

## Représentation PyTorch complète

Représentation issue de `print(model)` pour la forme d'entraînement B0, simplifiée
uniquement en omettant les `ModuleList` vides introduites pour l'ablation B1 :

<details>
<summary>Déplier l'architecture PyTorch</summary>

```text
TriPlaneRepSliceMixNet(
  (center_branch): Sequential(
    (0): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): GELU(approximate='none')
  )
  (stem): Sequential(
    (0): Conv3d(6, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
    (1): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): GELU(approximate='none')
  )
  (stage1): Sequential(
    (0): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=24, bias=False)
        (main_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(24, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=24, bias=False)
        (point_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (slice): RepDW3d(
        (main_conv): Conv3d(24, 24, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=24, bias=False)
        (main_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(24, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=24, bias=False)
        (point_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(24, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(48, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
    (1): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=24, bias=False)
        (main_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(24, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=24, bias=False)
        (point_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (slice): RepDW3d(
        (main_conv): Conv3d(24, 24, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=24, bias=False)
        (main_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(24, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=24, bias=False)
        (point_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(24, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(48, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
  )
  (film1): PlaneFiLM(
    (embedding): Embedding(3, 48)
  )
  (pool1): SlicePool()
  (down1): Sequential(
    (0): Conv3d(24, 48, kernel_size=(3, 1, 1), stride=(1, 2, 2), bias=False)
    (1): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): GELU(approximate='none')
  )
  (stage2): Sequential(
    (0): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(48, 48, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=48, bias=False)
        (main_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(48, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=48, bias=False)
        (point_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (slice): RepDW3d(
        (main_conv): Conv3d(48, 48, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=48, bias=False)
        (main_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(48, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=48, bias=False)
        (point_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(48, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(96, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
    (1): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(48, 48, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=48, bias=False)
        (main_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(48, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=48, bias=False)
        (point_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (slice): RepDW3d(
        (main_conv): Conv3d(48, 48, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=48, bias=False)
        (main_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(48, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=48, bias=False)
        (point_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(48, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(96, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
  )
  (film2): PlaneFiLM(
    (embedding): Embedding(3, 96)
  )
  (pool2): SlicePool()
  (down2): Sequential(
    (0): Conv3d(48, 96, kernel_size=(3, 1, 1), stride=(1, 2, 2), bias=False)
    (1): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): GELU(approximate='none')
  )
  (stage3): Sequential(
    (0): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(96, 96, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=96, bias=False)
        (main_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(96, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=96, bias=False)
        (point_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(96, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
    (1): RepSliceMix(
      (spatial): RepDW3d(
        (main_conv): Conv3d(96, 96, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=96, bias=False)
        (main_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv3d(96, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=96, bias=False)
        (point_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv3d(96, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): GELU(approximate='none')
        (2): Conv3d(192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (norm): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
  )
  (film3): PlaneFiLM(
    (embedding): Embedding(3, 192)
  )
  (pool3): SlicePool()
  (down3): Sequential(
    (0): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    (2): GELU(approximate='none')
  )
  (bottle): Sequential(
    (0): Block2D(
      (spatial): RepDW2d(
        (main_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (main_bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (point_conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), groups=192, bias=False)
        (point_bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (identity_bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
      (mlp): Sequential(
        (0): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1))
        (1): GELU(approximate='none')
        (2): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      )
      (norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
    )
  )
  (decoder): ModuleList(
    (0): DecoderStage(
      (project): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (fuse): Sequential(
        (0): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): GELU(approximate='none')
      )
      (block): Block2D(
        (spatial): RepDW2d(
          (main_conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (main_bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (point_conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
          (point_bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (identity_bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        )
        (mlp): Sequential(
          (0): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1))
          (1): GELU(approximate='none')
          (2): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        )
        (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
    )
    (1): DecoderStage(
      (project): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1))
      (fuse): Sequential(
        (0): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): GELU(approximate='none')
      )
      (block): Block2D(
        (spatial): RepDW2d(
          (main_conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
          (main_bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (point_conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), groups=48, bias=False)
          (point_bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (identity_bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        )
        (mlp): Sequential(
          (0): Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1))
          (1): GELU(approximate='none')
          (2): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1))
        )
        (norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
    )
    (2): DecoderStage(
      (project): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1))
      (fuse): Sequential(
        (0): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): GELU(approximate='none')
      )
      (block): Block2D(
        (spatial): RepDW2d(
          (main_conv): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
          (main_bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (point_conv): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=24, bias=False)
          (point_bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (identity_bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        )
        (mlp): Sequential(
          (0): Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1))
          (1): GELU(approximate='none')
          (2): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1))
        )
        (norm): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
    )
    (3): DecoderStage(
      (project): Conv2d(24, 16, kernel_size=(1, 1), stride=(1, 1))
      (fuse): Sequential(
        (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): GELU(approximate='none')
      )
      (block): Block2D(
        (spatial): RepDW2d(
          (main_conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          (main_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (point_conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), groups=16, bias=False)
          (point_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
          (identity_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        )
        (mlp): Sequential(
          (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): GELU(approximate='none')
          (2): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        )
        (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      )
    )
  )
  (head): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))
)
```

</details>
