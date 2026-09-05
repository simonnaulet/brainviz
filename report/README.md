# Rapport

Le rapport final est [`rapport.tex`](rapport.tex). Il décrit Rep-SliceMix-Net,
les baselines locales, les ablations et les limites du protocole expérimental.
La figure qualitative utilisée par le document est
[`assets/segmentation_examples.png`](assets/segmentation_examples.png).

Compilation depuis la racine du dépôt :

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=report report/rapport.tex
```

`latexmk` relance automatiquement LaTeX pour résoudre les références internes.
Le fichier
`rapport_entrainement.md` et les JSON présents dans ce dossier sont des archives
du prototype Compact U-Net antérieur au protocole commun. Leurs premiers scores
ne doivent pas être repris, car l'ancienne agrégation du Dice était biaisée.
