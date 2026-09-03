#!/usr/bin/env python3
"""Extrait les archives iSeg-2017 dans un dataset organisé train/test.

Les archives sont attendues à plat dans `archives/` :

    archives/
      iSeg-2017-Training.zip   -> sujets 1-10  (T1, T2, label)
      iSeg-2017-Testing.zip    -> sujets 11-23 (T1, T2)

et sont réorganisées en :

    dataset/
      train/subject-1/{T1.hdr,T1.img,T2.hdr,T2.img,label.hdr,label.img}
      ...
      test/subject-11/{T1.hdr,T1.img,T2.hdr,T2.img}
      ...

Usage :
    python scripts/prepare_dataset.py [--archives-dir archives] [--output-dir dataset] [--force]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ARCHIVES = {
    "train": "iSeg-2017-Training.zip",
    "test": "iSeg-2017-Testing.zip",
}

# subject-<id>-<modalite>.<ext>, p.ex. "subject-9-T1.img" ou "subject-9-label.hdr"
FILENAME_RE = re.compile(
    r"^subject-(?P<subject>\d+)-(?P<modality>T1|T2|label)\.(?P<ext>hdr|img)$",
    re.IGNORECASE,
)


def is_junk(member: zipfile.ZipInfo) -> bool:
    """Écarte les répertoires et les métadonnées macOS (__MACOSX/, ._*)."""
    name = member.filename
    if member.is_dir():
        return True
    parts = Path(name).parts
    return "__MACOSX" in parts or Path(name).name.startswith("._")


def extract_split(archive: Path, split_dir: Path) -> int:
    """Extrait une archive vers `split_dir`, un dossier par sujet. Renvoie le nb de fichiers."""
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if is_junk(member):
                continue

            match = FILENAME_RE.match(Path(member.filename).name)
            if match is None:
                print(f"  ! fichier inattendu, ignoré : {member.filename}", file=sys.stderr)
                continue

            subject = int(match["subject"])
            modality = match["modality"]
            ext = match["ext"].lower()

            # Le .hdr Analyze retrouve son .img par nom de base : renommer les deux
            # de la même façon garde la paire cohérente.
            target = split_dir / f"subject-{subject}" / f"{modality}.{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1

    return written


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archives-dir",
        type=Path,
        default=project_root / "archives",
        help="dossier contenant les deux archives (défaut : <projet>/archives)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset",
        help="dossier de sortie (défaut : <projet>/dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="supprime le dossier de sortie existant au lieu de s'arrêter",
    )
    args = parser.parse_args()

    missing = [name for name in ARCHIVES.values() if not (args.archives_dir / name).is_file()]
    if missing:
        print(
            f"Archive(s) introuvable(s) dans {args.archives_dir} : {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    if args.output_dir.exists():
        if not args.force:
            print(
                f"{args.output_dir} existe déjà. Relancer avec --force pour l'écraser.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(args.output_dir)

    for split, archive_name in ARCHIVES.items():
        archive = args.archives_dir / archive_name
        split_dir = args.output_dir / split
        print(f"Extraction de {archive_name} -> {split_dir}")
        written = extract_split(archive, split_dir)
        subjects = sorted(int(p.name.split("-")[1]) for p in split_dir.iterdir() if p.is_dir())
        print(f"  {written} fichiers, {len(subjects)} sujets : {subjects}")

    print(f"\nDataset prêt dans {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
