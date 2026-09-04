"""BrainViz: segmentation cérébrale 2D et 2.5D sur iSeg-2017."""

__version__ = "0.1.0"


def main() -> None:
    """Point d'entrée historique, conservé pour les environnements déjà synchronisés."""
    from brainviz.cli import main as cli_main

    cli_main()


__all__ = ["__version__", "main"]
