"""Génération programmatique des notebooks pédagogiques (nbformat).

Les notebooks ne sont pas des templates Jinja2 : un ``.ipynb`` est un JSON structuré, et le
générer avec du texte produit des fichiers fragiles. Chaque cellule est donc construite en
Python, ce qui garantit un notebook valide, exécutable et documenté.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.utils_notebooks import NotebookContext, build_context

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.config import ProjectSpec
    from tools.scaffold.registry import FamilySpec, StackSpec

#: Mapping ``family.notebook_builder`` -> module de construction.
BUILDERS = ("tabular", "text", "image", "platform")


def build_notebooks(
    spec: ProjectSpec,
    family: FamilySpec,
    stack: StackSpec,
    destination: Path,
) -> list[Path]:
    """Generate the six notebooks of a project.

    Args:
        spec: Validated manifest.
        family: Family definition.
        stack: Stack definition.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The list of written notebook paths.
    """
    from tools.scaffold.notebooks import tabular  # local import: évite une dépendance à nbformat au chargement

    destination.mkdir(parents=True, exist_ok=True)
    context = build_context(spec, family, stack)

    builder_name = family.notebook_builder if family.notebook_builder in BUILDERS else "tabular"
    if builder_name != "tabular":
        # Les autres modalités réutilisent le même squelette, adapté par la suite.
        builder_name = "tabular"
    builder = getattr(tabular, "build_all")
    return builder(context, destination)


__all__ = ["BUILDERS", "NotebookContext", "build_notebooks"]
