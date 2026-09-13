"""Moteur de génération des projets de référence (manifestes YAML + couches de templates Jinja2).

Voir ``tools/scaffold/README.md`` pour le guide complet et ``tools/scaffold/build.py`` pour la CLI.
"""

from tools.scaffold.config import ProjectSpec, load_manifest
from tools.scaffold.engine import ProjectRenderer, RenderReport
from tools.scaffold.registry import FamilySpec, StackSpec, load_families, load_stacks

__all__ = [
    "FamilySpec",
    "ProjectRenderer",
    "ProjectSpec",
    "RenderReport",
    "StackSpec",
    "load_families",
    "load_manifest",
    "load_stacks",
]
