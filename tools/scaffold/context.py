"""Build the Jinja rendering context (manifest + registries + template helpers)."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from datetime import date
from typing import Any

import yaml

from tools.scaffold.config import ColumnSpec, ProjectSpec
from tools.scaffold.pandera_helpers import (
    dtype_map_literal,
    pandera_annotation,
    pandera_columns_block,
    pandera_field,
)
from tools.scaffold.registry import FamilySpec, StackSpec

#: Common dependencies of every generated project (config, validation, quality, figures).
#:
#: ``matplotlib`` et ``seaborn`` sont des dépendances **runtime** et non de développement :
#: ``src/visualization/plots.py`` les importe au chargement du module, donc un environnement
#: installé avec le seul ``requirements.txt`` échouerait sur ``make evaluate`` (le rapport écrit
#: les figures). Elles restent listées côté dev pour que ``requirements-dev.txt`` soit autonome.
CORE_DEPENDENCIES: list[str] = [
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "pandera[pandas]>=0.20",
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "loguru>=0.7",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

#: Common development dependencies of every generated project.
DEV_DEPENDENCIES: list[str] = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "nbformat>=5.9",
    "nbclient>=0.10",
    "ipykernel>=6.29",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]


def py(value: Any) -> str:
    """Render a Python literal usable inside generated source code.

    Args:
        value: Any JSON-compatible Python value.

    Returns:
        Its ``repr``, with tuples converted to lists.
    """
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, dict):
        items = ", ".join(f"{py(k)}: {py(v)}" for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, (list, set)):
        items = ", ".join(py(v) for v in value)
        return "[" + items + "]" if isinstance(value, list) else "{" + items + "}"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def yaml_dump(value: Any, indent: int = 0) -> str:
    """Dump a value as YAML (used to render ``conf/*.yaml`` blocks).

    Args:
        value: Value to serialise.
        indent: Number of leading spaces for continuation lines.

    Returns:
        The YAML text without the trailing newline.
    """
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, default_flow_style=False)
    text = text.rstrip("\n")
    if indent:
        padding = " " * indent
        text = "\n".join(padding + line if line else line for line in text.splitlines())
    return text


def camel(value: str) -> str:
    """Convert ``snake_or-kebab_case`` into ``CamelCase``.

    Args:
        value: Identifier to convert.

    Returns:
        The CamelCase identifier.
    """
    parts = value.replace("-", "_").split("_")
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def const(value: str) -> str:
    """Convert an identifier into ``UPPER_SNAKE_CASE``.

    Args:
        value: Identifier to convert.

    Returns:
        The constant-style identifier.
    """
    return value.replace("-", "_").upper()


def wrap(text: str, width: int = 88, indent: int = 0, prefix: str = "") -> str:
    """Wrap a long text, useful to render docstrings and README paragraphs.

    Args:
        text: Text to wrap.
        width: Maximum line width.
        indent: Indentation applied to every line.
        prefix: Prefix prepended to the first line (e.g. ``"> "``).

    Returns:
        The wrapped text.
    """
    padding = " " * indent
    wrapped = textwrap.fill(
        text.strip(),
        width=width - indent,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )
    return "\n".join(padding + line if line else line for line in wrapped.splitlines())


def py_string_tuple(
    items: Iterable[str], indent: int = 4, width: int = 96, suffix: str = ","
) -> str:
    """Render long strings as Python literals split by implicit concatenation.

    Les textes métier (recommandations, hypothèses) viennent du manifeste et dépassent
    souvent la limite de longueur du linter. Plutôt que d'émettre une ligne de 140
    caractères, on découpe chaque texte en fragments concaténés implicitement par Python :
    le code généré reste conforme à ``ruff`` (E501) et parfaitement lisible.

    Args:
        items: Texts to render.
        indent: Indentation (in spaces) applied to every line.
        width: Maximum line width, quotes and suffix included.
        suffix: Appended after the last fragment of each item (``","`` for a tuple).

    Returns:
        The rendered Python source lines.
    """
    padding = " " * indent
    lines: list[str] = []
    for item in items:
        text = str(item).replace('"', "'").strip()
        budget = max(width - indent - len(suffix) - 2, 20)
        chunks = textwrap.wrap(text, width=budget) or ['""']
        last = len(chunks) - 1
        for index, chunk in enumerate(chunks):
            # L'espace de coupure est rétabli : la concaténation implicite ne l'ajoute pas.
            body = chunk if index == last else f"{chunk} "
            tail = suffix if index == last else ""
            lines.append(f'{padding}"{body}"{tail}')
    return "\n".join(lines)


def bullet_list(items: list[str], indent: int = 0, marker: str = "-") -> str:
    """Render a Markdown bullet list.

    Args:
        items: List items.
        indent: Indentation in spaces.
        marker: Bullet marker.

    Returns:
        The Markdown list.
    """
    padding = " " * indent
    return "\n".join(f"{padding}{marker} {item}" for item in items)


def numbered_list(items: list[str], indent: int = 0) -> str:
    """Render a Markdown numbered list.

    Args:
        items: List items.
        indent: Indentation in spaces.

    Returns:
        The Markdown list.
    """
    padding = " " * indent
    return "\n".join(f"{padding}{index}. {item}" for index, item in enumerate(items, start=1))


def column_dict(column: ColumnSpec) -> dict[str, Any]:
    """Flatten a :class:`ColumnSpec` into a plain dict usable in templates."""
    return column.model_dump(exclude_none=True)


def column_names(columns: list[ColumnSpec]) -> list[str]:
    """Return the names of a list of column specifications."""
    return [column.name for column in columns]


def names_with_role(columns: list[ColumnSpec], roles: list[str] | str) -> list[str]:
    """Return the names of the columns whose role is in ``roles``.

    Args:
        columns: Column specifications.
        roles: One role or a list of roles.

    Returns:
        The matching column names.
    """
    allowed = {roles} if isinstance(roles, str) else set(roles)
    return [column.name for column in columns if column.role in allowed]


def names_without_role(columns: list[ColumnSpec], roles: list[str] | str) -> list[str]:
    """Return the names of the columns whose role is *not* in ``roles``."""
    allowed = {roles} if isinstance(roles, str) else set(roles)
    return [column.name for column in columns if column.role not in allowed]


def columns_without_role(columns: list[ColumnSpec], roles: list[str] | str) -> list[ColumnSpec]:
    """Return the column specifications whose role is *not* in ``roles``."""
    allowed = {roles} if isinstance(roles, str) else set(roles)
    return [column for column in columns if column.role not in allowed]


def columns_with_role(columns: list[ColumnSpec], roles: list[str] | str) -> list[ColumnSpec]:
    """Return the column specifications whose role is in ``roles``."""
    allowed = {roles} if isinstance(roles, str) else set(roles)
    return [column for column in columns if column.role in allowed]


CHECK_LABELS: dict[str, str] = {
    "ge": ">=",
    "le": "<=",
    "gt": ">",
    "lt": "<",
    "eq": "==",
    "ne": "!=",
    "isin": "dans",
    "regex": "motif",
    "min_length": "longueur min",
    "max_length": "longueur max",
}


def format_checks(checks: dict[str, Any] | None) -> str:
    """Render Pandera checks as a short human readable string.

    Args:
        checks: Mapping of check name to value.

    Returns:
        A compact French description (``-`` when there is no check).
    """
    if not checks:
        return "-"
    parts: list[str] = []
    for key, value in checks.items():
        label = CHECK_LABELS.get(key, key)
        rendered = (
            ", ".join(str(item) for item in value)
            if isinstance(value, (list, tuple))
            else str(value)
        )
        parts.append(f"{label} {rendered}")
    return " · ".join(parts)


def build_context(spec: ProjectSpec, family: FamilySpec, stack: StackSpec) -> dict[str, Any]:
    """Assemble everything a template may need.

    Args:
        spec: Validated project manifest.
        family: Family definition of the manifest.
        stack: Stack definition of the manifest.

    Returns:
        The Jinja rendering context.
    """
    data = spec.data
    target_columns = [c for c in data.columns if c.role == "target"]
    target_column = target_columns[0] if target_columns else None

    dependencies = list(dict.fromkeys([*CORE_DEPENDENCIES, *stack.dependencies]))
    dev_dependencies = list(dict.fromkeys([*DEV_DEPENDENCIES, *stack.dependencies]))

    project = {
        "key": spec.key,
        "title": spec.title,
        "summary": spec.summary,
        "slug": spec.project_slug,
        "package": spec.package_name,
        "package_class": camel(spec.package_name),
        "relative_path": spec.relative_path,
        "domain": spec.domain,
        "problem": spec.problem,
        "stack_folder": f"with-{spec.stack_key}",
        "stack_key": spec.stack_key,
        "stack_name": stack.display_name,
        "family": spec.family,
        "modality": spec.modality,
        "experiment_name": f"{spec.domain}-{spec.problem}-{spec.stack_key}",
    }

    return {
        # --- manifest -------------------------------------------------------------------
        "spec": spec,
        "data": data,
        "business": spec.business,
        "model": spec.model,
        "metrics": spec.metrics,
        "train": spec.train,
        "preprocessing": spec.preprocessing,
        "hydra": spec.hydra,
        "extras": spec.extras,
        "learning_objectives": spec.learning_objectives,
        # --- registries -----------------------------------------------------------------
        "stack": stack,
        "family": family,
        "project": project,
        # --- dependencies ---------------------------------------------------------------
        "dependencies": dependencies,
        "dev_dependencies": dev_dependencies,
        "core_dependencies": CORE_DEPENDENCIES,
        # --- data conveniences ----------------------------------------------------------
        "columns": data.columns,
        "feature_columns": data.features,
        "numeric_columns": data.numeric_features,
        "categorical_columns": data.categorical_features,
        "feature_names": data.feature_names,
        "numeric_feature_names": data.numeric_feature_names,
        "categorical_feature_names": data.categorical_feature_names,
        "target_column": target_column,
        "dataset_name": data.dataset_name,
        "year": date.today().year,
        "today": date.today().isoformat(),
        # --- modèle -----------------------------------------------------------------------
        "model_class": stack.class_name or camel(stack.key) + "Model",
        # --- template helpers -----------------------------------------------------------
        "py": py,
        "yaml_dump": yaml_dump,
        "camel": camel,
        "const": const,
        "wrap": wrap,
        "py_string_tuple": py_string_tuple,
        "bullet_list": bullet_list,
        "numbered_list": numbered_list,
        "column_dict": column_dict,
        "pandera_annotation": pandera_annotation,
        "pandera_field": pandera_field,
        "pandera_columns_block": pandera_columns_block,
        "column_names": column_names,
        "names_with_role": names_with_role,
        "names_without_role": names_without_role,
        "columns_with_role": columns_with_role,
        "columns_without_role": columns_without_role,
        "format_checks": format_checks,
        "dtype_map_literal": dtype_map_literal,
    }
