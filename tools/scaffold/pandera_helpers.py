"""Helpers that turn manifest columns into Pandera field declarations."""

from __future__ import annotations

from typing import Any

from tools.scaffold.config import ColumnSpec

#: Mapping from the canonical dtype of a manifest onto the ``pandera.typing`` annotation.
PANDERA_ANNOTATIONS: dict[str, str] = {
    "int": "Series[int]",
    "float": "Series[float]",
    "str": "Series[str]",
    "category": "Series[str]",
    "bool": "Series[int]",
    "datetime": "Series[pa.DateTime]",
}

#: Mapping from a manifest check key onto the ``pa.Field`` keyword argument.
CHECK_TO_FIELD: dict[str, str] = {
    "ge": "ge",
    "le": "le",
    "gt": "gt",
    "lt": "lt",
    "eq": "eq",
    "ne": "ne",
    "isin": "isin",
    "regex": "str_matches",
    "min_length": "str_length",
    "max_length": "str_length",
    "nullable": "nullable",
    "unique": "unique",
}


def pandera_annotation(column: ColumnSpec) -> str:
    """Return the ``pandera.typing`` annotation of a column.

    Args:
        column: Column specification from the manifest.

    Returns:
        The annotation text, e.g. ``Series[float]``.
    """
    return PANDERA_ANNOTATIONS.get(column.dtype, "Series[Any]")


def _render_value(value: Any) -> str:
    """Render a Python literal for a Pandera keyword argument."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def pandera_field(
    column: ColumnSpec,
    *,
    inference: bool = False,
    drop_checks: tuple[str, ...] = ("unique",),
) -> str:
    """Render the ``pa.Field(...)`` declaration of a column.

    Args:
        column: Column specification.
        inference: When ``True``, relax the contract (columns optional, nulls allowed) because
            inference payloads are user-provided and incomplete by nature.
        drop_checks: Checks that must not be emitted (e.g. ``unique`` in an inference schema).

    Returns:
        The field declaration text.
    """
    kwargs: list[str] = []
    checks = dict(column.checks or {})

    nullable = bool(column.nullable) or (inference and column.dtype in {"int", "float", "str"})
    kwargs.append(f"nullable={_render_value(nullable)}")
    if inference:
        # Un payload d'inférence peut être partiel : la colonne n'est pas obligatoire.
        kwargs.append("required=False")
    if column.unique and "unique" not in drop_checks and not inference:
        kwargs.append("unique=True")

    str_length: dict[str, int] = {}
    for key, value in checks.items():
        if key in drop_checks:
            continue
        if key in {"min_length", "max_length"}:
            str_length["min_length" if key == "min_length" else "max_length"] = int(value)
            continue
        argument = CHECK_TO_FIELD.get(key, key)
        kwargs.append(f"{argument}={_render_value(value)}")
    if str_length:
        kwargs.append(f"str_length={_render_value(str_length)}")

    description = column.description.replace('"', "'")
    kwargs.append(f'description="{description}"')
    return "pa.Field(" + ", ".join(kwargs) + ")"


def pandera_columns_block(
    columns: list[ColumnSpec], *, inference: bool = False, indent: int = 4
) -> str:
    """Render a whole block of annotated columns for a ``DataFrameModel``.

    Args:
        columns: Columns to declare.
        inference: Relax the contract for inference payloads.
        indent: Indentation in spaces.

    Returns:
        The Python source block.
    """
    padding = " " * indent
    lines: list[str] = []
    for column in columns:
        annotation = pandera_annotation(column)
        field = pandera_field(column, inference=inference)
        lines.append(f"{padding}{column.name}: {annotation} = {field}")
    return "\n".join(lines)


def dtype_map_literal(columns: list[ColumnSpec]) -> str:
    """Render a ``{column: dtype}`` documentation mapping."""
    items = ", ".join(f'"{column.name}": "{column.dtype}"' for column in columns)
    return "{" + items + "}"
