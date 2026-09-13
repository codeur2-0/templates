"""Shared helpers for the notebook builders (cell factories + context object)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nbformat
from nbformat.notebooknode import NotebookNode

from tools.scaffold.config import ProjectSpec
from tools.scaffold.registry import FamilySpec, StackSpec

NBFORMAT_VERSION = 4


def markdown_cell(source: str) -> NotebookNode:
    """Create a Markdown cell.

    Args:
        source: Markdown content.

    Returns:
        The notebook node.
    """
    return nbformat.v4.new_markdown_cell(source=_clean(source))


def code_cell(source: str) -> NotebookNode:
    """Create a code cell.

    Args:
        source: Python source.

    Returns:
        The notebook node.
    """
    return nbformat.v4.new_code_cell(source=_clean(source))


def _clean(source: str) -> str:
    """Normalise a cell source: strip the common indentation, keep the trailing newline off."""
    lines = source.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    dedented = "\n".join(line[indent:] if line.strip() else "" for line in lines)
    return dedented.rstrip()


def write_notebook(path: Path, cells: Sequence[NotebookNode], *, kernel: str = "python3") -> Path:
    """Write a notebook file.

    Args:
        path: Destination ``.ipynb``.
        cells: Ordered cells.
        kernel: Kernel name declared in the metadata.

    Returns:
        The written path.
    """
    notebook = nbformat.v4.new_notebook()
    notebook["cells"] = list(cells)
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


@dataclass(slots=True)
class NotebookContext:
    """Everything a notebook builder needs, pre-resolved once.

    Attributes:
        spec: Validated manifest.
        family: Family definition.
        stack: Stack definition.
        model_class: Concrete model class name of the stack.
        metric_labels: Human readable metric names (French) for Markdown text.
    """

    spec: ProjectSpec
    family: FamilySpec
    stack: StackSpec
    model_class: str = "SklearnModel"
    metric_labels: dict[str, str] = field(default_factory=dict)

    # -- paths & names ---------------------------------------------------------------
    @property
    def dataset_name(self) -> str:
        """Base name of the dataset."""
        return self.spec.data.dataset_name

    @property
    def raw_parquet(self) -> str:
        """Relative path of the raw Parquet file."""
        return f"../data/raw/{self.dataset_name}.parquet"

    @property
    def raw_csv(self) -> str:
        """Relative path of the raw CSV file."""
        return f"../data/raw/{self.dataset_name}.csv"

    @property
    def target(self) -> str | None:
        """Target column name."""
        return self.spec.data.target

    @property
    def id_column(self) -> str | None:
        """Identifier column name."""
        return self.spec.data.id_column

    @property
    def feature_names(self) -> list[str]:
        """Feature column names."""
        return self.spec.data.feature_names

    @property
    def numeric_features(self) -> list[str]:
        """Numeric feature names."""
        return self.spec.data.numeric_feature_names

    @property
    def categorical_features(self) -> list[str]:
        """Categorical feature names."""
        return self.spec.data.categorical_feature_names

    @property
    def columns(self) -> list[str]:
        """Every column name."""
        return self.spec.data.all_names

    @property
    def task(self) -> str:
        """Learning task."""
        return self.spec.metrics.task

    @property
    def primary_metric(self) -> str:
        """Primary metric name."""
        return self.spec.metrics.primary

    @property
    def title(self) -> str:
        """Project title."""
        return self.spec.title

    def py_list(self, values: Sequence[Any]) -> str:
        """Render a Python list literal for notebook code."""
        return "[" + ", ".join(repr(str(value)) for value in values) + "]"


METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC (average precision)",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Précision (classe positive)",
    "recall": "Rappel (classe positive)",
    "f1": "F1-score (classe positive)",
    "f1_macro": "F1 macro-moyenné",
    "f1_weighted": "F1 pondéré",
    "log_loss": "Log-loss (entropie croisée)",
    "mcc": "Corrélation de Matthews",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE (%)",
    "smape": "sMAPE (%)",
    "r2": "R²",
    "max_error": "Erreur maximale",
    "mase": "MASE",
    "silhouette": "Silhouette",
    "calinski_harabasz": "Calinski-Harabasz",
    "davies_bouldin": "Davies-Bouldin",
    "n_clusters": "Nombre de clusters",
    "ndcg_at_k": "nDCG@K",
    "precision_at_k": "Precision@K",
    "recall_at_k": "Recall@K",
    "map_at_k": "MAP@K",
    "hit_rate_at_k": "Hit rate@K",
    "recall_at_budget": "Rappel au budget d'alerte",
    "precision_at_budget": "Précision au budget d'alerte",
}


def build_context(spec: ProjectSpec, family: FamilySpec, stack: StackSpec) -> NotebookContext:
    """Build the notebook context from the manifest and the registries.

    Args:
        spec: Validated manifest.
        family: Family definition.
        stack: Stack definition.

    Returns:
        The :class:`NotebookContext`.
    """
    model_class = stack.class_name or "".join(
        part.capitalize() for part in stack.key.replace("-", "_").split("_")
    ) + "Model"
    return NotebookContext(
        spec=spec,
        family=family,
        stack=stack,
        model_class=model_class,
        metric_labels=dict(METRIC_LABELS),
    )
