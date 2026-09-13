"""Typed models describing a project manifest.

The scaffold tool is fully declarative: every project of the repository is described by a
YAML manifest (``tools/scaffold/manifests/*.yaml``) which is validated with Pydantic before
being rendered through Jinja2 template layers.

The layering is the following (each layer may override files produced by the previous one):

1. ``templates/base``                -> files shared by *every* project (Makefile, conf/, utils)
2. ``templates/modality/<modality>`` -> preprocessing / features / loaders of a data modality
3. ``templates/family/<family>``     -> schemas, synthetic data generators, evaluator, predictor
4. ``templates/stack/<stack>``       -> framework specific model, trainer and dependencies

Attributes:
    ProjectSpec: Root object of a manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Modality = Literal["tabular", "text", "image", "platform"]
Task = Literal[
    "binary",
    "multiclass",
    "regression",
    "clustering",
    "forecasting",
    "ranking",
    "anomaly",
    "retrieval",
    "generation",
    "pipeline",
]

DTYPE_ALIASES: dict[str, str] = {
    "int": "int",
    "int64": "int",
    "integer": "int",
    "float": "float",
    "float64": "float",
    "double": "float",
    "str": "str",
    "string": "str",
    "category": "category",
    "bool": "bool",
    "boolean": "bool",
    "datetime": "datetime",
    "date": "datetime",
}


class ColumnSpec(BaseModel):
    """A single column of the synthetic dataset shipped with a project."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Column name as it appears in the raw dataset.")
    dtype: str = Field(description="Logical dtype: int, float, str, category, bool, datetime.")
    role: Literal["identifier", "feature", "target", "timestamp", "metadata", "group"]
    description: str = Field(description="Business meaning (French, used in READMEs).")
    unit: str | None = Field(default=None, description="Physical/business unit when relevant.")
    nullable: bool = False
    unique: bool = False
    checks: dict[str, Any] = Field(
        default_factory=dict,
        description="Pandera checks: ge, le, gt, lt, isin, regex, str_length, element_wise...",
    )
    distribution: str | None = Field(
        default=None, description="Expected distribution, documented in data/README.md."
    )
    generator: dict[str, Any] = Field(
        default_factory=dict,
        description="Recipe consumed by the family-specific synthetic data generator.",
    )

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalise_dtype(cls, value: str) -> str:
        """Map the many ways of spelling a dtype onto the canonical set."""
        key = str(value).strip().lower()
        if key not in DTYPE_ALIASES:
            msg = f"Unknown dtype '{value}'. Allowed: {sorted(set(DTYPE_ALIASES.values()))}"
            raise ValueError(msg)
        return DTYPE_ALIASES[key]


class DataSpec(BaseModel):
    """Everything needed to generate, document and validate the example dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(description="Snake_case dataset name, e.g. ``telecom_churn``.")
    title: str
    description: str
    n_samples: int = Field(default=4000, ge=50)
    seed: int = 42
    formats: list[str] = Field(default_factory=lambda: ["parquet", "csv"])
    columns: list[ColumnSpec] = Field(min_length=2)
    target: str | None = None
    id_column: str | None = None
    time_column: str | None = None
    group_column: str | None = None
    positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def by_role(self, role: str) -> list[ColumnSpec]:
        """Return the columns having the given role."""
        return [column for column in self.columns if column.role == role]

    @property
    def features(self) -> list[ColumnSpec]:
        """Model input columns."""
        return self.by_role("feature")

    @property
    def numeric_features(self) -> list[ColumnSpec]:
        """Numeric model input columns."""
        return [c for c in self.features if c.dtype in {"int", "float"}]

    @property
    def categorical_features(self) -> list[ColumnSpec]:
        """Categorical / textual model input columns."""
        return [c for c in self.features if c.dtype in {"str", "category", "bool"}]

    @property
    def feature_names(self) -> list[str]:
        """Names of the model input columns."""
        return [c.name for c in self.features]

    @property
    def numeric_feature_names(self) -> list[str]:
        """Names of the numeric input columns."""
        return [c.name for c in self.numeric_features]

    @property
    def categorical_feature_names(self) -> list[str]:
        """Names of the categorical input columns."""
        return [c.name for c in self.categorical_features]

    @property
    def all_names(self) -> list[str]:
        """Names of every column, in declaration order."""
        return [c.name for c in self.columns]


class BusinessSpec(BaseModel):
    """The concrete business use case demonstrated by the project."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(description="Who owns / consumes the model (Qui ?).")
    context: str = Field(description="Business context (Pourquoi ?).")
    problem: str = Field(description="Problem statement (Quel problème ?).")
    inputs: str
    outputs: str
    value: str = Field(description="Business value / expected impact.")
    success_criteria: list[str] = Field(min_length=1)
    cadence: str = Field(default="Batch quotidien", description="Serving / refresh cadence.")
    constraints: list[str] = Field(default_factory=list)


class ModelSpec(BaseModel):
    """Model choice documentation + hyper-parameters injected in ``conf/model/default.yaml``."""

    model_config = ConfigDict(extra="forbid")

    display_name: str
    algorithm: str
    rationale: str
    params: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[str] = Field(default_factory=list)


class MetricsSpec(BaseModel):
    """Metrics computed by the evaluator and asserted by the tests."""

    model_config = ConfigDict(extra="forbid")

    task: Task
    primary: str
    secondary: list[str] = Field(default_factory=list)
    baseline: str = Field(default="dummy", description="Baseline model name used for comparison.")
    min_primary: float | None = Field(
        default=None, description="Smoke-test threshold on the primary metric."
    )
    direction: Literal["maximize", "minimize"] = "maximize"


class ProjectSpec(BaseModel):
    """Root manifest object: one instance == one generated project directory."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Unique project key, e.g. ``ds-classification-sklearn``.")
    title: str
    summary: str
    domain: str = Field(description="Root domain folder: data-science, nlp, ai-eng, ...")
    problem: str = Field(description="Problem folder inside the domain: classification, rag, ...")
    stack: str = Field(description="Stack key of the registry: sklearn, pytorch, transformers, ...")
    project_slug: str = Field(description="Human readable project name used in titles and paths.")
    package_name: str = Field(description="Valid Python identifier used for imports/artifacts.")
    modality: Modality = "tabular"
    family: str = Field(description="Template family key, e.g. ``binary_classification``.")
    business: BusinessSpec
    data: DataSpec
    model: ModelSpec
    metrics: MetricsSpec
    train: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    hydra: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)
    learning_objectives: list[str] = Field(default_factory=list)

    @field_validator("package_name")
    @classmethod
    def _valid_identifier(cls, value: str) -> str:
        """Guarantee that the package name can be imported."""
        if not value.isidentifier():
            msg = f"package_name '{value}' is not a valid Python identifier"
            raise ValueError(msg)
        return value

    @property
    def relative_path(self) -> str:
        """Where the project is generated, relative to the repository root."""
        stack_folder = self.stack if self.stack.startswith("with-") else f"with-{self.stack}"
        return f"{self.domain}/{self.problem}/{stack_folder}"

    @property
    def stack_key(self) -> str:
        """Registry key of the stack (without the ``with-`` prefix)."""
        return self.stack.removeprefix("with-")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (returns a new dict).

    Args:
        base: Mapping providing the default values.
        override: Mapping whose values win over ``base``.

    Returns:
        The merged mapping.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_manifest(path: Path, defaults_dir: Path | None = None) -> ProjectSpec:
    """Load a manifest, apply the default layers and validate it.

    Args:
        path: Path to the ``*.yaml`` manifest.
        defaults_dir: Directory containing ``global.yaml`` and ``family/<family>.yaml`` defaults.

    Returns:
        A fully validated :class:`ProjectSpec`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Manifest {path} must contain a YAML mapping at the top level"
        raise TypeError(msg)

    payload: dict[str, Any] = {}
    if defaults_dir is not None:
        # Couche 1 : valeurs par défaut globales (tous les projets).
        global_defaults_path = Path(defaults_dir) / "global.yaml"
        if global_defaults_path.exists():
            payload = _deep_merge(
                payload, yaml.safe_load(global_defaults_path.read_text(encoding="utf-8")) or {}
            )
        # Couche 2 : valeurs par défaut de la famille (cas d'usage, données, métriques).
        # La famille doit être déclarée dans le manifeste lui-même : c'est elle qui sélectionne
        # cette couche.
        family = raw.get("family")
        if not family:
            msg = (
                f"Manifest {path} must declare 'family' (see tools/scaffold/registry/families.yaml)"
            )
            raise KeyError(msg)
        family_defaults_path = Path(defaults_dir) / "family" / f"{family}.yaml"
        if family_defaults_path.exists():
            payload = _deep_merge(
                payload, yaml.safe_load(family_defaults_path.read_text(encoding="utf-8")) or {}
            )
    # Couche 3 : le manifeste gagne toujours.
    payload = _deep_merge(payload, raw)
    return ProjectSpec.model_validate(payload)
