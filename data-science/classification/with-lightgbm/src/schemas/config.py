"""Typed configuration of the project.

Hydra/OmegaConf gives us composition and overrides, but a ``DictConfig`` is untyped: a typo
in a key silently becomes ``None`` and explodes far from its cause. This module declares the
contract with Pydantic models so that:

* every key of ``conf/`` is validated at start-up (``validate_config``),
* the IDE offers autocompletion on ``config.train.split.test_size``,
* an invalid value (negative epoch, unknown scaler) fails fast with an explicit message,
* family/stack specific options remain possible thanks to ``extra="allow"``.

Usage:
    >>> from omegaconf import OmegaConf
    >>> from src.schemas.config import validate_config
    >>> cfg = validate_config(OmegaConf.create({"mode": "train", "seed": 42}))
    >>> cfg.mode
    'train'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Learning tasks handled by this project template.
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

#: Execution modes of ``src/main.py``.
Mode = Literal["generate-data", "train", "evaluate", "predict", "all"]

_FLEXIBLE = ConfigDict(extra="allow", populate_by_name=True, validate_assignment=False)


class PathsConfig(BaseModel):
    """Filesystem layout. ``None`` values are resolved from ``src/utils/paths.py``."""

    model_config = _FLEXIBLE

    root: Path | None = None
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    outputs_dir: Path | None = None


class ProjectConfig(BaseModel):
    """Identity of the project (used in logs, artefacts and reports)."""

    model_config = _FLEXIBLE

    key: str = "ds-classification-lightgbm"
    name: str = "telecom-churn-lightgbm"
    domain: str = "data-science"
    problem: str = "classification"
    stack: str = "lightgbm"
    family: str = "binary_classification"
    modality: str = "tabular"
    title: str = "Prédiction d'attrition client (churn télécom) avec LightGBM"
    version: str = "1.0.0"


class ValidationConfig(BaseModel):
    """Which Pandera contracts are enforced, and how."""

    model_config = _FLEXIBLE

    raw: bool = True
    processed: bool = True
    inference: bool = True
    strict: bool = True
    lazy: bool = False


class SamplingConfig(BaseModel):
    """Row subsetting applied after loading (useful for quick iterations)."""

    model_config = _FLEXIBLE

    limit: int | None = Field(default=None, ge=1)


class DataConfig(BaseModel):
    """Dataset description: what to load, which column is the target, what to validate."""

    model_config = _FLEXIBLE

    dataset_name: str = "telecom_churn"
    n_samples: int = Field(default=4000, ge=50)
    seed: int = 42
    formats: list[str] = Field(default_factory=lambda: ["parquet", "csv"])
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")
    target: str | None = "churned"
    id_column: str | None = "customer_id"
    time_column: str | None = None
    group_column: str | None = None
    positive_rate: float | None = Field(default=0.26, ge=0.0, le=1.0)
    drop_columns: list[str] = Field(default_factory=list)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, value: list[str]) -> list[str]:
        """Reject unsupported storage formats."""
        allowed = {"parquet", "csv"}
        unknown = [fmt for fmt in value if fmt not in allowed]
        if unknown:
            msg = f"Unsupported data formats {unknown}. Allowed: {sorted(allowed)}"
            raise ValueError(msg)
        return value


class NumericPreprocessingConfig(BaseModel):
    """Strategy applied to numeric columns."""

    model_config = _FLEXIBLE

    imputer: Literal["mean", "median", "constant", "none"] = "median"
    imputer_fill_value: float = 0.0
    clip_outliers: bool = True
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"
    log1p_columns: list[str] = Field(default_factory=list)

    @field_validator("clip_quantiles")
    @classmethod
    def _valid_quantiles(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Ensure the quantile window is ordered and inside ``[0, 1]``."""
        low, high = float(value[0]), float(value[1])
        if not 0.0 <= low < high <= 1.0:
            msg = f"clip_quantiles must satisfy 0 <= low < high <= 1, got ({low}, {high})"
            raise ValueError(msg)
        return (low, high)


class CategoricalPreprocessingConfig(BaseModel):
    """Strategy applied to categorical / textual columns."""

    model_config = _FLEXIBLE

    imputer: Literal["most_frequent", "constant", "none"] = "most_frequent"
    imputer_fill_value: str = "unknown"
    encoder: Literal["onehot", "ordinal", "target", "none"] = "onehot"
    handle_unknown: Literal["ignore", "error", "infrequent_if_exist"] = "ignore"
    min_frequency: float = Field(default=0.01, ge=0.0, le=1.0)
    max_categories: int | None = Field(default=25, ge=2)


class PreprocessingConfig(BaseModel):
    """Cleaning, encoding, scaling and derived features."""

    model_config = _FLEXIBLE

    numeric: NumericPreprocessingConfig = Field(default_factory=NumericPreprocessingConfig)
    categorical: CategoricalPreprocessingConfig = Field(
        default_factory=CategoricalPreprocessingConfig
    )
    features: list[dict[str, Any]] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    keep_columns: list[str] | None = None
    #: Renommé `validation` (et non `validate`) pour ne pas masquer `BaseModel.validate`.
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class SplitConfig(BaseModel):
    """Train / validation / test split policy."""

    model_config = _FLEXIBLE

    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    val_size: float = Field(default=0.15, ge=0.0, lt=0.5)
    stratify: bool = True
    shuffle: bool = True
    time_based: bool = False

    @field_validator("val_size")
    @classmethod
    def _compatible_sizes(cls, value: float, info: Any) -> float:
        """Guarantee that the training split remains non-empty."""
        test_size = float(info.data.get("test_size", 0.2))
        if test_size + value >= 0.9:
            msg = f"test_size + val_size must stay below 0.9, got {test_size + value:.2f}"
            raise ValueError(msg)
        return value


class EarlyStoppingConfig(BaseModel):
    """Early stopping rule (advisory: honoured by the training loop)."""

    model_config = _FLEXIBLE

    enabled: bool = True
    monitor: str = "val_loss"
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    mode: Literal["min", "max"] = "min"


class CallbacksConfig(BaseModel):
    """Callbacks enabled during training."""

    model_config = _FLEXIBLE

    logging_every: int = Field(default=1, ge=1)
    progress_bar: bool = True
    metric_history: bool = True


class ArtifactsConfig(BaseModel):
    """File names of every artefact written by the training pipeline."""

    model_config = _FLEXIBLE

    model_file: str = "model.joblib"
    pipeline_file: str = "preprocessing.joblib"
    feature_builder_file: str = "feature_builder.joblib"
    metrics_file: str = "training_metrics.json"
    model_card_file: str = "model_card.json"
    report_file: str = "evaluation_report.md"
    predictions_file: str = "predictions.csv"


class CrossValidationConfig(BaseModel):
    """Cross-validation used for model selection."""

    model_config = _FLEXIBLE

    enabled: bool = True
    folds: int = Field(default=5, ge=2, le=20)
    strategy: Literal["kfold", "stratified", "timeseries", "group"] = "kfold"
    n_jobs: int = -1


class TrainConfig(BaseModel):
    """Everything the trainer needs: data split, optimisation, callbacks, artefacts."""

    model_config = _FLEXIBLE

    split: SplitConfig = Field(default_factory=SplitConfig)
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    n_jobs: int = -1


class ModelConfig(BaseModel):
    """Model selection and hyper-parameters."""

    model_config = _FLEXIBLE

    name: str = "Gradient boosting LightGBM"
    algorithm: str = "lightgbm"
    model_class: str = Field(default="LightGBMModel", alias="class")
    task: Task = "binary"
    supports_proba: bool = True
    epochs_based: bool = False
    random_state: int = 42
    params: dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """Metrics computed by the evaluator and asserted by the smoke tests."""

    model_config = _FLEXIBLE

    task: Task = "binary"
    primary: str = "roc_auc"
    secondary: list[str] = Field(
        default_factory=lambda: [
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "log_loss",
        ]
    )
    baseline: str = "dummy"
    direction: Literal["maximize", "minimize"] = "maximize"
    min_primary: float | None = 0.7

    @property
    def all_metrics(self) -> list[str]:
        """Primary metric first, then the secondary ones (deduplicated)."""
        return list(dict.fromkeys([self.primary, *self.secondary]))


class PredictConfig(BaseModel):
    """Options of the inference mode."""

    model_config = _FLEXIBLE

    input: Path | None = None
    n_samples: int = Field(default=5, ge=1)
    output: Path = Path("artifacts/reports/predictions.csv")


class AppConfig(BaseModel):
    """Root configuration object, mirroring ``conf/config.yaml``."""

    model_config = _FLEXIBLE

    mode: Mode = "train"
    seed: int = 42
    log_level: str = "INFO"
    log_file: Path | None = None
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        """Normalise and validate the log level."""
        level = str(value).upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            msg = f"Unknown log level '{value}'. Allowed: {sorted(allowed)}"
            raise ValueError(msg)
        return level

    def project_paths(self) -> Any:
        """Build the :class:`src.utils.paths.ProjectPaths` layout from this config.

        Returns:
            The resolved project paths.
        """
        from src.utils.paths import ProjectPaths

        return ProjectPaths.from_config(self.model_dump())


def validate_config(config: DictConfig | dict[str, Any] | AppConfig) -> AppConfig:
    """Validate a raw Hydra configuration into a typed :class:`AppConfig`.

    Args:
        config: OmegaConf ``DictConfig``, plain mapping or already validated config.

    Returns:
        The validated configuration object.

    Raises:
        pydantic.ValidationError: When the configuration does not respect the contract.
    """
    if isinstance(config, AppConfig):
        return config
    container = (
        OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    )
    if not isinstance(container, dict):
        msg = f"Expected a mapping-like configuration, got {type(container).__name__}"
        raise TypeError(msg)
    return AppConfig.model_validate(container)
