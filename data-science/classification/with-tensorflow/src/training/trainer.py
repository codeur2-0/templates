"""Training orchestration.

The :class:`Trainer` owns *the process* of training, not the model:

* it builds the callbacks declared in the configuration,
* it calls ``model.fit`` with the training and validation matrices,
* it computes the validation metrics with the shared metric registry,
* it persists the artefacts (model, model card, metrics JSON),
* it returns a structured :class:`TrainingOutcome` instead of printing things.

Because it only depends on :class:`src.models.base.BaseModel`, the exact same trainer works
for scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow and Keras implementations.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel, FitResult
from src.training.callbacks import (
    BaseCallback,
    CallbackContext,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    MetricThresholdCallback,
    ProgressBarCallback,
)
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.utils.utils import timer

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingData:
    """Matrices handed to the trainer (already preprocessed).

    Attributes:
        X_train: Training features.
        y_train: Training target (``None`` for unsupervised tasks).
        X_val: Validation features.
        y_val: Validation target.
        feature_names: Ordered feature names.
        task: Learning task identifier.
    """

    X_train: pd.DataFrame
    y_train: pd.Series | None
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    feature_names: list[str] = field(default_factory=list)
    task: str = "binary"

    @classmethod
    def from_bundle(cls, bundle: Any) -> TrainingData:
        """Build the training data from a :class:`src.data.loaders.DatasetBundle`.

        Args:
            bundle: Dataset bundle produced by the data layer.

        Returns:
            The training matrices (the test split is *not* given to the trainer).
        """
        return cls(
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            X_val=bundle.X_val,
            y_val=bundle.y_val,
            feature_names=list(bundle.feature_names),
            task=str(bundle.task),
        )

    def describe(self) -> dict[str, Any]:
        """Return a compact summary for logs and artefacts."""
        return {
            "n_train": len(self.X_train),
            "n_val": 0 if self.X_val is None else len(self.X_val),
            "n_features": int(self.X_train.shape[1]),
            "task": self.task,
        }


@dataclass(slots=True)
class TrainingOutcome:
    """Everything produced by a training run.

    Attributes:
        model: The fitted model instance.
        fit_result: Metrics and history returned by ``model.fit``.
        validation_metrics: Metrics computed by the trainer on the validation split.
        history: Per-epoch metric history (deep learning / boosting).
        artifacts: Artefact name -> path.
        duration_seconds: Total training duration.
    """

    model: BaseModel
    fit_result: FitResult
    validation_metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def metrics(self) -> dict[str, float]:
        """All known metrics, trainer-computed values winning over model-reported ones."""
        return {**self.fit_result.metrics, **self.validation_metrics}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the outcome (JSON friendly)."""
        return {
            "model": self.model.summary(),
            "framework": self.model.framework,
            "task": self.model.task,
            "metrics": self.metrics,
            "history": self.history,
            "artifacts": self.artifacts,
            "duration_seconds": round(self.duration_seconds, 4),
            "fit_result": self.fit_result.to_dict(),
        }


class Trainer:
    """Framework-agnostic training orchestrator."""

    def __init__(
        self,
        model: BaseModel,
        *,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        metric_names: Sequence[str] | None = None,
        task: str | None = None,
        callbacks: Sequence[BaseCallback] | None = None,
        min_primary_metric: float | None = None,
        primary_metric: str | None = None,
        primary_direction: str = "maximize",
    ) -> None:
        """Inject the model and the training policy.

        Args:
            model: Model implementing :class:`BaseModel`.
            config: Root (or ``train``) configuration mapping.
            paths: Filesystem layout used to persist artefacts.
            metric_names: Metrics to compute on the validation split.
            task: Learning task (defaults to the model's ``task`` class attribute).
            callbacks: Explicit callbacks; when ``None`` they are built from the config.
            min_primary_metric: Optional quality gate enforced during training.
            primary_metric: Metric watched by the quality gate.
            primary_direction: ``maximize`` (AUC, R2) or ``minimize`` (RMSE, MAE) — fixe le sens
                dans lequel le seuil de qualité est évalué.
        """
        self.model = model
        self.config: dict[str, Any] = _normalise_train_config(config)
        self.paths = paths or ProjectPaths.from_root()
        self.task = task or model.task
        self.metric_names = list(metric_names or [])
        self.min_primary_metric = min_primary_metric
        self.primary_metric = primary_metric
        self.primary_direction = str(primary_direction)
        self.callbacks: list[BaseCallback] = (
            list(callbacks) if callbacks is not None else self.build_callbacks()
        )

    # ------------------------------------------------------------------ callbacks -------
    def build_callbacks(self) -> list[BaseCallback]:
        """Build the callbacks declared in ``conf/train/default.yaml``.

        Returns:
            The ordered callback list.
        """
        callback_config = dict(self.config.get("callbacks", {}) or {})
        early_stopping_config = dict(self.config.get("early_stopping", {}) or {})
        callbacks: list[BaseCallback] = [
            LoggingCallback(every=int(callback_config.get("logging_every", 1))),
        ]
        if bool(callback_config.get("metric_history", True)):
            callbacks.append(MetricHistoryCallback())
        if bool(callback_config.get("progress_bar", True)):
            callbacks.append(ProgressBarCallback(enabled=_interactive_progress_bar()))
        if bool(early_stopping_config.get("enabled", False)):
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=str(early_stopping_config.get("monitor", "val_loss")),
                    patience=int(early_stopping_config.get("patience", 5)),
                    min_delta=float(early_stopping_config.get("min_delta", 0.0)),
                    mode=str(early_stopping_config.get("mode", "min")),
                )
            )
        if self.min_primary_metric is not None and self.primary_metric:
            # `mode` dépend du **sens** de la métrique : un RMSE doit rester sous le seuil
            # (mode="min"), un ROC AUC doit le dépasser (mode="max").
            callbacks.append(
                MetricThresholdCallback(
                    monitor=self.primary_metric,
                    threshold=float(self.min_primary_metric),
                    mode="max" if self.primary_direction == "maximize" else "min",
                )
            )
        logger.debug("Callbacks enabled: {}", [callback.name for callback in callbacks])
        return callbacks

    # ------------------------------------------------------------------ training --------
    def train(self, data: TrainingData) -> TrainingOutcome:
        """Train the model and persist the artefacts.

        Args:
            data: Preprocessed training and validation matrices.

        Returns:
            The :class:`TrainingOutcome`.

        Raises:
            ValueError: When the data does not match the model contract.
        """
        logger.info("Trainer started | model={} | {}", self.model.summary(), data.describe())
        self._check_contract(data)

        context = CallbackContext(
            model_name=type(self.model).__name__,
            params=dict(self.model.params),
            epochs=int(self.config.get("epochs", 1)),
        )
        with timer("training") as elapsed:
            fit_result = self.model.fit(
                data.X_train,
                data.y_train,
                X_val=data.X_val,
                y_val=data.y_val,
                callbacks=self.callbacks,
            )
        context.history = dict(fit_result.history)
        for callback in self.callbacks:
            callback.on_train_end(context)

        validation_metrics = self.compute_validation_metrics(data)
        merged = {**fit_result.metrics, **validation_metrics}
        fit_result.metrics = merged
        fit_result.duration_seconds = elapsed["seconds"]
        fit_result.n_samples = len(data.X_train)
        fit_result.n_features = int(data.X_train.shape[1])

        outcome = TrainingOutcome(
            model=self.model,
            fit_result=fit_result,
            validation_metrics=validation_metrics,
            history=fit_result.history,
            duration_seconds=elapsed["seconds"],
        )
        outcome.artifacts = {
            str(name): str(path) for name, path in self.save_artifacts(outcome).items()
        }
        logger.info(
            "Trainer finished in {:.2f}s | {}",
            outcome.duration_seconds,
            ", ".join(f"{key}={value:.5f}" for key, value in list(merged.items())[:6])
            or "no metric",
        )
        return outcome

    def _check_contract(self, data: TrainingData) -> None:
        """Assert that the data respects the model contract."""
        if data.X_train.empty:
            msg = "Training data is empty"
            raise ValueError(msg)
        missing = [name for name in self.model.feature_names if name not in data.X_train.columns]
        if missing:
            msg = f"Training data misses model features {missing[:10]} (total {len(missing)})"
            raise ValueError(msg)
        if data.X_train.isna().any().any():
            missing_cells = int(data.X_train.isna().to_numpy().sum())
            msg = f"Training features contain {missing_cells} missing value(s): preprocess first"
            raise ValueError(msg)
        if (
            self.model.task in {"binary", "multiclass", "regression", "forecasting"}
            and data.y_train is None
        ):
            msg = f"Task '{self.model.task}' is supervised but no target was provided"
            raise ValueError(msg)

    def compute_validation_metrics(self, data: TrainingData) -> dict[str, float]:
        """Score the model on the validation split using the shared metric registry.

        Args:
            data: Training data (the validation split is used when present).

        Returns:
            Mapping of metric name to value; empty when no validation split or no metric.
        """
        if data.X_val is None or not self.metric_names:
            return {}
        # Une tâche supervisée sans cible de validation ne peut pas être scorée ; en non
        # supervisé (clustering), les métriques internes se calculent sur X et les affectations.
        if data.y_val is None and self.task != "clustering":
            return {}
        predictions = self.model.predict(data.X_val)
        probabilities = None
        if getattr(self.model, "supports_proba", False):
            try:
                probabilities = self.model.predict_proba(data.X_val)
            except NotImplementedError:
                probabilities = None
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(
                y_true=data.y_val,
                y_pred=predictions,
                y_proba=probabilities,
                X=data.X_val,
                extra={"groups": data.X_val.get("group") if hasattr(data.X_val, "get") else None},
            )
        )
        # Une métrique non mesurable (tâche non supervisée sans cible, split dégénéré) est omise
        # plutôt que publiée en NaN : les callbacks, les courbes et l'artefact JSON travaillent
        # tous sur des nombres finis, et une clé absente dit exactement ce qui a été mesuré.
        prefixed = {
            f"val_{name}": float(value)
            for name, value in values.items()
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))
        }
        logger.info("Validation metrics: {}", {k: round(v, 5) for k, v in prefixed.items()})
        return prefixed

    # ------------------------------------------------------------------ artefacts -------
    def save_artifacts(self, outcome: TrainingOutcome) -> dict[str, Path]:
        """Persist the model, its card and the metrics.

        Args:
            outcome: Training outcome to persist.

        Returns:
            Mapping of artefact kind to written path.
        """
        artifact_config = dict(self.config.get("artifacts", {}) or {})
        written: dict[str, Path] = {}

        model_path = self.paths.models_dir / str(artifact_config.get("model_file", "model.joblib"))
        written["model"] = outcome.model.save(model_path)

        card = outcome.model.model_card(metrics=outcome.metrics)
        card.library_versions.update(_library_versions())
        card_path = self.paths.models_dir / str(
            artifact_config.get("model_card_file", "model_card.json")
        )
        written["model_card"] = card.to_json(card_path)

        metrics_payload = {
            "task": self.task,
            "framework": outcome.model.framework,
            "model": type(outcome.model).__name__,
            "metrics": _sanitise(outcome.metrics),
            "history": _sanitise_history(outcome.history),
            "duration_seconds": round(outcome.duration_seconds, 4),
            "params": outcome.model.params,
            "artifacts": {key: str(path) for key, path in written.items()},
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        metrics_path = self.paths.metrics_dir / str(
            artifact_config.get("metrics_file", "training_metrics.json")
        )
        written["metrics"] = write_json(metrics_path, metrics_payload)
        logger.info("Artefacts written: {}", {key: str(path) for key, path in written.items()})
        return written


def _normalise_train_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract the ``train`` node from a root configuration (or accept it directly).

    Args:
        config: Root configuration or ``train`` node.

    Returns:
        The training configuration mapping.
    """
    raw = dict(config or {})
    if "train" in raw and isinstance(raw["train"], Mapping):
        raw = dict(raw["train"])
    raw.setdefault("epochs", 1)
    raw.setdefault("callbacks", {})
    raw.setdefault("early_stopping", {})
    raw.setdefault("artifacts", {})
    return raw


def _interactive_progress_bar() -> bool:
    """Disable progress bars in non interactive environments (CI, pytest)."""
    if not sys.stdout.isatty():
        return False
    return "PYTEST_CURRENT_TEST" not in os.environ


#: Librairies dont la version est consignée pour la traçabilité du run.
_TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "scipy",
    "torch",
    "tensorflow",
    "xgboost",
    "lightgbm",
    "catboost",
    "pandera",
    "pydantic",
    "hydra",
    "omegaconf",
    "loguru",
    "matplotlib",
    "seaborn",
    "plotly",
    "joblib",
)


def _library_versions() -> dict[str, str]:
    """Collect the versions of the libraries **already loaded** in this process.

    Importer une librairie uniquement pour lire sa version est coûteux — et dangereux : faire
    cohabiter PyTorch et TensorFlow dans un même processus peut déclencher un conflit de runtime
    OpenMP (segfault). On se limite donc à ``sys.modules``, qui reflète fidèlement l'environnement
    réellement utilisé par le projet.

    Returns:
        Library name -> version mapping (best effort).
    """
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in _TRACKED_LIBRARIES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[module_name] = str(version)
    return versions


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one.

    Args:
        value: Raw metric value (NumPy scalar, ``None``, NaN, text, ...).

    Returns:
        The finite float, or ``None``.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sanitise(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert metrics to JSON-serialisable **finite** floats.

    A metric that could not be computed (tâche non supervisée entraînée sans cible, split
    dégénéré) is omitted rather than published as NaN: NaN is not valid strict JSON, breaks most
    dashboards, and pretends a measurement happened when it did not.

    Args:
        metrics: Raw metric mapping.

    Returns:
        A mapping containing only finite floats.
    """
    sanitised: dict[str, float] = {}
    for key, value in metrics.items():
        number = _finite(value)
        if number is not None:
            sanitised[str(key)] = number
    return sanitised


def _sanitise_history(history: Mapping[str, Any]) -> dict[str, list[float]]:
    """Keep only the finite values of a per-epoch metric history.

    Args:
        history: Raw history filled by the metric callbacks (``train_*`` / ``val_*`` keys).

    Returns:
        A mapping of metric name to the finite values recorded during training. A key whose values
        are all undefined disappears entirely, so no curve can be drawn from phantom epochs.
    """
    sanitised: dict[str, list[float]] = {}
    for key, values in dict(history or {}).items():
        finite = [
            number
            for number in (_finite(item) for item in list(values or []))
            if number is not None
        ]
        if finite:
            sanitised[str(key)] = finite
    return sanitised
