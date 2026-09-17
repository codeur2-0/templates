"""Framework-agnostic model contract.

Everything in the project depends on :class:`BaseModel` and on nothing else: the trainer, the
evaluation pipeline, the predictor and the notebooks manipulate this abstraction, never a
framework class. Swapping scikit-learn for XGBoost or PyTorch is therefore a configuration change
(``model.class`` in ``conf/model/default.yaml``), not a refactor.

The contract is deliberately small but strict:

* ``fit`` trains the model and returns a :class:`FitResult` (metrics, history, timing),
* ``predict`` / ``predict_proba`` refuse to run on an unfitted model (no silently wrong output),
* every input frame is aligned on ``feature_names`` (missing column -> explicit ``ValueError``,
  extra columns ignored, order restored),
* ``save`` / ``load`` round-trip faithfully: the reloaded model predicts exactly like the in-memory
  one, and :class:`ModelCard` documents the run for the reviewers.

Two dataclasses complete the contract: :class:`FitResult` (what a training run produced) and
:class:`ModelCard` (what must be archived for traceability).
"""

from __future__ import annotations

import platform
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Libraries whose version is archived in the model card when they are already imported. Reading a
#: version from ``sys.modules`` avoids importing a heavy framework just for metadata (and avoids
#: mixing PyTorch and TensorFlow runtimes in one process).
TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "scipy",
    "xgboost",
    "lightgbm",
    "torch",
    "tensorflow",
    "keras",
    "joblib",
    "hydra",
    "pandera",
    "pydantic",
)

#: Tasks that require a target column (an unsupervised task is fitted with ``y=None``).
SUPERVISED_TASKS: frozenset[str] = frozenset(
    {"binary", "multiclass", "regression", "forecasting", "ranking"}
)

#: Tasks whose predictions are normalised probabilities.
#: ``ranking`` en fait partie : un modèle de classement pointwise score chaque couple
#: (utilisateur, candidat) par sa probabilité de pertinence, et c'est ce score — jamais la
#: classe prédite — qui détermine l'ordre publié.
PROBABILITY_TASKS: frozenset[str] = frozenset({"binary", "multiclass", "ranking"})


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 (second precision)."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def library_versions() -> dict[str, str]:
    """Collect the versions of the libraries already loaded in this process.

    Returns:
        Mapping of library name to version, always containing ``python`` and ``platform``.
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in TRACKED_LIBRARIES:
        module = sys.modules.get(name)
        if module is None:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _is_finite(value: Any) -> bool:
    """Return whether a metric value is a real, finite number.

    Args:
        value: Raw value produced by the metric registry.

    Returns:
        ``True`` for finite integers and floats, ``False`` for ``None``, NaN, infinities and
        anything non-numeric. A metric that is not finite must never reach an artefact: NaN is not
        valid strict JSON and carries no information for whoever reads the run.
    """
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def score_diagnostics(scores: Any) -> dict[str, float]:
    """Summarise a model's own output when no target is available.

    Unsupervised tasks (anomaly detection, density clustering) cannot compute a supervised metric
    during training: there is nothing to compare against. Rather than reporting NaN, the model
    describes the distribution of its own score on the training data, which is exactly what an
    engineer needs to sanity-check a fit (is the score degenerate? is it concentrated?).

    Args:
        scores: Predictions or scores computed on the training features.

    Returns:
        Finite descriptive statistics of the score distribution; empty when the scores carry no
        numeric signal (e.g. discrete cluster identifiers are still numeric, but a textual output
        is not).
    """
    flat = pd.to_numeric(
        pd.Series(np.asarray(scores, dtype=object).ravel()), errors="coerce"
    ).dropna()
    if flat.empty:
        return {}
    values = flat.astype("float64")
    return {
        "score_mean": float(values.mean()),
        "score_std": float(values.std(ddof=0)),
        "score_p50": float(values.quantile(0.50)),
        "score_p95": float(values.quantile(0.95)),
        "score_p99": float(values.quantile(0.99)),
        "score_min": float(values.min()),
        "score_max": float(values.max()),
    }


def _sanitise(value: Any) -> Any:
    """Convert NumPy/pandas scalars to JSON-serialisable Python objects.

    Args:
        value: Arbitrary payload (metrics, parameters, ...).

    Returns:
        A JSON-safe copy of ``value``.
    """
    if isinstance(value, Mapping):
        return {str(key): _sanitise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitise(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_sanitise(item) for item in value.tolist()]
    if isinstance(value, (str, int)) or value is None:
        return value
    return str(value)


@dataclass
class FitResult:
    """What one training run produced.

    Attributes:
        model_name: Class name of the model (``SklearnModel``, ``PyTorchModel``, ...).
        algorithm: Algorithm identifier resolved by the factory (``random_forest``, ``mlp``, ...).
        metrics: Metrics computed on the training split (plain names) plus, once the trainer has
            scored the validation split, the ``val_*`` entries.
        history: Per-epoch metric history, keyed by ``train_*`` / ``val_*`` / ``loss`` names.
        n_samples: Number of training rows.
        n_features: Number of features handed to the model.
        duration_seconds: Wall-clock duration of the fit.
        started_at: ISO timestamp of the beginning of the fit.
        finished_at: ISO timestamp of the end of the fit.
        params: Effective hyper-parameters (after resolution of ``auto`` values).
        epochs: Number of epochs / boosting rounds actually run.
        extra: Framework specific payload (best iteration, device, cross-validation, ...).
    """

    model_name: str
    algorithm: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    duration_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    epochs: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return _sanitise(asdict(self))

    def to_json(self, path: str | Path) -> Path:
        """Write the result as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FitResult:
        """Rebuild a result from its JSON payload (unknown keys are ignored).

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`FitResult`.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{str(key): value for key, value in payload.items() if key in known})


@dataclass
class ModelCard:
    """Traceability document archived next to the model artefact.

    Attributes:
        model_name: Class name of the model.
        framework: Framework identifier (``sklearn``, ``xgboost``, ``pytorch``, ...).
        algorithm: Algorithm identifier.
        task: Learning task (``binary``, ``regression``, ``clustering``, ...).
        target_name: Business target column, ``None`` for unsupervised tasks.
        feature_names: Exact features expected at inference time, in order.
        params: Effective hyper-parameters.
        metrics: Metrics of the run (training and validation).
        library_versions: Versions of the libraries used, for reproducibility.
        created_at: ISO timestamp of the card creation.
        n_samples: Number of training rows.
        n_features: Number of features.
        artifact: File name of the serialised model, when known.
        notes: Human readable remarks (data leakage guards, known limitations, ...).
    """

    model_name: str
    framework: str
    algorithm: str = ""
    task: str = ""
    target_name: str | None = None
    feature_names: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    n_samples: int = 0
    n_features: int = 0
    artifact: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return _sanitise(asdict(self))

    def to_json(self, path: str | Path) -> Path:
        """Write the card as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCard:
        """Rebuild a card from its JSON payload (unknown keys are ignored).

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`ModelCard`.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{str(key): value for key, value in payload.items() if key in known})


class BaseModel(ABC):
    """Abstract model every stack implements.

    Subclasses provide four things: the training implementation (:meth:`_fit`), the prediction
    implementation (:meth:`_predict`, and optionally :meth:`_predict_proba`), and the persistence
    pair (:meth:`save` / :meth:`load`). Everything else — contract checks, feature alignment,
    callbacks, metric computation, model card — is implemented once here so that all stacks behave
    identically.
    """

    #: Framework identifier, overridden by every concrete stack.
    framework: ClassVar[str] = "base"

    #: Default file name used when ``save`` receives a directory or no name.
    default_model_file: ClassVar[str] = "model.joblib"

    #: Whether the stack trains by epochs (deep learning, boosting rounds) or in one shot.
    epochs_based: ClassVar[bool] = False

    def __init__(
        self,
        *,
        algorithm: str = "",
        params: Mapping[str, Any] | None = None,
        task: str = "binary",
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        random_state: int = 42,
        supports_proba: bool | None = None,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the model contract.

        Args:
            algorithm: Algorithm identifier resolved by the factory.
            params: Hyper-parameters forwarded to the underlying estimator.
            task: Learning task (``binary``, ``multiclass``, ``regression``, ``clustering``, ...).
            feature_names: Features expected at inference time, in order.
            target_name: Business target column (``None`` for unsupervised tasks).
            random_state: Reproducibility seed.
            supports_proba: Explicit probability support; ``None`` lets the implementation decide
                (typically from the fitted estimator's capabilities).
            config: Full application configuration (metrics, train options) used to compute the
                training metrics and to document the run.
            name: Human readable model name (defaults to the algorithm identifier).
        """
        self.algorithm = str(algorithm)
        self.params: dict[str, Any] = dict(params or {})
        self.task = str(task)
        self.feature_names: list[str] = [str(column) for column in (feature_names or [])]
        self.target_name = target_name
        self.random_state = int(random_state)
        self.config: dict[str, Any] = dict(config or {})
        self.name = str(name or algorithm or type(self).__name__)
        self._supports_proba = supports_proba
        self._is_fitted = False
        self.fit_result_: FitResult | None = None
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------ identité ----------
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained at least once."""
        return self._is_fitted

    @property
    def state(self) -> str:
        """Training state, used in logs and reports (``fitted`` / ``unfitted``)."""
        return "fitted" if self._is_fitted else "unfitted"

    @property
    def supports_proba(self) -> bool:
        """Whether :meth:`predict_proba` produces meaningful probabilities."""
        if self._supports_proba is not None:
            return bool(self._supports_proba)
        return self.task in PROBABILITY_TASKS

    @property
    def is_supervised(self) -> bool:
        """Whether the task requires a target column."""
        return self.task in SUPERVISED_TASKS

    def summary(self) -> str:
        """Return a one-line, human readable description of the model.

        Returns:
            The summary line (class name, algorithm, task, feature count, state).
        """
        return repr(self)

    def _repr_fields(self) -> list[str]:
        """Fields displayed by :meth:`__repr__`; stacks append their own details."""
        return [
            f"algorithm={self.algorithm or '-'}",
            f"task={self.task}",
            f"features={len(self.feature_names)}",
            f"state={self.state}",
        ]

    def __repr__(self) -> str:
        """Return the log-friendly representation."""
        return f"{type(self).__name__}({', '.join(self._repr_fields())})"

    # ------------------------------------------------------------------ contrat -----------
    def check_is_fitted(self) -> None:
        """Assert that the model is trained.

        Raises:
            RuntimeError: When the model has never been fitted (or was reset).
        """
        if not self._is_fitted:
            msg = (
                f"{type(self).__name__} is not fitted: call fit(X, y) before predicting "
                "(or load a persisted artefact with load_model)."
            )
            raise RuntimeError(msg)

    def align_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Align an input frame on the training feature contract.

        Extra columns are ignored (an inference payload may carry identifiers), missing columns
        raise, and the column order is restored so that the matrix always matches what the model
        saw during training.

        Args:
            X: Input frame (or matrix, returned unchanged when there is no feature contract).

        Returns:
            The aligned frame.

        Raises:
            ValueError: When at least one expected feature is missing.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        if not self.feature_names:
            return X
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            preview = ", ".join(missing[:8])
            msg = (
                f"{len(missing)} missing feature(s) in the input frame: {preview}"
                f"{'...' if len(missing) > 8 else ''}. Expected {len(self.feature_names)} "
                "column(s) in the training order."
            )
            raise ValueError(msg)
        extra = [str(column) for column in X.columns if column not in set(self.feature_names)]
        if extra:
            logger.debug("Ignoring {} extra column(s): {}", len(extra), extra[:8])
        return X.loc[:, self.feature_names]

    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert an aligned frame to a float matrix.

        Args:
            X: Aligned input.

        Returns:
            A contiguous ``float64`` matrix.

        Raises:
            ValueError: When the frame still holds non-numeric values (preprocessing not applied).
        """
        frame = self.align_features(X)
        if isinstance(frame, pd.DataFrame):
            try:
                return frame.to_numpy(dtype="float64")
            except (TypeError, ValueError) as error:
                non_numeric = [
                    str(column)
                    for column in frame.columns
                    if not pd.api.types.is_numeric_dtype(frame[column])
                ]
                msg = (
                    f"Non-numeric column(s) reached the model: {non_numeric[:8]}. "
                    "Apply the fitted preprocessing pipeline before predicting."
                )
                raise ValueError(msg) from error
        return np.ascontiguousarray(np.asarray(frame, dtype="float64"))

    # ------------------------------------------------------------------ entraînement ------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        groups: Sequence[Any] | np.ndarray | None = None,
        groups_val: Sequence[Any] | np.ndarray | None = None,
        callbacks: Sequence[Any] | None = None,
    ) -> FitResult:
        """Train the model, fire the callbacks and return the run result.

        This is a *template method*: it owns the contract checks, the timing, the callback
        orchestration and the metric computation, while the stack implements :meth:`_fit`.

        Args:
            X: Training features (already preprocessed).
            y: Training target; ``None`` for unsupervised tasks.
            X_val: Validation features, used for early stopping and ``val_*`` metrics.
            y_val: Validation target.
            groups: Group identifier per training row. A metric computed per group (a ranking
                metric averages per-user scores) is **skipped** by the registry without it, since
                the preprocessed matrices no longer carry the identifier column.
            groups_val: Same, for the validation split.
            callbacks: Objects implementing ``on_train_begin`` / ``on_epoch_end`` /
                ``on_train_end`` (see :mod:`src.training.callbacks`).

        Returns:
            The :class:`FitResult` of the run.

        Raises:
            ValueError: When a supervised task is fitted without a target.
        """
        started_at = _utc_now()
        if self.is_supervised and y is None:
            msg = (
                f"Task '{self.task}' is supervised but no target was provided to fit(). "
                "Pass y (or set metrics.task to an unsupervised task)."
            )
            raise ValueError(msg)

        callbacks = list(callbacks or [])
        features = self.align_features(X)
        self.feature_names = (
            [str(column) for column in features.columns]
            if isinstance(features, pd.DataFrame)
            else self.feature_names
        )
        context = self._callback_context(epochs=self._planned_epochs(), callbacks=callbacks)
        # Les groupes voyagent avec le contexte : les stacks les lisent pour calculer leurs
        # métriques d'époque, et ils ne sont jamais sérialisés (contrairement à ``extra``).
        context.groups = groups
        context.groups_val = groups_val
        for callback in callbacks:
            callback.on_train_begin(context)

        self._fit(
            features,
            y,
            X_val=None if X_val is None else self.align_features(X_val),
            y_val=y_val,
            context=context,
            callbacks=callbacks,
        )
        self._is_fitted = True

        result = FitResult(
            model_name=type(self).__name__,
            algorithm=self.algorithm,
            metrics=self.training_metrics(features, y, groups=groups),
            history=dict(context.history),
            n_samples=len(features),
            n_features=int(np.shape(features)[1]),
            started_at=started_at,
            finished_at=_utc_now(),
            params=dict(self.params),
            epochs=max(int(context.epoch + 1), 1),
            extra=dict(context.extra),
        )
        for callback in callbacks:
            callback.on_train_end(context)
        self.fit_result_ = result
        logger.info(
            "Fit finished | {} | epochs={} | {}",
            self.summary(),
            result.epochs,
            {key: round(value, 5) for key, value in list(result.metrics.items())[:5]},
        )
        return result

    def training_metrics(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None,
        *,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Score the fitted model on its training data (diagnostic, never a decision metric).

        Args:
            X: Training features.
            y: Training target (``None`` for unsupervised tasks).
            groups: Group identifier per row, required by the per-group metrics.

        Returns:
            Mapping of metric name to **finite** value. Metrics the registry cannot compute
            without a target are dropped rather than returned as NaN; when nothing finite remains
            (unsupervised task with a label-based registry), the score distribution is reported
            instead — see :func:`score_diagnostics`.
        """
        calculator = self._metric_calculator()
        if calculator is None:
            return {}
        predictions = self._predict(X)
        probabilities = None
        if self.supports_proba:
            try:
                probabilities = self._predict_proba(X)
            except (NotImplementedError, ValueError) as error:
                logger.debug("Training probabilities unavailable: {}", error)
        values = calculator.evaluate(
            self._metric_inputs(
                y_true=y,
                y_pred=self._ordered_scores(predictions, probabilities),
                y_proba=probabilities,
                X=X,
                groups=groups,
            )
        )
        finite = {
            str(name): float(value) for name, value in dict(values).items() if _is_finite(value)
        }
        if finite:
            return finite
        logger.debug("No finite training metric (unsupervised task): reporting score diagnostics")
        return score_diagnostics(predictions)

    def _metric_calculator(self) -> Any:
        """Build the metric calculator declared in the configuration (``None`` when absent)."""
        try:
            from src.training.losses_metrics import MetricCalculator
        except ImportError:  # pragma: no cover - modality without a metric registry
            logger.debug("Metric registry unavailable: training metrics skipped")
            return None
        node = dict(self.config.get("metrics") or {})
        if not node:
            return None
        try:
            return MetricCalculator.from_config(self.config)
        except (ValueError, KeyError) as error:
            logger.warning("Metric selection rejected ({}): training metrics skipped", error)
            return None

    def _ordered_scores(self, predictions: Any, probabilities: Any) -> Any:
        """Return the prediction the metric registry must read as an **ordering**.

        In a ranking task ``y_pred`` is consumed as a continuous score by the group-aware metrics:
        the hard classes of a classifier carry no ordering information, and scoring a list with
        them produces ties everywhere and a meaningless NDCG. The positive-class probability
        therefore becomes the score. Every other task keeps its hard predictions, so accuracy and
        F1 stay computed on classes.

        Args:
            predictions: Hard predictions (labels or values).
            probabilities: Probability matrix, when the model exposes one.

        Returns:
            The array the registry should read as ``y_pred``.
        """
        if str(self.task) != "ranking" or probabilities is None:
            return predictions
        matrix = np.asarray(probabilities, dtype="float64")
        return matrix[:, -1].ravel() if matrix.ndim == 2 else matrix.ravel()

    def _metric_inputs(
        self, *, y_true: Any, y_pred: Any, y_proba: Any, X: Any, groups: Any = None
    ) -> Any:
        """Build the :class:`MetricInputs` payload expected by the metric registry."""
        from src.training.losses_metrics import MetricInputs, metric_extra_from_config

        return MetricInputs(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X=X,
            groups=groups,
            extra=metric_extra_from_config(self.config),
        )

    def _planned_epochs(self) -> int:
        """Number of epochs announced to the callbacks (``1`` for single-shot estimators)."""
        if not self.epochs_based:
            return 1
        train_node = dict(self.config.get("train") or {})
        return max(int(train_node.get("epochs", 1) or 1), 1)

    def _callback_context(self, *, epochs: int, callbacks: Sequence[Any]) -> Any:
        """Build the :class:`CallbackContext` shared by every hook.

        Args:
            epochs: Total number of epochs / rounds.
            callbacks: Callbacks (unused here, kept for symmetry with the emission helpers).

        Returns:
            A fresh callback context.
        """
        del callbacks
        from src.training.callbacks import CallbackContext

        return CallbackContext(
            model_name=type(self).__name__,
            params=dict(self.params),
            epochs=epochs,
        )

    def emit_epoch(
        self,
        context: Any,
        callbacks: Sequence[Any],
        logs: Mapping[str, float],
        *,
        epoch: int,
    ) -> bool:
        """Fire ``on_epoch_end`` for one epoch and accumulate the history.

        Args:
            context: Mutable callback context.
            callbacks: Registered callbacks.
            logs: Metrics of the epoch (``loss``, ``train_*``, ``val_*`` keys).
            epoch: 0-based epoch index.

        Returns:
            ``True`` when a callback requested an early stop.
        """
        context.epoch = int(epoch)
        context.logs = {str(key): float(value) for key, value in logs.items() if value is not None}
        for callback in callbacks:
            callback.on_epoch_end(context)
        return bool(context.stopped_early)

    # ------------------------------------------------------------------ prédiction --------
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict one value per row.

        Args:
            X: Features (a frame is aligned on the training contract first).

        Returns:
            A 1-D array of predictions (class labels, values or cluster ids).

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.check_is_fitted()
        predictions = self._predict(self.align_features(X))
        return np.asarray(predictions).ravel()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class / cluster probabilities.

        Args:
            X: Features (a frame is aligned on the training contract first).

        Returns:
            A ``(n_samples, n_classes)`` array whose rows sum to 1.

        Raises:
            NotImplementedError: When the algorithm exposes no probabilities (checked first: the
                capability is a property of the algorithm, not of its training state).
            RuntimeError: When the model is not fitted.
        """
        if not self.supports_proba:
            msg = (
                f"{type(self).__name__}(algorithm={self.algorithm}) does not produce "
                "probabilities for this task; use predict()."
            )
            raise NotImplementedError(msg)
        self.check_is_fitted()
        probabilities = np.asarray(self._predict_proba(self.align_features(X)), dtype="float64")
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return probabilities

    # ------------------------------------------------------------------ traçabilité -------
    def model_card(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> ModelCard:
        """Build the traceability document of this model.

        Args:
            metrics: Metrics to archive (the caller — usually the trainer — decides which ones:
                the card mirrors exactly what is passed, so a test can assert on it).
            artifact: Serialised model file name, when known.
            notes: Extra human readable remarks.

        Returns:
            The :class:`ModelCard`.
        """
        result = self.fit_result_
        payload = {str(key): value for key, value in dict(metrics or {}).items()}
        return ModelCard(
            model_name=type(self).__name__,
            framework=self.framework,
            algorithm=self.algorithm,
            task=self.task,
            target_name=self.target_name,
            feature_names=list(self.feature_names),
            params=dict(self.params),
            metrics={str(key): float(value) for key, value in payload.items()},
            library_versions=library_versions(),
            created_at=_utc_now(),
            n_samples=int(result.n_samples) if result else 0,
            n_features=int(result.n_features) if result else len(self.feature_names),
            artifact=artifact,
            notes=[str(note) for note in (notes or [])],
        )

    # ------------------------------------------------------------------ à implémenter -----
    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None,
        *,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        context: Any = None,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        """Train the underlying estimator (implemented by each stack).

        Args:
            X: Aligned training features.
            y: Training target (``None`` for unsupervised tasks).
            X_val: Aligned validation features.
            y_val: Validation target.
            context: Mutable callback context (epoch, logs, history, early stop flag).
            callbacks: Callbacks to fire once per epoch / boosting round.
        """

    @abstractmethod
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Produce one prediction per row (implemented by each stack).

        Args:
            X: Aligned features.

        Returns:
            The predictions.
        """

    def _predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Produce class probabilities (optional; only probabilistic algorithms implement it).

        Args:
            X: Aligned features.

        Returns:
            A ``(n_samples, n_classes)`` array.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        del X
        msg = f"{type(self).__name__}(algorithm={self.algorithm}) exposes no probabilities"
        raise NotImplementedError(msg)

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the fitted model.

        Args:
            path: Destination file (a directory receives :attr:`default_model_file`).

        Returns:
            The written path.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Reload a persisted model.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration used to restore runtime options.

        Returns:
            The reloaded model, ready to predict.
        """

    # ------------------------------------------------------------------ helpers -----------
    def _resolve_path(self, path: str | Path) -> Path:
        """Normalise a persistence path (a directory receives the default file name).

        Args:
            path: File or directory.

        Returns:
            The destination file, with its parent directory created.
        """
        destination = Path(path)
        if destination.is_dir() or not destination.suffix:
            destination = destination / self.default_model_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _effective_params(self) -> dict[str, Any]:
        """Copy of the hyper-parameters (stacks resolve ``auto`` values inside ``_fit``)."""
        return dict(self.params)

    def _seed_everything(self) -> None:
        """Seed Python, NumPy and the project's global state before training."""
        try:
            from src.utils.utils import set_seed

            set_seed(self.random_state)
        except ImportError:  # pragma: no cover - project without the utils layer
            np.random.seed(self.random_state)


__all__ = [
    "PROBABILITY_TASKS",
    "SUPERVISED_TASKS",
    "BaseModel",
    "FitResult",
    "ModelCard",
    "library_versions",
]
