"""Model evaluation and error analysis.

The evaluator is the *only* place where predictions are turned into numbers and diagnostics.
It returns a structured :class:`EvaluationResult` instead of printing, so that the same
computation feeds:

* the Markdown report and the figures (``src/evaluation/reports.py``),
* the metrics JSON consumed by CI / a model registry,
* the notebooks (error analysis, per-class breakdown).

Three levels of analysis are produced:

1. **global metrics** — computed with the shared registry (``src/training/losses_metrics.py``),
2. **per-class diagnostics** — precision / recall / F1 / support and the confusion matrix,
3. **error analysis** — the worst misclassified rows, ranked by predicted confidence, which is
   what actually drives the next iteration of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of an evaluation.

    Attributes:
        task: Learning task.
        split: Split that was scored (``test``, ``val``, ...).
        primary_metric: Name of the metric driving the decision.
        metrics: Metric name -> value.
        predictions: Row-level predictions (``y_true``, ``y_pred``, probabilities, ``is_error``).
        labels: Ordered class labels.
        confusion_matrix: Raw confusion matrix (rows = truth, columns = prediction).
        classification_report: Per-class precision / recall / F1 / support.
        per_class: Same information as a DataFrame (convenient for reports).
        errors: Misclassified rows, sorted by descending confidence.
        curves: Points of the ROC / precision-recall / calibration curves.
        feature_importance: Feature -> importance (when the model exposes one).
        extras: Free-form payload (thresholds, budgets, model summary, ...).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    confusion_matrix: np.ndarray | None = None
    classification_report: dict[str, Any] = field(default_factory=dict)
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def primary_value(self) -> float:
        """Value of the primary metric (NaN when unavailable)."""
        return float(self.metrics.get(self.primary_metric, float("nan")))

    @property
    def error_rate(self) -> float:
        """Share of misclassified rows."""
        if self.predictions.empty or "is_error" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["is_error"].mean())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result (JSON friendly, heavy frames reduced to summaries)."""
        return {
            "task": self.task,
            "split": self.split,
            "n_samples": self.n_samples,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "labels": self.labels,
            "confusion_matrix": None
            if self.confusion_matrix is None
            else self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "error_rate": self.error_rate,
            "n_errors": int(len(self.errors)),
            "feature_importance_top": (
                self.feature_importance.head(15).to_dict(orient="records")
                if not self.feature_importance.empty
                else []
            ),
            "extras": self.extras,
        }


class Evaluator:
    """Score a fitted model and produce diagnostics."""

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = 25,
    ) -> None:
        """Inject the model and the metric policy.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of worst errors kept for the error analysis.
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self.primary_metric = str(self.config.get("primary", "accuracy"))
        self.metric_names = [
            self.primary_metric,
            *[str(name) for name in self.config.get("secondary", []) or []],
        ]

    @classmethod
    def from_config(
        cls, model: BaseModel, config: Mapping[str, Any], paths: ProjectPaths | None = None
    ) -> Evaluator:
        """Build an evaluator from a root configuration mapping.

        Args:
            model: Fitted model.
            config: Root configuration.
            paths: Optional project layout.

        Returns:
            The configured evaluator.
        """
        return cls(
            model,
            metrics_config=dict(config.get("metrics", {}) or {}),
            paths=paths,
            task=config.get("metrics", {}).get("task") if config.get("metrics") else None,
        )

    # ------------------------------------------------------------------ evaluation ------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any] | None = None,
        *,
        split: str = "test",
        context: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Score the model on a feature matrix.

        Args:
            X: Preprocessed features.
            y: Ground truth labels (required for supervised evaluation).
            split: Split name, used in logs and artefacts.
            context: Optional raw/enriched rows aligned with ``X`` (identifiers, raw features),
                used to make the error analysis readable.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        if y is None:
            msg = "Classification evaluation requires ground truth labels"
            raise ValueError(msg)

        labels_series = pd.Series(np.asarray(y)).reset_index(drop=True)
        predictions = self.model.predict(X)
        probabilities = self._safe_probabilities(X)

        predictions_frame = self._predictions_frame(
            labels_series, predictions, probabilities, context
        )
        metrics = self._compute_metrics(labels_series, predictions, probabilities)
        matrix, label_names = self._confusion(labels_series, predictions)
        report, per_class = self._class_report(labels_series, predictions, label_names)
        errors = self._error_analysis(predictions_frame, context)
        curves = self._curves(labels_series, probabilities, predictions_frame)
        importance = self.feature_importance(X, y=labels_series)

        result = EvaluationResult(
            task=self.task,
            split=split,
            primary_metric=self.primary_metric,
            metrics=metrics,
            predictions=predictions_frame,
            labels=label_names,
            confusion_matrix=matrix,
            classification_report=report,
            per_class=per_class,
            errors=errors,
            curves=curves,
            feature_importance=importance,
            extras={
                "model": self.model.summary(),
                "n_features": int(X.shape[1]),
                "supports_proba": bool(getattr(self.model, "supports_proba", False)),
                "class_balance": {
                    str(key): round(float(value), 4)
                    for key, value in labels_series.value_counts(normalize=True).items()
                },
            },
            n_samples=int(len(X)),
        )
        logger.info(
            "Evaluation on '{}' | n={} {}={} error_rate={:.3f}",
            split,
            result.n_samples,
            self.primary_metric,
            round(result.primary_value, 5),
            result.error_rate,
        )
        return result

    def compare_to_baseline(
        self, X: pd.DataFrame, y: pd.Series | Sequence[Any], *, baseline: str = "dummy"
    ) -> dict[str, float]:
        """Score a trivial baseline on the same data, to prove the model adds value.

        Args:
            X: Preprocessed features (unused by most baselines, kept for API symmetry).
            y: Ground truth.
            baseline: ``dummy`` (most frequent class) or ``random``.

        Returns:
            The baseline metrics, prefixed with ``baseline_``.
        """
        from sklearn.dummy import DummyClassifier

        estimator = DummyClassifier(strategy="most_frequent" if baseline == "dummy" else "uniform")
        estimator.fit(X, np.asarray(y))
        predictions = estimator.predict(X)
        probabilities = estimator.predict_proba(X)
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(y_true=np.asarray(y), y_pred=predictions, y_proba=probabilities, X=X)
        )
        baseline_metrics = {f"baseline_{name}": _to_float(value) for name, value in values.items()}
        logger.info("Baseline '{}' | {}", baseline, baseline_metrics)
        return baseline_metrics

    def threshold_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any],
        thresholds: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Business-facing trade-off table: what happens when the decision threshold moves.

        Args:
            X: Preprocessed features.
            y: Ground truth.
            thresholds: Thresholds to explore (defaults to a 0.05..0.95 grid).

        Returns:
            A DataFrame with precision, recall, F1, flagged volume and captured positives per
            threshold. This is the table to show a business stakeholder.
        """
        probabilities = self._safe_probabilities(X)
        if probabilities is None or probabilities.ndim < 2:
            logger.warning("Threshold analysis requires class probabilities; skipped")
            return pd.DataFrame()
        scores = probabilities[:, 1]
        truth = np.asarray(y).astype(int)
        grid = (
            list(thresholds)
            if thresholds is not None
            else list(np.round(np.arange(0.05, 0.96, 0.05), 2))
        )

        rows: list[dict[str, float]] = []
        for threshold in grid:
            flagged = (scores >= float(threshold)).astype(int)
            true_positives = int(((flagged == 1) & (truth == 1)).sum())
            false_positives = int(((flagged == 1) & (truth == 0)).sum())
            false_negatives = int(((flagged == 0) & (truth == 1)).sum())
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
                    "flagged": int(flagged.sum()),
                    "flagged_rate": round(float(flagged.mean()), 4),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
            )
        return pd.DataFrame(rows)

    def feature_importance(
        self, X: pd.DataFrame, *, y: pd.Series | Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Extract a feature importance ranking when the model exposes one.

        Two mechanisms are supported: a native ``feature_importances_`` / ``coef_`` attribute,
        and permutation importance as a model-agnostic fallback.

        Args:
            X: Features used for permutation importance (small frames only).
            y: Ground truth, required by permutation importance.

        Returns:
            A DataFrame ``feature | importance | method`` sorted by descending importance.
        """
        estimator = getattr(self.model, "estimator_", None) or getattr(self.model, "model_", None)
        native = getattr(estimator, "feature_importances_", None)
        if native is None:
            native = getattr(estimator, "coef_", None)
            if native is not None:
                native = (
                    np.abs(np.asarray(native)).mean(axis=0)
                    if np.ndim(native) > 1
                    else np.abs(np.asarray(native)).ravel()
                )
        if native is not None and len(native) == X.shape[1]:
            frame = pd.DataFrame(
                {
                    "feature": list(X.columns),
                    "importance": np.asarray(native, dtype="float64"),
                    "method": "native",
                }
            )
            return frame.sort_values("importance", ascending=False).reset_index(drop=True)

        try:
            from sklearn.inspection import permutation_importance
        except ImportError:  # pragma: no cover
            return pd.DataFrame()
        if y is None:
            logger.warning("Permutation importance requires ground truth; skipped")
            return pd.DataFrame()
        sample_X, sample_y = X, np.asarray(y)
        if len(sample_X) > 1500:
            sample_X = sample_X.sample(n=1500, random_state=0)
            sample_y = np.asarray(y)[sample_X.index.to_numpy()]
        try:
            scoring = permutation_importance(
                _SklearnAdapter(self.model), sample_X, sample_y, n_repeats=3, random_state=0
            )
        except Exception as exc:  # noqa: BLE001 - importance is a diagnostic, never blocking
            logger.warning("Permutation importance unavailable: {}", exc)
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "feature": list(sample_X.columns),
                "importance": np.asarray(scoring.importances_mean, dtype="float64"),
                "method": "permutation",
            }
        )
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    def save_metrics(self, result: EvaluationResult, path: str | Path | None = None) -> Path:
        """Persist the evaluation metrics as JSON.

        Args:
            result: Evaluation result.
            path: Destination file (defaults to ``artifacts/metrics/evaluation_metrics.json``).

        Returns:
            The written path.
        """
        destination = Path(path) if path else self.paths.metrics_dir / "evaluation_metrics.json"
        return write_json(destination, result.to_dict())

    # ------------------------------------------------------------------ internals -------
    def _safe_probabilities(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return class probabilities when the model supports them."""
        if not getattr(self.model, "supports_proba", False):
            return None
        try:
            return np.asarray(self.model.predict_proba(X))
        except (NotImplementedError, AttributeError) as exc:
            logger.debug("predict_proba unavailable: {}", exc)
            return None

    def _predictions_frame(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        context: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Assemble a row-level prediction frame (the base of every diagnostic)."""
        frame = pd.DataFrame({"y_true": truth.to_numpy(), "y_pred": np.asarray(predictions)})
        if probabilities is not None and probabilities.ndim == 2:
            for index in range(probabilities.shape[1]):
                frame[f"probability_class_{index}"] = probabilities[:, index]
            frame["probability_positive"] = (
                probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            )
            frame["confidence"] = probabilities.max(axis=1)
        frame["is_error"] = (frame["y_true"].astype(str) != frame["y_pred"].astype(str)).astype(int)
        frame = frame.reset_index(drop=True)

        if context is not None and len(context) == len(frame):
            identifiers = [column for column in context.columns if column in _CONTEXT_COLUMNS]
            if identifiers:
                frame = pd.concat(
                    [context.loc[:, identifiers].reset_index(drop=True), frame], axis=1
                )
        return frame

    def _compute_metrics(
        self, truth: pd.Series, predictions: np.ndarray, probabilities: np.ndarray | None
    ) -> dict[str, float]:
        """Compute the configured metrics through the shared registry."""
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(y_true=truth.to_numpy(), y_pred=predictions, y_proba=probabilities)
        )
        return {name: _to_float(value) for name, value in values.items()}

    def _confusion(self, truth: pd.Series, predictions: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Compute the confusion matrix and its labels."""
        label_names = sorted(
            {str(value) for value in truth.unique()}
            | {str(value) for value in np.asarray(predictions)}
        )
        matrix = confusion_matrix(
            truth.astype(str), np.asarray(predictions).astype(str), labels=label_names
        )
        return matrix, label_names

    def _class_report(
        self, truth: pd.Series, predictions: np.ndarray, label_names: Sequence[str]
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Build the per-class report (dict + DataFrame)."""
        from sklearn.metrics import classification_report

        report = classification_report(
            truth.astype(str),
            np.asarray(predictions).astype(str),
            labels=list(label_names),
            output_dict=True,
            zero_division=0,
        )
        rows: list[dict[str, Any]] = []
        for label in label_names:
            entry = report.get(str(label), {})
            support = float(truth.astype(str).eq(str(label)).sum())
            predicted_support = float(pd.Series(predictions).astype(str).eq(str(label)).sum())
            rows.append(
                {
                    "class": str(label),
                    "precision": round(float(entry.get("precision", 0.0)), 4),
                    "recall": round(float(entry.get("recall", 0.0)), 4),
                    "f1": round(float(entry.get("f1-score", 0.0)), 4),
                    "support": int(support),
                    "predicted_support": int(predicted_support),
                    "under_prediction": int(support - predicted_support),
                }
            )
        return report, pd.DataFrame(rows)

    def _error_analysis(
        self, predictions_frame: pd.DataFrame, context: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Rank the misclassified rows by confidence (the most damaging errors first)."""
        if predictions_frame.empty or "is_error" not in predictions_frame.columns:
            return pd.DataFrame()
        errors = predictions_frame.loc[predictions_frame["is_error"] == 1].copy()
        if errors.empty:
            return errors
        sort_column = "confidence" if "confidence" in errors.columns else "probability_positive"
        if sort_column in errors.columns:
            errors = errors.sort_values(sort_column, ascending=False)
        errors = errors.head(self.top_k_errors).reset_index(drop=True)

        if context is not None and not errors.empty:
            feature_columns = [
                column for column in context.columns if column not in {"y_true", "y_pred"}
            ]
            join_key = _CONTEXT_COLUMNS[0] if _CONTEXT_COLUMNS[0] in errors.columns else None
            if join_key and join_key in context.columns:
                merged = errors.merge(context.loc[:, feature_columns], on=join_key, how="left")
                return merged.head(self.top_k_errors)
        return errors

    def _curves(
        self, truth: pd.Series, probabilities: np.ndarray | None, predictions_frame: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute the points of the ROC, precision-recall and calibration curves."""
        curves: dict[str, Any] = {}
        if probabilities is None or probabilities.ndim < 2 or probabilities.shape[1] < 2:
            return curves
        scores = probabilities[:, 1]
        labels = truth.astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            return curves

        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        fraction_positive, mean_predicted = calibration_curve(
            labels, scores, n_bins=10, strategy="quantile"
        )

        curves["roc"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
        curves["precision_recall"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
        curves["calibration"] = {
            "fraction_positive": fraction_positive.tolist(),
            "mean_predicted": mean_predicted.tolist(),
        }
        curves["score_distribution"] = {
            "positive": scores[labels == 1].tolist()[:500],
            "negative": scores[labels == 0].tolist()[:500],
        }
        return curves


class _SklearnAdapter:
    """Minimal scikit-learn facade used by ``permutation_importance``."""

    def __init__(self, model: BaseModel) -> None:
        """Wrap a :class:`BaseModel`.

        Args:
            model: Fitted model.
        """
        self.model = model

    def score(self, X: pd.DataFrame, y: Any = None) -> float:  # noqa: ARG002
        """Return the accuracy of the wrapped model (permutation importance only needs a scalar).

        Args:
            X: Features.
            y: Ground truth.

        Returns:
            The accuracy.
        """
        predictions = self.model.predict(X)
        return float(np.mean(np.asarray(predictions).astype(str) == np.asarray(y).astype(str)))

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> _SklearnAdapter:
        """Delegate to the wrapped model (``permutation_importance`` requires a ``fit`` method).

        Args:
            X: Features.
            y: Ground truth.
            **fit_params: Extra keyword arguments accepted by the wrapped model.

        Returns:
            ``self``, as expected by the scikit-learn estimator protocol.
        """
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate to the wrapped model.

        Args:
            X: Features.

        Returns:
            The predictions.
        """
        return self.model.predict(X)


#: Raw columns worth keeping next to a prediction for error analysis.
_CONTEXT_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "id",
    "record_id",
    "contract_type",
    "tenure_months",
    "monthly_charges",
    "satisfaction_score",
    "support_tickets_6m",
    "payment_method",
    "internet_service",
)


def _to_float(value: Any) -> float:
    """Coerce a metric value to ``float`` (NaN when impossible)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")
