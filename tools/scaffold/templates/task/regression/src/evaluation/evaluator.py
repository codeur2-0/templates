"""Model evaluation and error analysis for a **regression** target.

The evaluator is the *only* place where predictions are turned into numbers and diagnostics.
It returns a structured :class:`EvaluationResult` instead of printing, so that the same
computation feeds:

* the Markdown report and the figures (``src/evaluation/reports.py``),
* the metrics JSON consumed by CI / a model registry,
* the notebooks (error analysis, per-segment breakdown).

Three levels of analysis are produced:

1. **global metrics** — computed with the shared registry (``src/training/losses_metrics.py``):
   RMSE, MAE, R2, MAPE, sMAPE, max error;
2. **segment diagnostics** — the same errors recomputed per district, per energy rating and per
   price bucket, because a global RMSE hides a systematic bias on one segment;
3. **error analysis** — the worst rows ranked by relative error, plus the coverage of the
   business tolerance band (``± 10 %`` by default), which is what actually drives the next
   iteration of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Business tolerance band: a prediction is « acceptable » inside ± this percentage of the truth.
TOLERANCE_PCT = 10.0

#: Number of price buckets used for the segmented analysis.
N_BUCKETS = 8

#: Maximum number of distinct values for a column to be usable as a segmentation axis.
MAX_SEGMENT_CARDINALITY = 12


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of a regression evaluation.

    Attributes:
        task: Learning task (``regression``).
        split: Split that was scored (``test``, ``val``, ...).
        primary_metric: Name of the metric driving the decision (typically ``rmse``).
        metrics: Metric name -> value.
        predictions: Row-level frame (``y_true``, ``y_pred``, ``residual``, ``absolute_error``,
            ``relative_error_pct``, ``within_tolerance``, ``price_bucket`` + context columns).
        labels: Segmentation axes used for :attr:`per_segment` (kept for contract symmetry with
            the classification result, where it holds the class names).
        per_segment: Error statistics per segment (one row per segment value).
        errors: Worst rows, sorted by descending absolute relative error.
        curves: Down-sampled points of the predicted / observed distributions (JSON friendly).
        feature_importance: Feature -> importance (when the model exposes one).
        extras: Free-form payload (residuals, bias, coverage, bucket tables, model summary).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    per_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
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
        """Share of predictions **outside** the business tolerance band.

        A regression has no « misclassification rate »: the closest business reading is the
        proportion of estimates the product would refuse to publish (outside ± tolerance).
        """
        if self.predictions.empty or "within_tolerance" not in self.predictions.columns:
            return float("nan")
        return float(1.0 - self.predictions["within_tolerance"].mean())

    @property
    def coverage(self) -> float:
        """Share of predictions inside the business tolerance band (``1 - error_rate``)."""
        if self.predictions.empty or "within_tolerance" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["within_tolerance"].mean())

    @property
    def bias(self) -> float:
        """Mean signed residual (positive = systematic over-estimation)."""
        if self.predictions.empty or "residual" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["residual"].mean())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result (JSON friendly, heavy frames reduced to summaries)."""
        return {
            "task": self.task,
            "split": self.split,
            "n_samples": self.n_samples,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "segmentation_axes": self.labels,
            "error_rate": self.error_rate,
            "coverage": self.coverage,
            "bias": self.bias,
            "n_errors": int(len(self.errors)),
            "per_segment": (
                self.per_segment.head(40).to_dict(orient="records")
                if not self.per_segment.empty
                else []
            ),
            "feature_importance_top": (
                self.feature_importance.head(15).to_dict(orient="records")
                if not self.feature_importance.empty
                else []
            ),
            "extras": self.extras,
        }


class Evaluator:
    """Score a fitted regressor and produce diagnostics."""

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = 25,
        tolerance_pct: float = TOLERANCE_PCT,
        context_columns: Sequence[str] | None = None,
    ) -> None:
        """Inject the model and the metric policy.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of worst errors kept for the error analysis.
            tolerance_pct: Business tolerance band, in percent of the observed value.
            context_columns: Columns copied from the enriched context into the diagnostic frame
                (identifier, group, timestamp). Defaults to :data:`_CONTEXT_COLUMNS`.
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self.tolerance_pct = float(tolerance_pct)
        self.context_columns: tuple[str, ...] = (
            tuple(str(column) for column in context_columns)
            if context_columns
            else _CONTEXT_COLUMNS
        )
        self.primary_metric = str(self.config.get("primary", "rmse"))
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
        metrics_node = config.get("metrics") or {}
        data_node = config.get("data") or {}
        # Les colonnes de contexte utiles à l'analyse d'erreurs sont déclarées dans la
        # configuration (identifiant, groupe, date) : aucun nom de colonne n'est codé en dur.
        context_columns = tuple(
            str(data_node[key])
            for key in ("id_column", "group_column", "time_column")
            if data_node.get(key)
        )
        return cls(
            model,
            metrics_config=dict(metrics_node),
            paths=paths,
            task=metrics_node.get("task"),
            context_columns=context_columns or None,
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
            y: Observed target values (required for supervised evaluation).
            split: Split name, used in logs and artefacts.
            context: Optional raw/enriched rows aligned with ``X`` (identifiers, raw features),
                used to make the segmented analysis readable.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            ValueError: When the ground truth is missing.
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        if y is None:
            msg = "Regression evaluation requires the observed target values"
            raise ValueError(msg)

        truth = pd.Series(np.asarray(y, dtype="float64")).reset_index(drop=True)
        predictions = np.asarray(self.model.predict(X), dtype="float64").ravel()

        frame = self._predictions_frame(truth, predictions, context)
        metrics = self._compute_metrics(truth, predictions)
        axes = self._segmentation_axes(frame, context)
        per_segment = self._segment_analysis(frame, axes)
        errors = self._error_analysis(frame)
        extras = self._error_statistics(frame)
        curves = self._distribution_points(truth, predictions, frame)
        importance = self.feature_importance(X, y=truth)

        extras.update(
            {
                "model": self.model.summary(),
                "n_features": int(X.shape[1]),
                "segmentation_axes": list(axes),
                "tolerance_pct": self.tolerance_pct,
            }
        )

        result = EvaluationResult(
            task=self.task,
            split=split,
            primary_metric=self.primary_metric,
            metrics=metrics,
            predictions=frame,
            labels=list(axes),
            per_segment=per_segment,
            errors=errors,
            curves=curves,
            feature_importance=importance,
            extras=extras,
            n_samples=int(len(X)),
        )
        logger.info(
            "Evaluation on '{}' | n={} {}={} mape={:.4f} couverture={:.1%}",
            split,
            result.n_samples,
            self.primary_metric,
            round(result.primary_value, 3),
            metrics.get("mape", float("nan")),
            result.coverage,
        )
        return result

    def compare_to_baseline(
        self, X: pd.DataFrame, y: pd.Series | Sequence[Any], *, baseline: str = "dummy"
    ) -> dict[str, float]:
        """Score a trivial baseline on the same data, to prove the model adds value.

        Args:
            X: Preprocessed features (unused by most baselines, kept for API symmetry).
            y: Observed target values.
            baseline: ``dummy`` (median predictor) or ``mean`` (mean predictor).

        Returns:
            The baseline metrics, prefixed with ``baseline_``.
        """
        from sklearn.dummy import DummyRegressor

        estimator = DummyRegressor(strategy="median" if baseline == "dummy" else "mean")
        estimator.fit(X, np.asarray(y, dtype="float64"))
        predictions = estimator.predict(X)
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(y_true=np.asarray(y, dtype="float64"), y_pred=predictions, X=X)
        )
        baseline_metrics = {f"baseline_{name}": _to_float(value) for name, value in values.items()}
        logger.info("Baseline '{}' | {}", baseline, baseline_metrics)
        return baseline_metrics

    def threshold_analysis(
        self,
        X: pd.DataFrame,  # noqa: ARG002 - présent pour la symétrie d'API
        y: pd.Series | Sequence[Any],  # noqa: ARG002
        thresholds: Sequence[float] | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Not applicable to a regression: there is no decision threshold to arbitrate.

        The method exists so that shared code (pipelines, notebooks) can call it unconditionally;
        it always returns an empty frame and logs the reason.

        Args:
            X: Features (unused).
            y: Observed values (unused).
            thresholds: Candidate thresholds (unused).

        Returns:
            An empty ``DataFrame``.
        """
        logger.info(
            "Aucun arbitrage de seuil en régression : la sortie est une valeur continue. "
            "L'équivalent métier est la fourchette de tolérance (± {:.1f} %).",
            self.tolerance_pct,
        )
        return pd.DataFrame()

    def feature_importance(
        self, X: pd.DataFrame, *, y: pd.Series | Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Extract a feature importance ranking when the model exposes one.

        Two mechanisms are supported: a native ``feature_importances_`` / ``coef_`` attribute,
        and permutation importance as a model-agnostic fallback (scored with R2).

        Args:
            X: Features used for permutation importance (small frames only).
            y: Observed values, required by permutation importance.

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
            logger.warning("Permutation importance requires the observed values; skipped")
            return pd.DataFrame()
        sample_X, sample_y = X, np.asarray(y, dtype="float64")
        if len(sample_X) > 1500:
            sample_X = sample_X.sample(n=1500, random_state=0)
            sample_y = np.asarray(y, dtype="float64")[sample_X.index.to_numpy()]
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
    def _predictions_frame(
        self, truth: pd.Series, predictions: np.ndarray, context: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Assemble the row-level diagnostic frame (the base of every analysis).

        Args:
            truth: Observed values.
            predictions: Model predictions.
            context: Optional enriched rows (identifiers, raw segments).

        Returns:
            A frame with ``y_true``, ``y_pred``, error columns and the context columns.
        """
        frame = pd.DataFrame({"y_true": truth.to_numpy(dtype="float64"), "y_pred": predictions})
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["absolute_error"] = frame["residual"].abs()
        denominator = frame["y_true"].where(frame["y_true"].abs() > 1e-9, other=np.nan)
        frame["relative_error_pct"] = 100.0 * frame["residual"] / denominator
        frame["within_tolerance"] = (
            frame["relative_error_pct"].abs() <= self.tolerance_pct
        ).astype("int64")
        # Contrat partagé avec les figures et le rapport : « erreur » = hors fourchette métier.
        frame["is_error"] = 1 - frame["within_tolerance"]
        frame["price_bucket"] = _bucket_labels(frame["y_true"])
        frame = frame.reset_index(drop=True)

        if context is not None and len(context) == len(frame):
            keep = [column for column in context.columns if column in self.context_columns]
            keep += [
                column
                for column in _segment_candidates(context)
                if column not in frame.columns and column not in keep
            ]
            if keep:
                frame = pd.concat(
                    [context.loc[:, keep].reset_index(drop=True), frame], axis=1
                )
        return frame

    def _compute_metrics(self, truth: pd.Series, predictions: np.ndarray) -> dict[str, float]:
        """Compute the configured metrics through the shared registry."""
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(y_true=truth.to_numpy(dtype="float64"), y_pred=predictions)
        )
        return {name: _to_float(value) for name, value in values.items()}

    def _segmentation_axes(
        self, frame: pd.DataFrame, context: pd.DataFrame | None
    ) -> list[str]:
        """Return the columns usable as segmentation axes.

        Args:
            frame: Row-level diagnostic frame.
            context: Optional enriched rows.

        Returns:
            Column names present in ``frame`` (always includes ``price_bucket``).
        """
        axes = [column for column in _segment_candidates(context) if column in frame.columns]
        if "price_bucket" in frame.columns:
            axes.append("price_bucket")
        return list(dict.fromkeys(axes))[:5]

    def _segment_analysis(self, frame: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
        """Recompute the error statistics for every segment of every axis.

        Args:
            frame: Row-level diagnostic frame.
            axes: Segmentation columns.

        Returns:
            A DataFrame ``axis | segment | n | rmse | mae | bias | bias_pct | coverage``.
        """
        rows: list[dict[str, Any]] = []
        for axis in axes:
            if axis not in frame.columns:
                continue
            grouped = frame.groupby(frame[axis].astype(str), observed=True)
            for segment, block in grouped:
                residual = block["residual"].to_numpy(dtype="float64")
                absolute = block["absolute_error"].to_numpy(dtype="float64")
                relative = block["relative_error_pct"].to_numpy(dtype="float64")
                rows.append(
                    {
                        "axis": axis,
                        "segment": str(segment),
                        "n": int(len(block)),
                        "rmse": float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan"),
                        "mae": float(np.mean(absolute)) if absolute.size else float("nan"),
                        "median_absolute_error": (
                            float(np.median(absolute)) if absolute.size else float("nan")
                        ),
                        "bias": float(np.mean(residual)) if residual.size else float("nan"),
                        "bias_pct": float(np.nanmean(relative)) if relative.size else float("nan"),
                        "coverage": (
                            float(block["within_tolerance"].mean())
                            if "within_tolerance" in block
                            else float("nan")
                        ),
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["axis", "n"], ascending=[True, False]).reset_index(
            drop=True
        )

    def _error_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rank the rows by absolute relative error (the most damaging first).

        Args:
            frame: Row-level diagnostic frame.

        Returns:
            The ``top_k_errors`` worst rows, with their original row index.
        """
        if frame.empty or "relative_error_pct" not in frame.columns:
            return pd.DataFrame()
        ranked = frame.copy()
        ranked["row_index"] = ranked.index
        ranked = ranked.assign(_severity=ranked["relative_error_pct"].abs())
        ranked = ranked.sort_values("_severity", ascending=False).drop(columns=["_severity"])
        return ranked.head(self.top_k_errors).reset_index(drop=True)

    def _error_statistics(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Build the JSON-serialisable error statistics consumed by the figures.

        Args:
            frame: Row-level diagnostic frame.

        Returns:
            Mapping with residual arrays, bias, quantiles and per-bucket tables.
        """
        residual = frame["residual"].to_numpy(dtype="float64") if "residual" in frame else np.array([])
        absolute = (
            frame["absolute_error"].to_numpy(dtype="float64")
            if "absolute_error" in frame
            else np.array([])
        )
        relative = (
            frame["relative_error_pct"].to_numpy(dtype="float64")
            if "relative_error_pct" in frame
            else np.array([])
        )
        stats: dict[str, Any] = {
            "residuals": residual,
            "absolute_error": absolute,
            "relative_error_pct": relative,
            "bias": float(np.mean(residual)) if residual.size else float("nan"),
            "bias_pct": float(np.nanmean(relative)) if relative.size else float("nan"),
            "median_absolute_error": float(np.median(absolute)) if absolute.size else float("nan"),
            "absolute_error_p95": (
                float(np.percentile(absolute, 95)) if absolute.size else float("nan")
            ),
            "residual_std": float(np.std(residual)) if residual.size else float("nan"),
            "coverage": (
                float(frame["within_tolerance"].mean()) if "within_tolerance" in frame else float("nan")
            ),
        }
        if "price_bucket" in frame.columns:
            grouped = frame.groupby(frame["price_bucket"].astype(str), observed=True)
            stats["error_by_bucket"] = pd.DataFrame(
                {
                    "n": grouped.size(),
                    "bias": grouped["residual"].mean(),
                    "median_relative_error_pct": grouped["relative_error_pct"].median(),
                    "rmse": grouped["residual"].apply(
                        lambda values: float(np.sqrt(np.mean(np.asarray(values, dtype="float64") ** 2)))
                    ),
                    "coverage": grouped["within_tolerance"].mean(),
                }
            ).reset_index(names="bucket")
            stats["coverage_by_bucket"] = stats["error_by_bucket"].loc[
                :, ["bucket", "n", "coverage"]
            ].copy()
        return stats

    def _distribution_points(
        self, truth: pd.Series, predictions: np.ndarray, frame: pd.DataFrame
    ) -> dict[str, Any]:
        """Down-sample the distributions so that they can be serialised in JSON.

        Args:
            truth: Observed values.
            predictions: Model predictions.
            frame: Row-level diagnostic frame.

        Returns:
            Mapping with the sampled points (max 500 per series).
        """
        observed = truth.to_numpy(dtype="float64")
        limit = 500
        step = max(int(np.ceil(len(observed) / limit)), 1)
        points: dict[str, Any] = {
            "distribution": {
                "y_true": observed[::step].tolist(),
                "y_pred": predictions[::step].tolist(),
            }
        }
        if "price_bucket" in frame.columns:
            summary = (
                frame.groupby(frame["price_bucket"].astype(str), observed=True)
                .agg(
                    n=("y_true", "size"),
                    bias=("residual", "mean"),
                    coverage=("within_tolerance", "mean"),
                )
                .reset_index(names="bucket")
            )
            points["by_bucket"] = summary.to_dict(orient="records")
        return points


class _SklearnAdapter:
    """Expose a :class:`BaseModel` through the minimal scikit-learn scoring protocol."""

    def __init__(self, model: BaseModel) -> None:
        """Store the wrapped model.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
        """
        self.model = model

    def score(self, X: pd.DataFrame, y: Any = None) -> float:
        """Return the coefficient of determination (R2), the default regressor score.

        Args:
            X: Features.
            y: Observed values.

        Returns:
            The R2 score (0.0 when ``y`` is missing).
        """
        if y is None:
            return 0.0
        from sklearn.metrics import r2_score

        predictions = np.asarray(self.model.predict(X), dtype="float64").ravel()
        return float(r2_score(np.asarray(y, dtype="float64"), predictions))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate to the wrapped model.

        Args:
            X: Features.

        Returns:
            The predictions.
        """
        return np.asarray(self.model.predict(X), dtype="float64").ravel()


#: Fallback context columns, used when the configuration declares none.
_CONTEXT_COLUMNS: tuple[str, ...] = ("id", "date", "group")


def _segment_candidates(context: pd.DataFrame | None) -> list[str]:
    """Return the categorical-like columns of ``context`` usable as segmentation axes.

    Args:
        context: Enriched rows aligned with the predictions.

    Returns:
        Column names with a small cardinality (<= :data:`MAX_SEGMENT_CARDINALITY`).
    """
    if context is None:
        return []
    candidates: list[str] = []
    for column in context.columns:
        series = context[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
            unique = int(series.nunique(dropna=True))
            if 1 < unique <= MAX_SEGMENT_CARDINALITY:
                candidates.append(str(column))
    return candidates[:4]


def _bucket_labels(values: pd.Series, n_buckets: int = N_BUCKETS) -> pd.Series:
    """Assign each observed value to a quantile bucket (readable label).

    Args:
        values: Observed target values.
        n_buckets: Number of buckets.

    Returns:
        A categorical series of bucket labels (``Q1`` .. ``Qn``).
    """
    try:
        ranked = pd.qcut(values.rank(method="first"), q=min(n_buckets, max(len(values), 1)), labels=False)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index, name="price_bucket")
    labels = pd.Series(ranked, index=values.index).map(
        lambda index: "all" if pd.isna(index) else f"Q{int(index) + 1}"
    )
    return labels.astype(str).rename("price_bucket")


def _to_float(value: Any) -> float:
    """Coerce a metric value to ``float`` (NaN when impossible)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


__all__ = ["EvaluationResult", "Evaluator", "TOLERANCE_PCT"]
