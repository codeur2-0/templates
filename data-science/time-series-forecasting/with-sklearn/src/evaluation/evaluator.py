"""Model evaluation and error analysis for a **forecasting** target.

A forecast is not evaluated like a regression on independent rows. Three properties of temporal
data change what "good" means, and this module is built around them:

1. **A score without a reference is meaningless.** The evaluator therefore always scores the
   trivial references an operator could produce by hand — persistence (last observed value),
   seasonal naive (same weekday one week earlier) and monthly climatology (28-day rolling
   mean) —
   and reports the *relative improvement*. A model that does not beat the seasonal naive has no
   operational value, whatever its R² says.
2. **One global number hides the structure.** Every metric is broken down **by horizon** (the J+7
   error is roughly twice the J+1 error and drives a different business decision), by month, by
   weekday and by regime (cold snap, heatwave, industrial shutdown).
3. **Rows are not independent.** Residual autocorrelation is measured explicitly, prediction
   intervals are built from *validation* residuals per horizon (never from the evaluated split,
   which would be in-sample), and temporal stability is assessed with a rolling-origin backtest
   instead of a random cross-validation.

The evaluator returns a structured :class:`EvaluationResult` rather than printing, so that the same
computation feeds the Markdown report and figures (``src/evaluation/reports.py``), the metrics JSON
consumed by CI or a model registry, and the notebooks.

Metric conventions
    ``mape`` and ``smape`` come from the shared registry in **percent** (4.0 means 4 %).
    ``mase``
    is scaled by the seasonal-naive error *on the evaluated rows*, so ``mase < 1`` reads
    directly as
    "better than copying last week".
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import read_table, write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Business tolerance band: a forecast is « publishable as is » inside ± this percentage of
# truth.
TOLERANCE_PCT = 5.0

#: Nominal level of the published prediction interval.
INTERVAL_LEVEL = 0.90

#: Maximum number of distinct values for a column to be usable as a segmentation axis.
MAX_SEGMENT_CARDINALITY = 12

#: Lag orders used for the residual autocorrelation diagnostic (daily series, weekly season).
RESIDUAL_AC_LAGS: tuple[int, ...] = (1, 2, 7, 14)

#: Columns that may serve as a trivial reference, in decreasing order of business relevance.
# Each
#: must be computable at the origin — that is what makes it a legitimate reference and a
# feature.
NAIVE_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "load_seasonal_naive",
        "naif saisonnier (même jour de la semaine, une semaine avant la cible)",
    ),
    ("load_last_observed", "persistance (dernière valeur connue à l'origine)"),
    ("load_rolling_mean_28d", "climatologie mensuelle (moyenne glissante 28 jours)"),
    ("load_rolling_mean_7d", "moyenne glissante 7 jours"),
)

#: Fallback horizon column when the resolved configuration does not declare one.
FALLBACK_HORIZON_COLUMN = "horizon_days"

#: Default dispersion scale of a normalised conformal interval. On a series whose noise is
#: multiplicative (the case for an electricity load), the residual dispersion is proportional
# to the
#: level, so the last observed load is a scale that needs no extra model — and it is known at
# the
#: origin, which keeps the interval legal for a real publication.
DEFAULT_INTERVAL_SCALE = "load_last_observed"

#: Interval construction methods, in increasing order of robustness.
INTERVAL_METHODS: tuple[str, ...] = ("residual_quantile", "conformal", "normalized_conformal")


@dataclass(slots=True)
class ForecastSettings:
    """Forecasting knobs resolved from the project configuration — never hard-coded in the code.

    Attributes:
        horizon_column: Column holding the forecast horizon, in days.
        horizons: Horizons the project publishes (used to order the tables and warn on gaps).
        long_horizon: Horizon considered « long » in reports and drift alerts.
        interval_level: Nominal coverage of the published prediction interval.
        interval_method: ``normalized_conformal`` (scores divided by a scale column,
        recommended),
            ``conformal`` (absolute residual scores) or ``residual_quantile`` (signed two-sided
            quantiles, the simplest and the narrowest).
        interval_scale_column: Column carrying the dispersion scale for
        ``normalized_conformal``.
            It must be known at the origin, and proportional to the residual dispersion — on a
            multiplicative-error series, the last observed load is exactly that.
        backtest_folds: Number of rolling-origin folds used for the stability diagnostic.
        review_mape_threshold: Per-horizon MAPE (percent) above which a forecast is flagged for
            human review instead of being committed as is.
        tolerance_pct: Business tolerance band, in percent of the observed value.
        time_column: Column used to order rows chronologically (backtest, autocorrelation).
        event_column: Column describing the regime of the target day (error breakdown).
    """

    horizon_column: str = FALLBACK_HORIZON_COLUMN
    horizons: tuple[int, ...] = ()
    long_horizon: int = 7
    interval_level: float = INTERVAL_LEVEL
    interval_method: str = "normalized_conformal"
    interval_scale_column: str = DEFAULT_INTERVAL_SCALE
    backtest_folds: int = 5
    review_mape_threshold: float = 6.0
    tolerance_pct: float = TOLERANCE_PCT
    time_column: str | None = None
    event_column: str | None = None

    @classmethod
    def resolve(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        paths: ProjectPaths | None = None,
        context_columns: Sequence[str] = (),
    ) -> ForecastSettings:
        """Resolve the settings from the configuration, then from the persisted snapshot.

        Order of precedence: the mapping given by the caller (a notebook or the evaluation
        pipeline), then ``artifacts/models/resolved_config.json`` written by the
        training pipeline,
        then the documented defaults. Reading the snapshot matters because
        ``mode=evaluate`` may run
        long after training, in another process, with no Hydra state available.

        Args:
            config: Root configuration mapping, when the caller has one.
            paths: Project layout, used to locate the resolved-configuration artefact.
            context_columns: Columns available in the evaluation context (used to pick the time
                and event columns when the configuration does not name them).

        Returns:
            The resolved settings.
        """
        layout = paths or ProjectPaths.from_root()
        merged: dict[str, Any] = {}
        snapshot = Path(layout.models_dir) / "resolved_config.json"
        if snapshot.exists():
            try:
                merged.update(json.loads(snapshot.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Resolved configuration unreadable ({}); using defaults", exc)
        if config:
            merged.update(dict(config))

        family = merged.get("load_forecasting") or {}
        data_node = merged.get("data") or {}
        columns = [str(column) for column in context_columns]

        time_column = data_node.get("time_column")
        if not time_column:
            time_column = next(
                (name for name in ("origin_date", "target_date") if name in columns), None
            )
        event_column = next(
            (name for name in ("event_type", "is_extreme_event") if name in columns), None
        )
        horizons = tuple(int(value) for value in (family.get("horizons") or ()) if value)

        return cls(
            horizon_column=str(family.get("horizon_column") or FALLBACK_HORIZON_COLUMN),
            horizons=horizons,
            long_horizon=int(family.get("long_horizon") or (max(horizons) if horizons else 7)),
            interval_level=float(family.get("interval_level") or INTERVAL_LEVEL),
            interval_method=str(family.get("interval_method") or "normalized_conformal"),
            interval_scale_column=str(
                family.get("interval_scale_column") or DEFAULT_INTERVAL_SCALE
            ),
            backtest_folds=int(family.get("backtest_folds") or 5),
            review_mape_threshold=float(family.get("review_mape_threshold") or 6.0),
            tolerance_pct=float(family.get("tolerance_pct") or TOLERANCE_PCT),
            time_column=str(time_column) if time_column else None,
            event_column=str(event_column) if event_column else None,
        )


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of a forecasting evaluation.

    Attributes:
        task: Learning task (``forecasting``).
        split: Split that was scored (``test``, ``val``, ...).
        primary_metric: Name of the metric driving the decision (typically ``mape``).
        metrics: Metric name -> value (percent for ``mape`` / ``smape``).
        predictions: Row-level frame (``y_true``, ``y_pred``, ``residual``, ``absolute_error``,
            ``relative_error_pct``, ``ape_pct``, ``within_tolerance``, horizon and
            context columns,
            plus interval bounds when they could be built).
        labels: Segmentation axes used for :attr:`per_segment`.
        per_segment: Error statistics per segment value.
        per_horizon: One row per horizon: MAPE, sMAPE, MAE, RMSE, bias, coverage, naive
        reference.
        baselines: One row per trivial reference: MAPE, MAE, MASE and improvement versus
        the model.
        backtest: One row per rolling-origin fold: MAPE, MAE, bias, spread.
        intervals: One row per horizon: quantile bounds, width and realised coverage.
        errors: Worst rows, sorted by descending absolute percentage error.
        curves: Down-sampled points of the predicted / observed distributions (JSON friendly).
        feature_importance: Feature -> importance (when the model exposes one).
        extras: Free-form payload (residual diagnostics, bias per month, model summary).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    per_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
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
        """Share of forecasts **outside** the business tolerance band.

        A forecast has no « misclassification rate »: the closest operational reading is
        the share
        of published values the business would refuse to commit without a human review.
        """
        if self.predictions.empty or "within_tolerance" not in self.predictions.columns:
            return float("nan")
        return float(1.0 - self.predictions["within_tolerance"].mean())

    @property
    def coverage(self) -> float:
        """Share of forecasts inside the business tolerance band (``1 - error_rate``)."""
        if self.predictions.empty or "within_tolerance" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["within_tolerance"].mean())

    @property
    def bias(self) -> float:
        """Mean signed **relative** bias (positive = systematic over-forecast), in percent."""
        if self.predictions.empty or "relative_error_pct" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["relative_error_pct"].mean())

    @property
    def interval_coverage(self) -> float:
        """Realised coverage of the published prediction interval (weighted over horizons)."""
        if self.intervals.empty or "coverage" not in self.intervals.columns:
            return float("nan")
        weights = self.intervals["rows"].to_numpy(dtype="float64")
        values = self.intervals["coverage"].to_numpy(dtype="float64")
        total = float(weights.sum())
        return float(np.sum(values * weights) / total) if total > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result (JSON friendly, heavy frames reduced to summaries)."""
        return {
            "task": self.task,
            "split": self.split,
            "n_samples": self.n_samples,
            "primary_metric": self.primary_metric,
            "primary_value": _to_float(self.primary_value),
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "segmentation_axes": self.labels,
            "error_rate": _to_float(self.error_rate),
            "coverage": _to_float(self.coverage),
            "bias_pct": _to_float(self.bias),
            "interval_coverage": _to_float(self.interval_coverage),
            "n_errors": len(self.errors),
            "per_horizon": (
                self.per_horizon.to_dict(orient="records") if not self.per_horizon.empty else []
            ),
            "baselines": self.baselines.to_dict(orient="records")
            if not self.baselines.empty
            else [],
            "backtest": self.backtest.to_dict(orient="records") if not self.backtest.empty else [],
            "intervals": self.intervals.to_dict(orient="records")
            if not self.intervals.empty
            else [],
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
    """Score a fitted forecaster, compare it to trivial references, diagnose residuals."""

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = 25,
        tolerance_pct: float | None = None,
        context_columns: Sequence[str] | None = None,
        config: Mapping[str, Any] | None = None,
        settings: ForecastSettings | None = None,
    ) -> None:
        """Inject the model, the metric policy and the forecasting settings.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics and to read the persisted splits.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of worst errors kept for the error analysis.
            tolerance_pct: Business tolerance band in percent; defaults to the configured value.
            context_columns: Columns copied from the enriched context into the diagnostic frame
                (identifier, time, horizon, regime). Defaults to the resolved configuration.
            config: Root configuration mapping, used to resolve :class:`ForecastSettings`.
            settings: Pre-resolved settings (takes precedence over ``config``).
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self.settings = settings or ForecastSettings.resolve(config, paths=self.paths)
        if tolerance_pct is not None:
            self.settings.tolerance_pct = float(tolerance_pct)
        self.tolerance_pct = float(self.settings.tolerance_pct)
        self.context_columns: tuple[str, ...] = tuple(
            str(column) for column in (context_columns or _default_context_columns(self.settings))
        )
        self.primary_metric = str(self.config.get("primary", "mape"))
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
        settings = ForecastSettings.resolve(config, paths=paths)
        # Les colonnes de contexte utiles à l'analyse d'erreurs sont déclarées dans la
        # configuration (identifiant, groupe, date) : aucun nom de colonne n'est codé en dur.
        context_columns = [
            str(data_node[key])
            for key in ("id_column", "group_column", "time_column")
            if data_node.get(key)
        ]
        for extra_column in (settings.horizon_column, settings.event_column):
            if extra_column and extra_column not in context_columns:
                context_columns.append(str(extra_column))
        return cls(
            model,
            metrics_config=dict(metrics_node),
            paths=paths,
            task=metrics_node.get("task"),
            context_columns=tuple(context_columns) or None,
            config=config,
            settings=settings,
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
        """Score the model on a feature matrix and produce every forecasting diagnostic.

        Args:
            X: Preprocessed features.
            y: Observed target values (required: a forecast is meaningless without truth).
            split: Split name, used in logs and artefacts.
            context: Optional raw/enriched rows aligned with ``X`` (identifiers,
            horizon, regime,
                naive references), which is what makes the per-horizon and per-regime analyses
                possible at all.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            ValueError: When the ground truth is missing.
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        if y is None:
            msg = "Forecast evaluation requires the observed target values"
            raise ValueError(msg)

        truth = pd.Series(np.asarray(y, dtype="float64")).reset_index(drop=True)
        predicted = np.asarray(self.model.predict(X), dtype="float64").ravel()

        frame = self._predictions_frame(truth, predicted, context)
        naive = self._resolve_naive(frame)
        metrics = self._compute_metrics(truth, predicted, frame, naive)

        # Les intervalles sont construits AVANT la ventilation par horizon : ils enrichissent la
        # frame (`lower`, `upper`), que la table par horizon lit ensuite pour mesurer
        # la couverture.
        intervals = self.prediction_intervals(frame, X=X)
        axes = self._segmentation_axes(frame, context)
        per_segment = self._segment_analysis(frame, axes)
        per_horizon = self.per_horizon(frame, naive_column=naive)
        baselines = self.compare_to_baselines(frame)
        backtest = self.backtest(frame, X=X, y=truth)
        errors = self._error_analysis(frame)
        extras = self._diagnostics(frame, intervals=intervals, backtest=backtest)
        curves = self._distribution_points(truth, predicted, frame)
        importance = self.feature_importance(X, y=truth)

        if not intervals.empty and {"lower", "upper"} <= set(frame.columns):
            metrics["interval_coverage"] = _to_float(self._weighted_coverage(intervals))

        extras.update(
            {
                "model": self.model.summary(),
                "n_features": int(X.shape[1]),
                "segmentation_axes": list(axes),
                "tolerance_pct": self.tolerance_pct,
                "naive_reference": naive,
                "horizon_column": self.settings.horizon_column,
                "interval_level": self.settings.interval_level,
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
            per_horizon=per_horizon,
            baselines=baselines,
            backtest=backtest,
            intervals=intervals,
            errors=errors,
            curves=curves,
            feature_importance=importance,
            extras=extras,
            n_samples=len(X),
        )
        # Deux couvertures différentes, deux libellés différents : `bande_±5%` est la part des
        # prévisions tombant dans la tolérance métier, `couverture_intervalle` est la part des
        # réalisations tombant dans l'intervalle publié. Les confondre dans un seul mot
        # « couverture » fait lire 73 % là où l'intervalle en couvre 85 %.
        logger.info(
            "Evaluation on '{}' | n={} {}={:.3f}% naive={:.3f}% gain={:+.1f}% "
            "bande_±5%={:.1%} couverture_intervalle={:.1%}",
            split,
            result.n_samples,
            self.primary_metric,
            result.primary_value,
            _to_float(extras.get("naive_mape", float("nan"))),
            _to_float(extras.get("improvement_vs_naive_pct", float("nan"))),
            result.coverage,
            result.interval_coverage,
        )
        return result

    # ------------------------------------------------------------------ per horizon -----
    def per_horizon(self, frame: pd.DataFrame, *, naive_column: str | None = None) -> pd.DataFrame:
        """Break the error down by forecast horizon.

        This is the single most informative table of a forecasting project: the J+1 error drives
        the daily balancing decision, the J+7 error drives the weekly purchasing
        programme, and a
        global MAPE averages two different products into one unreadable number.

        Args:
            frame: Row-level diagnostic frame produced by :meth:`_predictions_frame`.
            naive_column: Column holding the seasonal-naive reference, when available.

        Returns:
            A DataFrame indexed by horizon with MAPE, sMAPE, MAE, RMSE, bias,
            tolerance coverage,
            naive MAPE and relative improvement.
        """
        column = self.settings.horizon_column
        if column not in frame.columns:
            logger.warning("Horizon column '{}' absent: per-horizon breakdown skipped", column)
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        # La table expose une colonne stable nommée `horizon` : les figures et le
        # rapport n'ont pas
        # à connaître le nom déclaré dans la configuration (disponible dans `extras`).
        for horizon, block in frame.groupby(column, observed=True):
            truth = block["y_true"].to_numpy(dtype="float64")
            predicted = block["y_pred"].to_numpy(dtype="float64")
            entry: dict[str, Any] = {
                "horizon": horizon,
                "rows": len(block),
                "mape_pct": _mape(truth, predicted),
                "smape_pct": _smape(truth, predicted),
                "mae_mw": float(np.mean(np.abs(predicted - truth))),
                "rmse_mw": float(np.sqrt(np.mean((predicted - truth) ** 2))),
                "bias_pct": float(
                    np.nanmean(block["relative_error_pct"].to_numpy(dtype="float64"))
                ),
                "within_tolerance_pct": 100.0 * float(block["within_tolerance"].mean()),
                "max_ape_pct": float(np.nanmax(block["ape_pct"].to_numpy(dtype="float64"))),
            }
            if naive_column and naive_column in block.columns:
                reference = block[naive_column].to_numpy(dtype="float64")
                entry["naive_mape_pct"] = _mape(truth, reference)
                entry["improvement_vs_naive_pct"] = _improvement(
                    entry["naive_mape_pct"], entry["mape_pct"]
                )
            if "lower" in block.columns and "upper" in block.columns:
                entry["interval_coverage_pct"] = 100.0 * float(
                    np.mean(
                        (truth >= block["lower"].to_numpy(dtype="float64"))
                        & (truth <= block["upper"].to_numpy(dtype="float64"))
                    )
                )
            rows.append(entry)
        table = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        return table.round(3)

    # ------------------------------------------------------------------ baselines -------
    def compare_to_baselines(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score the trivial references an operator could build without any model.

        Every row is a forecast that uses only information available at the origin, so each is a
        legitimate competitor. The seasonal naive is the one that matters: it is what
        the business
        does today by hand.

        Args:
            frame: Row-level diagnostic frame (must carry the reference columns).

        Returns:
            A DataFrame with one row per reference: MAPE, MAE, RMSE and the model's relative
            improvement over it.
        """
        truth = frame["y_true"].to_numpy(dtype="float64")
        predicted = frame["y_pred"].to_numpy(dtype="float64")
        model_mape = _mape(truth, predicted)
        model_mae = float(np.mean(np.abs(predicted - truth)))
        rows: list[dict[str, Any]] = [
            {
                "reference": "modèle appris",
                "description": "prévision produite par le modèle évalué",
                "mape_pct": model_mape,
                "mae_mw": model_mae,
                "rmse_mw": float(np.sqrt(np.mean((predicted - truth) ** 2))),
                "improvement_pct": 0.0,
            }
        ]
        for column, description in NAIVE_CANDIDATES:
            if column not in frame.columns:
                continue
            reference = frame[column].to_numpy(dtype="float64")
            if np.isnan(reference).all():
                continue
            reference_mape = _mape(truth, reference)
            rows.append(
                {
                    "reference": column,
                    "description": description,
                    "mape_pct": reference_mape,
                    "mae_mw": float(np.nanmean(np.abs(reference - truth))),
                    "rmse_mw": float(np.sqrt(np.nanmean((reference - truth) ** 2))),
                    "improvement_pct": _improvement(reference_mape, model_mape),
                }
            )
        return pd.DataFrame(rows).round(3)

    # ------------------------------------------------------------------ backtest --------
    def backtest(
        self,
        frame: pd.DataFrame,
        *,
        X: pd.DataFrame | None = None,
        y: pd.Series | Sequence[Any] | None = None,
        model_factory: Any = None,
        refit: bool = False,
    ) -> pd.DataFrame:
        """Rolling-origin backtest: the temporal equivalent of cross-validation.

        The evaluated rows are cut into consecutive chronological folds. By default the *already
        fitted* model is scored on each fold, which answers the operational question « does the
        error drift over time? » at no training cost. With ``refit=True`` and a
        ``model_factory``,
        each fold retrains on everything preceding it — the honest rolling-origin
        protocol, used by
        the notebooks on a reduced sample because it multiplies the training cost by the
        fold count.

        Args:
            frame: Row-level diagnostic frame, ordered or orderable by the time column.
            X: Preprocessed features aligned with ``frame`` (required when ``refit``).
            y: Observed values aligned with ``frame`` (required when ``refit``).
            model_factory: Callable returning an unfitted model, used when ``refit``.
            refit: Whether to retrain on the expanding window of each fold.

        Returns:
            A DataFrame with one row per fold: period, MAPE, MAE, bias, and — for
            the refit mode —
            the training window size. A final ``spread`` row summarises the dispersion.
        """
        ordered = self._chronological(frame)
        folds = max(int(self.settings.backtest_folds), 2)
        if len(ordered) < folds * 8:
            logger.warning(
                "Backtest skipped: {} rows are not enough for {} chronological folds",
                len(ordered),
                folds,
            )
            return pd.DataFrame()

        boundaries = np.linspace(0, len(ordered), folds + 1, dtype=int)
        time_column = self.settings.time_column
        rows: list[dict[str, Any]] = []
        for index in range(folds):
            start, stop = int(boundaries[index]), int(boundaries[index + 1])
            if stop <= start:
                continue
            block = ordered.iloc[start:stop]
            truth = block["y_true"].to_numpy(dtype="float64")
            predicted = block["y_pred"].to_numpy(dtype="float64")
            if refit and X is not None and y is not None and model_factory is not None:
                predicted = self._refit_fold(X, y, ordered, start, stop, model_factory)
            entry: dict[str, Any] = {
                "fold": index + 1,
                "rows": len(block),
                "mape_pct": _mape(truth, predicted),
                "mae_mw": float(np.mean(np.abs(predicted - truth))),
                "bias_pct": float(
                    np.nanmean(block["relative_error_pct"].to_numpy(dtype="float64"))
                ),
                "mode": "refit" if refit else "fixed_model",
            }
            if time_column and time_column in block.columns:
                stamps = pd.to_datetime(block[time_column])
                entry["period_start"] = str(stamps.min().date())
                entry["period_end"] = str(stamps.max().date())
            rows.append(entry)

        table = pd.DataFrame(rows)
        if table.empty:
            return table
        summary: dict[str, Any] = {
            "fold": "dispersion",
            "rows": int(table["rows"].sum()),
            "mape_pct": float(np.nanstd(table["mape_pct"].to_numpy(dtype="float64"))),
            "mae_mw": float(np.nanstd(table["mae_mw"].to_numpy(dtype="float64"))),
            "bias_pct": float(np.nanmean(table["bias_pct"].to_numpy(dtype="float64"))),
            "mode": str(table["mode"].iloc[0]),
        }
        for key, position in (("period_start", 0), ("period_end", -1)):
            if key in table.columns:
                summary[key] = table[key].iloc[position]
        return pd.concat([table, pd.DataFrame([summary])], ignore_index=True).round(3)

    def _refit_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any],
        ordered: pd.DataFrame,
        start: int,
        stop: int,
        model_factory: Any,
    ) -> np.ndarray:
        """Retrain on everything preceding a fold and predict that fold.

        Args:
            X: Preprocessed features (same order as the unsorted input frame).
            y: Observed values.
            ordered: Chronologically ordered diagnostic frame (index maps back to ``X``).
            start: First row of the fold, in chronological order.
            stop: Row after the last one of the fold.
            model_factory: Callable returning an unfitted model exposing ``fit`` / ``predict``.

        Returns:
            The fold predictions.
        """
        positions = ordered.index.to_numpy()
        fold_positions = positions[start:stop]
        train_positions = positions[:start]
        if len(train_positions) < 20:
            logger.warning(
                "Fold {} has fewer than 20 training rows; reusing the fitted model", stop
            )
            return ordered.iloc[start:stop]["y_pred"].to_numpy(dtype="float64")
        try:
            candidate = model_factory()
            fitted = candidate.fit(X.iloc[train_positions], np.asarray(y)[train_positions])
            estimator = getattr(fitted, "estimator_", None) or fitted
            return np.asarray(estimator.predict(X.iloc[fold_positions]), dtype="float64").ravel()
        except Exception as exc:
            logger.warning("Refit backtest failed on fold ending at {}: {}", stop, exc)
            return ordered.iloc[start:stop]["y_pred"].to_numpy(dtype="float64")

    # ------------------------------------------------------------------ intervals -------
    def prediction_intervals(
        self, frame: pd.DataFrame, *, X: pd.DataFrame | None = None, adaptive: bool = False
    ) -> pd.DataFrame:
        """Build per-horizon prediction intervals from **validation** residuals and measure them.

        Three construction methods are available, all calibrated on the validation split
        persisted
        by the training pipeline and applied to the evaluated rows. Building the bounds on the
        evaluated split itself would be in-sample and would report a flattering coverage:

        * ``residual_quantile`` — the two-sided quantiles of the signed residuals. Simple, and the
          narrowest of the three: it implicitly assumes the residual distribution is
          symmetric and
          stationary between the calibration window and the evaluated period.
        * ``conformal`` — the quantile of the **absolute** residuals at the nominal level, applied
          symmetrically. This is split conformal prediction: under exchangeability it covers at
          least the nominal level, which time series violate, so the realised coverage
          is measured
          rather than assumed.
        * ``normalized_conformal`` — the same scores divided by a dispersion **scale** known at the
          origin (by default the last observed load). On a series whose noise is
          multiplicative, the
          residual dispersion grows with the level, so a single absolute width
          calibrated on a calm
          summer window is far too narrow for a winter peak. Normalising is what
          recovers most of
          the missing coverage, and it costs nothing at inference time.

        ``adaptive=True`` extends the calibration pool with the residual of every
        evaluated row that
        **chronologically precedes** the row being bounded — the daily recalibration a production
        system performs, since yesterday's error is known this morning. It is off by
        default because
        the shipped predictor publishes the static interval; the notebooks turn it on to
        measure the
        gain, and the returned table records which mode produced the numbers.

        Args:
            frame: Row-level diagnostic frame of the evaluated split.
            X: Preprocessed features of the evaluated split (kept for API symmetry with variants
                that need the feature matrix).
            adaptive: Whether to recalibrate on the preceding evaluated residuals.

        Returns:
            A DataFrame with one row per horizon: method, score quantile, absolute
            offsets, interval
            width, realised coverage (overall and on rows outside any extreme regime) and the
            calibration source. The evaluated ``frame`` gains ``lower`` / ``upper`` /
            ``interval_width`` columns.
        """
        del X  # la calibration n'utilise que les résidus, pas la matrice de features
        level = float(self.settings.interval_level)
        alpha = (1.0 - level) / 2.0
        method = str(self.settings.interval_method)
        if method not in INTERVAL_METHODS:
            logger.warning("Méthode d'intervalle inconnue '{}': repli sur 'conformal'", method)
            method = "conformal"
        column = self.settings.horizon_column
        scale_column = (
            self.settings.interval_scale_column if method == "normalized_conformal" else None
        )

        calibration, source = self._validation_residuals(scale_column=scale_column)
        if calibration is None:
            logger.warning(
                "Aucun résidu de validation disponible : les intervalles sont"
                " calibrés sur le split "
                "évalué (intra-échantillon, la couverture sera optimiste)"
            )
            calibration = frame
            source = "evaluated_split"

        predicted = frame["y_pred"].to_numpy(dtype="float64")
        residual = frame["residual"].to_numpy(dtype="float64")
        evaluated_scale = _scale_values(frame, scale_column)
        lower = np.full(len(frame), np.nan)
        upper = np.full(len(frame), np.nan)

        rows: list[dict[str, Any]] = []
        grouped = _group_positions(frame, column)
        for horizon, positions in grouped:
            calibration_block = _select_horizon(calibration, column, horizon)
            calibration_residual = calibration_block["residual"].to_numpy(dtype="float64")
            if len(calibration_residual) == 0:
                continue
            offsets = (float("nan"), float("nan"))
            score_quantile = float("nan")

            if method == "residual_quantile":
                low, high = np.nanquantile(calibration_residual, [alpha, 1.0 - alpha])
                offsets = (float(low), float(high))
                lower[positions] = predicted[positions] + offsets[0]
                upper[positions] = predicted[positions] + offsets[1]
                if adaptive:
                    logger.warning(
                        "Le mode adaptatif n'est pas défini pour 'residual_quantile' : ignoré"
                    )
            else:
                calibration_scale = _scale_values(calibration_block, scale_column)
                scores = np.abs(calibration_residual) / calibration_scale
                score_quantile = float(np.nanquantile(scores, level))
                if adaptive:
                    running = list(scores[np.isfinite(scores)])
                    for position in positions:
                        current = (
                            float(np.nanquantile(np.asarray(running, dtype="float64"), level))
                            if running
                            else score_quantile
                        )
                        width = current * float(evaluated_scale[position])
                        lower[position] = predicted[position] - width
                        upper[position] = predicted[position] + width
                        local_scale = float(evaluated_scale[position])
                        if local_scale > 1e-9:
                            running.append(abs(float(residual[position])) / local_scale)
                else:
                    widths = score_quantile * evaluated_scale[positions]
                    lower[positions] = predicted[positions] - widths
                    upper[positions] = predicted[positions] + widths
                offsets = (-score_quantile, score_quantile)

            truth = frame["y_true"].to_numpy(dtype="float64")[positions]
            covered = (truth >= lower[positions]) & (truth <= upper[positions])
            widths = upper[positions] - lower[positions]
            calm = _calm_mask(frame, positions)
            rows.append(
                {
                    "horizon": horizon,
                    "rows": len(positions),
                    "level_pct": round(100.0 * level, 1),
                    "method": method,
                    "adaptive": bool(adaptive),
                    "scale_column": scale_column or "-",
                    "score_quantile": round(score_quantile, 4),
                    "offset_low_mw": round(offsets[0], 3),
                    "offset_high_mw": round(offsets[1], 3),
                    "mean_width_mw": round(float(np.nanmean(widths)), 1),
                    "relative_width_pct": round(
                        float(
                            np.nanmean(
                                widths / np.where(np.abs(truth) < 1e-9, np.nan, np.abs(truth))
                            )
                        )
                        * 100.0,
                        2,
                    ),
                    "coverage_pct": round(100.0 * float(np.mean(covered)), 1),
                    "coverage": float(np.mean(covered)),
                    "coverage_calm_pct": round(100.0 * float(np.mean(covered[calm])), 1)
                    if calm.any()
                    else None,
                    "coverage_extreme_pct": round(100.0 * float(np.mean(covered[~calm])), 1)
                    if (~calm).any()
                    else None,
                    "n_extreme": int((~calm).sum()),
                    "source": source,
                }
            )

        frame["lower"] = lower
        frame["upper"] = upper
        frame["interval_width"] = upper - lower
        return pd.DataFrame(rows)

    def _validation_residuals(
        self, scale_column: str | None = None
    ) -> tuple[pd.DataFrame | None, str]:
        """Load the validation split persisted by training and score it with the fitted model.

        Args:
            scale_column: Column carrying the dispersion scale, copied when present so that a
                normalised conformal interval can be calibrated.

        Returns:
            ``(residual frame, source label)``; ``(None, "unavailable")`` when the artefacts are
            missing, which happens when evaluation runs without a prior training in
            this workspace.
        """
        features_path = Path(self.paths.processed_dir) / "features_X_val.parquet"
        split_path = Path(self.paths.processed_dir) / "split_val.parquet"
        if not (features_path.exists() and split_path.exists()):
            return None, "unavailable"
        try:
            features = read_table(features_path)
            split = read_table(split_path)
        except (OSError, ValueError) as exc:
            logger.warning("Split de validation illisible ({}): repli des intervalles", exc)
            return None, "unavailable"
        if len(features) != len(split):
            logger.warning(
                "Artefacts de validation désalignés ({} features vs {} lignes):"
                " repli des intervalles",
                len(features),
                len(split),
            )
            return None, "unavailable"
        target = self._target_name(split)
        if target is None:
            return None, "unavailable"
        truth = pd.Series(split[target].to_numpy(dtype="float64"))
        predicted = pd.Series(np.asarray(self.model.predict(features), dtype="float64").ravel())
        residuals = pd.DataFrame({"y_true": truth, "y_pred": predicted})
        residuals["residual"] = residuals["y_pred"] - residuals["y_true"]
        column = self.settings.horizon_column
        if column in split.columns:
            residuals[column] = split[column].to_numpy()
        if scale_column and scale_column in split.columns:
            residuals[scale_column] = pd.to_numeric(split[scale_column], errors="coerce").to_numpy(
                dtype="float64"
            )
        return residuals.reset_index(drop=True), "validation_split"

    def _target_name(self, frame: pd.DataFrame) -> str | None:
        """Return the target column name, resolved from the persisted configuration."""
        snapshot = Path(self.paths.models_dir) / "resolved_config.json"
        if snapshot.exists():
            try:
                payload = json.loads(snapshot.read_text(encoding="utf-8"))
                name = (payload.get("data") or {}).get("target")
                if name and str(name) in frame.columns:
                    return str(name)
            except (OSError, json.JSONDecodeError):
                logger.debug("Resolved configuration unreadable while looking for the target")
        for candidate in ("load_mw", "y_true", "target"):
            if candidate in frame.columns:
                return candidate
        return None

    # ------------------------------------------------------------------ importance ------
    def feature_importance(
        self, X: pd.DataFrame, *, y: pd.Series | Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Extract a feature importance ranking when the model exposes one.

        Two mechanisms are supported: a native ``feature_importances_`` / ``coef_``
        attribute, and
        permutation importance as a model-agnostic fallback. For a forecast the native
        ranking of a
        tree ensemble is the one operators read: it tells whether the model leans on the
        weather, on
        the calendar or on the recent level — and a forecast driven mostly by
        ``load_lag_1d`` is a
        persistence model in disguise.

        Args:
            X: Features used for permutation importance (sub-sampled above 1500 rows).
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
        except Exception as exc:
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
            predictions: Model forecasts.
            context: Optional enriched rows (identifiers, horizon, regime, naive references).

        Returns:
            A frame with ``y_true``, ``y_pred``, error columns and the context columns.
        """
        frame = pd.DataFrame({"y_true": truth.to_numpy(dtype="float64"), "y_pred": predictions})
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["absolute_error"] = frame["residual"].abs()
        denominator = frame["y_true"].where(frame["y_true"].abs() > 1e-9, other=np.nan)
        frame["relative_error_pct"] = 100.0 * frame["residual"] / denominator
        frame["ape_pct"] = frame["relative_error_pct"].abs()
        frame["within_tolerance"] = (frame["ape_pct"] <= self.tolerance_pct).astype("int64")
        # Contrat partagé avec les figures et le rapport : « erreur » = hors tolérance métier.
        frame["is_error"] = 1 - frame["within_tolerance"]
        frame = frame.reset_index(drop=True)

        if context is not None and len(context) == len(frame):
            keep = [column for column in self.context_columns if column in context.columns]
            keep += [
                column
                for column, _ in NAIVE_CANDIDATES
                if column in context.columns and column not in keep
            ]
            keep += [
                column
                for column in _segment_candidates(context)
                if column not in frame.columns and column not in keep
            ]
            if keep:
                frame = pd.concat([context.loc[:, keep].reset_index(drop=True), frame], axis=1)
        return frame

    def _resolve_naive(self, frame: pd.DataFrame) -> str | None:
        """Pick the seasonal-naive reference column used to scale the MASE.

        Args:
            frame: Row-level diagnostic frame.

        Returns:
            The reference column name, or ``None`` when the context did not provide one.
        """
        for column, _ in NAIVE_CANDIDATES:
            if column in frame.columns and frame[column].notna().any():
                return column
        logger.warning(
            "No naive reference column found among {}: MASE falls back to a lagged-truth proxy",
            ", ".join(column for column, _ in NAIVE_CANDIDATES),
        )
        return None

    def _compute_metrics(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        frame: pd.DataFrame,
        naive_column: str | None,
    ) -> dict[str, float]:
        """Compute the configured metrics through the shared registry.

        The naive reference is forwarded through ``MetricInputs.extra`` so that ``mase``
        is scaled
        by a real seasonal-naive forecast rather than by a shift of the evaluated rows.

        Args:
            truth: Observed values.
            predictions: Model forecasts.
            frame: Row-level diagnostic frame, source of the naive reference column.
            naive_column: Reference column present in ``frame``, when any.

        Returns:
            Metric name -> value.
        """
        extra: dict[str, Any] = {}
        if naive_column and naive_column in frame.columns and len(frame) == len(truth):
            extra["naive_pred"] = frame[naive_column].to_numpy(dtype="float64")
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names, extra=extra)
        values = calculator.evaluate(
            MetricInputs(y_true=truth.to_numpy(dtype="float64"), y_pred=predictions, extra=extra)
        )
        return {name: _to_float(value) for name, value in values.items()}

    def _segmentation_axes(self, frame: pd.DataFrame, context: pd.DataFrame | None) -> list[str]:
        """Return the columns usable as segmentation axes.

        The horizon always comes first: it is the axis the business reads the forecast through.

        Args:
            frame: Row-level diagnostic frame.
            context: Optional enriched rows.

        Returns:
            The ordered axis names.
        """
        axes: list[str] = []
        horizon = self.settings.horizon_column
        if horizon in frame.columns:
            axes.append(horizon)
        candidates = _segment_candidates(context if context is not None else frame)
        for column in candidates:
            if column in frame.columns and column not in axes:
                axes.append(column)
        return axes[:5]

    def _segment_analysis(self, frame: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
        """Recompute the errors along each segmentation axis.

        Args:
            frame: Row-level diagnostic frame.
            axes: Columns to segment by.

        Returns:
            A long DataFrame ``axis | segment | rows | mape_pct | mae_mw | bias_pct | share``.
        """
        rows: list[dict[str, Any]] = []
        total = float(len(frame))
        for axis in axes:
            if axis not in frame.columns:
                continue
            values = frame[axis]
            if pd.api.types.is_numeric_dtype(values) and values.nunique() > MAX_SEGMENT_CARDINALITY:
                values = _bucket_labels(values)
            for segment, block in frame.groupby(values, observed=True):
                truth = block["y_true"].to_numpy(dtype="float64")
                predicted = block["y_pred"].to_numpy(dtype="float64")
                rows.append(
                    {
                        "axis": axis,
                        "segment": str(segment),
                        "rows": len(block),
                        "mape_pct": round(_mape(truth, predicted), 3),
                        "mae_mw": round(float(np.mean(np.abs(predicted - truth))), 2),
                        "bias_pct": round(
                            float(
                                np.nanmean(block["relative_error_pct"].to_numpy(dtype="float64"))
                            ),
                            3,
                        ),
                        "share": round(len(block) / total, 4) if total else 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def _error_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rank the worst forecasts by absolute percentage error.

        Args:
            frame: Row-level diagnostic frame.

        Returns:
            The ``top_k_errors`` worst rows, most recent context columns included.
        """
        if frame.empty:
            return frame
        ordered = frame.sort_values("ape_pct", ascending=False).head(self.top_k_errors)
        return ordered.reset_index(drop=True)

    def _diagnostics(
        self,
        frame: pd.DataFrame,
        *,
        intervals: pd.DataFrame,
        backtest: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute the residual diagnostics that only make sense on ordered temporal rows.

        Args:
            frame: Row-level diagnostic frame.
            intervals: Per-horizon interval table.
            backtest: Rolling-origin fold table.

        Returns:
            A JSON-friendly payload: naive and model errors, improvement, bias per
            month, residual
            autocorrelation, interval coverage and backtest spread.
        """
        ordered = self._chronological(frame)
        residual = ordered["residual"].to_numpy(dtype="float64")
        autocorrelation = {
            f"lag_{lag}": round(_autocorrelation(residual, lag), 4)
            for lag in RESIDUAL_AC_LAGS
            if len(residual) > lag + 2
        }

        naive_column = self._resolve_naive(frame)
        naive_mape = float("nan")
        if naive_column:
            naive_mape = _mape(
                ordered["y_true"].to_numpy(dtype="float64"),
                ordered[naive_column].to_numpy(dtype="float64"),
            )
        model_mape = _mape(
            ordered["y_true"].to_numpy(dtype="float64"),
            ordered["y_pred"].to_numpy(dtype="float64"),
        )

        bias_by_month: dict[str, float] = {}
        month_column = next(
            (name for name in ("target_month", "month") if name in ordered.columns), None
        )
        if month_column:
            for month, block in ordered.groupby(month_column, observed=True):
                bias_by_month[str(month)] = round(
                    float(np.nanmean(block["relative_error_pct"].to_numpy(dtype="float64"))), 3
                )

        spread = float("nan")
        if not backtest.empty:
            folds = backtest[backtest["fold"] != "dispersion"]
            if len(folds) > 1:
                spread = float(np.nanstd(folds["mape_pct"].to_numpy(dtype="float64")))

        return {
            "naive_reference": naive_column,
            "naive_mape": round(naive_mape, 3),
            "model_mape": round(model_mape, 3),
            "improvement_vs_naive_pct": round(_improvement(naive_mape, model_mape), 2),
            "bias_pct": round(float(np.nanmean(ordered["relative_error_pct"])), 3),
            "median_ape_pct": round(float(np.nanmedian(ordered["ape_pct"])), 3),
            "p95_ape_pct": round(float(np.nanpercentile(ordered["ape_pct"], 95)), 3),
            "residual_std": round(float(np.nanstd(residual)), 2),
            "residual_autocorrelation": autocorrelation,
            "bias_by_month": bias_by_month,
            "worst_month_bias_pct": round(
                max((abs(value) for value in bias_by_month.values()), default=float("nan")), 3
            ),
            "interval_coverage_pct": round(100.0 * self._weighted_coverage(intervals), 2)
            if not intervals.empty
            else None,
            "interval_source": str(intervals["source"].iloc[0]) if not intervals.empty else None,
            "interval_method": str(
                intervals["method"].iloc[0]
                if not intervals.empty
                else self.settings.interval_method
            ),
            "interval_scale_column": str(
                intervals["scale_column"].iloc[0]
                if not intervals.empty and "scale_column" in intervals.columns
                else self.settings.interval_scale_column
            ),
            "interval_adaptive": bool(
                intervals["adaptive"].iloc[0]
                if not intervals.empty and "adaptive" in intervals.columns
                else False
            ),
            "backtest_spread_pct": round(spread, 3),
            "n_extreme_rows": int(ordered["is_extreme_event"].sum())
            if "is_extreme_event" in ordered.columns
            else None,
        }

    def _weighted_coverage(self, intervals: pd.DataFrame) -> float:
        """Row-weighted coverage of the per-horizon intervals."""
        if intervals.empty or "coverage" not in intervals.columns:
            return float("nan")
        weights = intervals["rows"].to_numpy(dtype="float64")
        values = intervals["coverage"].to_numpy(dtype="float64")
        total = float(weights.sum())
        return float(np.sum(values * weights) / total) if total > 0 else float("nan")

    def _chronological(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame ordered by time then horizon, keeping the original row labels.

        Ordering matters for every temporal diagnostic (residual autocorrelation,
        backtest folds):
        on an unordered frame both are silently meaningless. The original labels are
        preserved on
        purpose — they are the positions of the rows inside the feature matrix ``X``,
        which is what
        lets the refit backtest slice ``X`` with the same indices.

        Args:
            frame: Row-level diagnostic frame (RangeIndex aligned with ``X``).

        Returns:
            The chronologically ordered frame, same index labels, sorted rows.
        """
        if frame.empty:
            return frame
        keys: list[str] = []
        working = frame
        time_column = self.settings.time_column
        if time_column and time_column in frame.columns:
            working = frame.assign(_stamp=pd.to_datetime(frame[time_column]))
            keys.append("_stamp")
        if self.settings.horizon_column in working.columns:
            keys.append(self.settings.horizon_column)
        if not keys:
            return working
        return working.sort_values(keys, kind="mergesort").drop(columns=["_stamp"], errors="ignore")

    def _distribution_points(
        self, truth: pd.Series, predictions: np.ndarray, frame: pd.DataFrame
    ) -> dict[str, Any]:
        """Down-sample the observed / forecast distributions for JSON-friendly figures.

        Args:
            truth: Observed values.
            predictions: Model forecasts.
            frame: Row-level diagnostic frame.

        Returns:
            Quantile points of both distributions plus the error distribution.
        """
        quantiles = np.linspace(0.0, 1.0, 41)
        return {
            "truth_quantiles": np.quantile(truth.to_numpy(dtype="float64"), quantiles)
            .round(2)
            .tolist(),
            "pred_quantiles": np.quantile(predictions, quantiles).round(2).tolist(),
            "error_quantiles": np.quantile(
                frame["relative_error_pct"].dropna().to_numpy(dtype="float64"), quantiles
            )
            .round(3)
            .tolist(),
            "quantile_levels": quantiles.round(4).tolist(),
        }


class _SklearnAdapter:
    """Expose a :class:`BaseModel` through the scikit-learn estimator protocol.

    ``sklearn.inspection.permutation_importance`` requires ``fit`` / ``predict`` /
    ``score``; the
    project's models expose ``fit`` / ``predict`` and a task-specific scoring. This adapter
    is the
    thin bridge, kept private so that no other module depends on it.
    """

    def __init__(self, model: BaseModel) -> None:
        """Store the wrapped model.

        Args:
            model: Fitted project model.
        """
        self.model = model

    def score(self, X: pd.DataFrame, y: Any = None) -> float:
        """Score with the negative mean absolute percentage error (higher is better).

        Args:
            X: Feature matrix.
            y: Observed values.

        Returns:
            The negative MAPE, so that ``permutation_importance`` maximises the right thing.
        """
        predicted = np.asarray(self.predict(X), dtype="float64").ravel()
        truth = np.asarray(y, dtype="float64").ravel()
        return -_mape(truth, predicted)

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> _SklearnAdapter:
        """Fit the wrapped model (permutation importance never calls it, protocol completeness).

        Args:
            X: Feature matrix.
            y: Observed values.
            **fit_params: Forwarded to the wrapped model.

        Returns:
            ``self``.
        """
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the wrapped model.

        Args:
            X: Feature matrix.

        Returns:
            The predictions, flattened.
        """
        return np.asarray(self.model.predict(X), dtype="float64").ravel()


def _scale_values(frame: pd.DataFrame, scale_column: str | None) -> np.ndarray:
    """Return the per-row dispersion scale used to normalise conformal scores.

    The scale must be strictly positive and known at the origin. Missing or non-positive
    values fall
    back to the median of the usable ones, so that a sensor outage widens the interval
    rather than
    collapsing it to zero.

    Args:
        frame: Rows whose scale is needed.
        scale_column: Scale column name, or ``None`` for an unnormalised method.

    Returns:
        An array of positive scales, one per row.
    """
    ones = np.ones(len(frame), dtype="float64")
    if not scale_column or scale_column not in frame.columns:
        if scale_column:
            logger.warning(
                "Colonne d'échelle '{}' absente : intervalles non normalisés", scale_column
            )
        return ones
    values = np.abs(pd.to_numeric(frame[scale_column], errors="coerce").to_numpy(dtype="float64"))
    usable = values[np.isfinite(values) & (values > 1e-9)]
    if len(usable) == 0:
        logger.warning(
            "Colonne d'échelle '{}' inutilisable : intervalles non normalisés", scale_column
        )
        return ones
    fallback = float(np.median(usable))
    return np.where(np.isfinite(values) & (values > 1e-9), values, fallback)


def _group_positions(frame: pd.DataFrame, column: str) -> list[tuple[Any, np.ndarray]]:
    """Return the row positions of each horizon, ordered chronologically.

    Ordering inside a horizon matters for the adaptive recalibration: a row may only be
    bounded by
    the residuals that precede it in time.

    Args:
        frame: Row-level diagnostic frame.
        column: Horizon column name.

    Returns:
        A list of ``(horizon value, positions)`` pairs.
    """
    if column not in frame.columns:
        return [(None, np.arange(len(frame)))]
    working = frame
    time_column = next(
        (name for name in ("origin_date", "target_date") if name in frame.columns), None
    )
    if time_column is not None:
        working = frame.assign(_stamp=pd.to_datetime(frame[time_column])).sort_values(
            ["_stamp", column], kind="mergesort"
        )
    groups: list[tuple[Any, np.ndarray]] = []
    for horizon, _block in working.groupby(column, observed=True, sort=True):
        groups.append((horizon, working.index[working[column] == horizon].to_numpy()))
    return groups


def _select_horizon(frame: pd.DataFrame, column: str, horizon: Any) -> pd.DataFrame:
    """Return the calibration rows of one horizon.

    Args:
        frame: Calibration frame (validation residuals).
        column: Horizon column name.
        horizon: Horizon value, or ``None`` when the frame carries no horizon.

    Returns:
        The selected rows (the whole frame when no horizon is available).
    """
    if horizon is None or column not in frame.columns:
        return frame
    return frame[frame[column] == horizon]


def _calm_mask(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Flag the evaluated rows that belong to no extreme regime.

    Reporting a single coverage number hides the only fact that matters operationally: the
    interval
    usually covers well on calm days and not at all during a regime the calibration window never
    saw. The split is therefore measured, not assumed.

    Args:
        frame: Row-level diagnostic frame.
        positions: Row positions of the current horizon.

    Returns:
        A boolean mask over ``positions`` (``True`` = calm day).
    """
    if "event_type" in frame.columns:
        values = frame["event_type"].to_numpy()[positions]
        return np.asarray([str(value) == "none" for value in values], dtype=bool)
    if "is_extreme_event" in frame.columns:
        return ~frame["is_extreme_event"].to_numpy()[positions].astype(bool)
    return np.ones(len(positions), dtype=bool)


def _default_context_columns(settings: ForecastSettings) -> tuple[str, ...]:
    """Return the context columns to carry into the diagnostic frame.

    Args:
        settings: Resolved forecasting settings.

    Returns:
        Column names; those absent from the context are silently ignored.
    """
    columns = [settings.horizon_column, "target_month", "target_date", "event_type"]
    if settings.time_column:
        columns.append(settings.time_column)
    return tuple(dict.fromkeys(columns))


def _segment_candidates(context: pd.DataFrame | None) -> list[str]:
    """Return the low-cardinality columns of a context frame usable as segmentation axes.

    Args:
        context: Enriched raw rows, or ``None``.

    Returns:
        Candidate column names.
    """
    if context is None:
        return []
    candidates: list[str] = []
    for column in context.columns:
        series = context[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
            if series.nunique(dropna=True) <= MAX_SEGMENT_CARDINALITY:
                candidates.append(str(column))
        elif pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 12:
            candidates.append(str(column))
    return candidates


def _bucket_labels(values: pd.Series, n_buckets: int = 6) -> pd.Series:
    """Discretise a numeric column into readable quantile buckets.

    Args:
        values: Numeric series.
        n_buckets: Number of buckets.

    Returns:
        A categorical series of interval labels.
    """
    try:
        return pd.qcut(values, q=n_buckets, duplicates="drop").astype(str)
    except (ValueError, TypeError):
        return values.astype(str)


def _mape(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent, ignoring zero ground truth.

    Args:
        truth: Observed values.
        predicted: Forecasts.

    Returns:
        The MAPE in percent (NaN when every observed value is zero).
    """
    truth = np.asarray(truth, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    mask = np.abs(truth) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.nanmean(np.abs((truth[mask] - predicted[mask]) / truth[mask])) * 100.0)


def _smape(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE, in percent, bounded to [0, 200].

    Args:
        truth: Observed values.
        predicted: Forecasts.

    Returns:
        The sMAPE in percent.
    """
    truth = np.asarray(truth, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    denominator = (np.abs(truth) + np.abs(predicted)) / 2.0
    mask = denominator > 1e-8
    if not mask.any():
        return 0.0
    return float(np.nanmean(np.abs(truth[mask] - predicted[mask]) / denominator[mask]) * 100.0)


def _improvement(reference: float, model: float) -> float:
    """Relative improvement of the model over a reference, in percent.

    Args:
        reference: Reference error (same unit as ``model``).
        model: Model error.

    Returns:
        ``100 * (reference - model) / reference``; positive means the model is better.
    """
    if not np.isfinite(reference) or abs(reference) < 1e-9:
        return float("nan")
    return float(100.0 * (reference - model) / reference)


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    """Lag-k autocorrelation of a residual series.

    Residual autocorrelation is the diagnostic that a random-split model never sees: a strongly
    autocorrelated residual means the model leaves temporal structure on the table (and that
    naive
    Gaussian intervals are too narrow).

    Args:
        values: Residuals, chronologically ordered.
        lag: Lag order.

    Returns:
        The autocorrelation coefficient, or NaN when it cannot be computed.
    """
    values = np.asarray(values, dtype="float64")
    if len(values) <= lag + 1:
        return float("nan")
    first = values[:-lag]
    second = values[lag:]
    first = first - float(np.nanmean(first))
    second = second - float(np.nanmean(second))
    denominator = float(np.sqrt(np.nansum(first**2) * np.nansum(second**2)))
    if denominator < 1e-12:
        return float("nan")
    return float(np.nansum(first * second) / denominator)


def _to_float(value: Any) -> float:
    """Coerce a value to float, mapping anything unusable to NaN.

    Args:
        value: Value to coerce.

    Returns:
        The float value.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result
