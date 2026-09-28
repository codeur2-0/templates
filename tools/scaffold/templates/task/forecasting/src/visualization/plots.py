"""Figures used by the evaluation report and the notebooks of a **forecasting** project.

Every function returns the path of the written PNG: figures are artefacts, not side effects. The
rendering backend is forced to ``Agg`` so the code works headless (CI, containers, SSH sessions).

The figure set answers the questions a forecast reviewer actually asks, in order:

1. does the curve follow the observed load? (``forecast_vs_actual``)
2. how does the error degrade with the horizon, and does the model beat the naive reference at
   every horizon? (``error_by_horizon``, ``improvement_vs_baselines``)
3. can the published interval be trusted? (``interval_coverage``)
4. is the error stable over time, or drifting? (``backtest_stability``, ``residual_diagnostics``)
5. where does it fail — which regime, which month, which day? (``error_by_regime``,
   ``bias_by_month``, ``worst_errors``)
6. what does the model actually lean on? (``feature_importance``)

Figures never raise: a missing table returns ``None`` and :meth:`ForecastingPlots.save_all` skips
it, so a partial evaluation still produces a readable report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_DPI = 140

#: Relative tolerance band (± %) the business accepts for a forecast published as is.
TOLERANCE_PCT = 5.0

#: Number of worst days displayed by :meth:`ForecastingPlots.worst_errors`.
TOP_ERRORS = 12

#: Colour of the observed series, the forecast and the naive reference, kept constant
#everywhere so
#: that a reader never has to re-learn which line is which.
COLOR_ACTUAL = "#1f4e79"
COLOR_FORECAST = "#c0504d"
COLOR_NAIVE = "#7f7f7f"


def _save(fig: Any, path: str | Path, *, close: bool = True) -> Path:
    """Write a figure to disk.

    Args:
        fig: Matplotlib figure.
        path: Destination file.
        close: Close the figure after writing (avoids memory leaks in loops).

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=DEFAULT_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    logger.debug("Figure written: {}", destination)
    return destination


def _diagnostic_frame(result: Any) -> pd.DataFrame:
    """Return the row-level diagnostic frame of an evaluation result.

    Args:
        result: Evaluation result (or any object exposing ``predictions``).

    Returns:
        The diagnostic frame (possibly empty).
    """
    predictions = getattr(result, "predictions", None)
    if isinstance(predictions, pd.DataFrame):
        return predictions
    if predictions is None:
        return pd.DataFrame()
    frame = pd.DataFrame({"y_pred": np.asarray(predictions, dtype="float64").ravel()})
    labels = getattr(result, "labels", None)
    if labels is not None and np.ndim(labels) == 1 and len(np.asarray(labels)) == len(frame):
        frame["y_true"] = np.asarray(labels, dtype="float64").ravel()
    return frame


def _table(result: Any, name: str) -> pd.DataFrame:
    """Return a table of the evaluation result, or an empty frame.

    Args:
        result: Evaluation result.
        name: Attribute name (``per_horizon``, ``baselines``, ``backtest``, ``intervals``).

    Returns:
        The table, never ``None``.
    """
    value = getattr(result, name, None)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _horizon_column(result: Any, frame: pd.DataFrame) -> str | None:
    """Resolve the horizon column name without hard-coding it.

    The name is declared in the project configuration and echoed in ``result.extras``; the
    fallback
    looks for any column whose name mentions a horizon, so the figures still work on an ad-hoc
    result built in a notebook.

    Args:
        result: Evaluation result.
        frame: Row-level diagnostic frame.

    Returns:
        The horizon column name, or ``None`` when the frame carries no horizon.
    """
    extras = getattr(result, "extras", None) or {}
    declared = extras.get("horizon_column")
    if isinstance(declared, str) and declared in frame.columns:
        return declared
    for column in frame.columns:
        if "horizon" in str(column).lower():
            return str(column)
    return None


def _time_column(result: Any, frame: pd.DataFrame) -> str | None:
    """Resolve a column usable as the chronological axis.

    Args:
        result: Evaluation result.
        frame: Row-level diagnostic frame.

    Returns:
        The time column name, or ``None``.
    """
    for column in ("origin_date", "target_date"):
        if column in frame.columns:
            return column
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column)
    del result  # aucune information supplémentaire à lire sur le résultat
    return None


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Coerce ``value`` to ``float`` without raising.

    Args:
        value: Value to coerce (may be ``None``, a NumPy scalar, ...).
        default: Value returned when the coercion fails.

    Returns:
        The coerced float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ForecastingPlots:
    """Figure factory for a forecast evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = "viridis") -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Seaborn/matplotlib palette for density and heat maps.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ the curve -------
    def forecast_vs_actual(
        self, result: Any, *, name: str = "forecast_vs_actual.png", max_points: int = 240
    ) -> Path | None:
        """Plot the observed load against the forecast, in chronological order.

        Only one horizon is drawn (the shortest available, i.e. the daily balancing
        forecast) so the
        two curves stay comparable: overlaying four horizons would hide the J+1 error behind the
        J+7 one. The naive reference is drawn in grey to make the added value visible at
        a glance.

        Args:
            result: Evaluation result.
            name: Output file name.
            max_points: Maximum number of origins drawn (keeps the figure readable).

        Returns:
            The written path, or ``None`` when the frame carries no time axis.
        """
        frame = _diagnostic_frame(result)
        time_column = _time_column(result, frame)
        if frame.empty or time_column is None or "y_true" not in frame.columns:
            return None
        horizon_column = _horizon_column(result, frame)
        plot_frame = frame
        if horizon_column is not None:
            shortest = frame[horizon_column].min()
            plot_frame = frame[frame[horizon_column] == shortest]
        plot_frame = plot_frame.sort_values(time_column).tail(int(max_points))

        fig, axis = plt.subplots(figsize=(9.6, 3.9))
        stamps = pd.to_datetime(plot_frame[time_column])
        axis.plot(stamps, plot_frame["y_true"], color=COLOR_ACTUAL, lw=1.7, label="observé")
        axis.plot(
            stamps, plot_frame["y_pred"], color=COLOR_FORECAST, lw=1.4, ls="--", label="prévision"
        )
        for column, label in (("load_seasonal_naive", "naif saisonnier"),):
            if column in plot_frame.columns:
                axis.plot(stamps, plot_frame[column], color=COLOR_NAIVE, lw=1.0, ls=":", label=label)
        if {"lower", "upper"} <= set(plot_frame.columns):
            axis.fill_between(
                stamps,
                plot_frame["lower"],
                plot_frame["upper"],
                color=COLOR_FORECAST,
                alpha=0.13,
                label="intervalle",
            )
        axis.set_ylabel("Consommation (MW)")
        axis.set_title(
            "Prévision vs consommation observée — horizon"
            f" {plot_frame[horizon_column].iloc[0] if horizon_column else '-'} j "
            f"({len(plot_frame)} origines)"
        )
        axis.legend(loc="best", fontsize=8, ncol=4)
        fig.autofmt_xdate(rotation=30)
        return _save(fig, self.figures_dir / name)

    def predicted_vs_actual(
        self, result: Any, *, name: str = "predicted_vs_actual.png"
    ) -> Path | None:
        """Scatter forecast against observation, with the identity line and the tolerance band.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the frame is unusable.
        """
        frame = _diagnostic_frame(result)
        if frame.empty or {"y_true", "y_pred"} > set(frame.columns):
            return None
        truth = frame["y_true"].to_numpy(dtype="float64")
        predicted = frame["y_pred"].to_numpy(dtype="float64")
        fig, axis = plt.subplots(figsize=(5.4, 5.0))
        axis.scatter(truth, predicted, s=13, alpha=0.42, color=COLOR_FORECAST, edgecolors="none")
        low = float(min(truth.min(), predicted.min()))
        high = float(max(truth.max(), predicted.max()))
        span = np.linspace(low, high, 50)
        axis.plot(span, span, color=COLOR_ACTUAL, lw=1.6, label="prévision parfaite")
        axis.plot(
            span, span * (1 + TOLERANCE_PCT / 100.0), color=COLOR_NAIVE, lw=1.0, ls=":", label=f"±{TOLERANCE_PCT:.0f} %"
        )
        axis.plot(span, span * (1 - TOLERANCE_PCT / 100.0), color=COLOR_NAIVE, lw=1.0, ls=":")
        axis.set_xlabel("Consommation observée (MW)")
        axis.set_ylabel("Prévision (MW)")
        axis.set_title("Prévision vs observation")
        axis.legend(loc="upper left", fontsize=8)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ by horizon ------
    def error_by_horizon(self, result: Any, *, name: str = "error_by_horizon.png") -> Path | None:
        """Bar chart of the MAPE per horizon, model against naive reference.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the per-horizon table is absent.
        """
        table = _table(result, "per_horizon")
        if table.empty or "mape_pct" not in table.columns:
            return None
        labels = [f"J+{value}" for value in table["horizon"].tolist()]
        positions = np.arange(len(table))
        width = 0.38 if "naive_mape_pct" in table.columns else 0.6

        fig, axis = plt.subplots(figsize=(7.2, 3.9))
        axis.bar(
            positions - width / 2 if "naive_mape_pct" in table.columns else positions,
            table["mape_pct"],
            width=width,
            color=COLOR_FORECAST,
            label="modèle",
        )
        if "naive_mape_pct" in table.columns:
            axis.bar(
                positions + width / 2,
                table["naive_mape_pct"],
                width=width,
                color=COLOR_NAIVE,
                label="naif saisonnier",
            )
        axis.set_xticks(positions, labels)
        axis.set_ylabel("MAPE (%)")
        axis.set_title("Erreur par horizon — le modèle doit rester sous la référence à chaque horizon")
        axis.legend(fontsize=8)
        for index, value in enumerate(table["mape_pct"]):
            axis.text(
                positions[index] - (width / 2 if "naive_mape_pct" in table.columns else 0),
                _safe_float(value),
                f"{_safe_float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        return _save(fig, self.figures_dir / name)

    def improvement_vs_baselines(
        self, result: Any, *, name: str = "improvement_vs_baselines.png"
    ) -> Path | None:
        """Horizontal bars of the relative improvement over each trivial reference.

        A negative bar is the honest verdict of the project: the model would be worse
        than what an
        operator does by hand, and no R² can hide it.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the baseline table is absent.
        """
        table = _table(result, "baselines")
        if table.empty or "improvement_pct" not in table.columns:
            return None
        references = table[table["reference"] != "modèle appris"]
        if references.empty:
            return None
        references = references.sort_values("improvement_pct")
        colours = [
            "#2e7d32" if _safe_float(value) > 0 else "#b3261e"
            for value in references["improvement_pct"]
        ]
        fig, axis = plt.subplots(figsize=(7.6, 3.4))
        axis.barh(
            references["reference"].astype(str),
            references["improvement_pct"].astype(float),
            color=colours,
        )
        axis.axvline(0.0, color="#333333", lw=1.0)
        axis.set_xlabel("Gain de MAPE du modèle par rapport à la référence (%)")
        axis.set_title("Valeur ajoutée réelle : ce que le modèle apporte face aux références triviales")
        for index, value in enumerate(references["improvement_pct"].astype(float)):
            axis.text(
                value,
                index,
                f" {value:+.1f} %",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ intervals -------
    def interval_coverage(self, result: Any, *, name: str = "interval_coverage.png") -> Path | None:
        """Realised coverage per horizon against the nominal level, plus the interval width.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no interval could be built.
        """
        table = _table(result, "intervals")
        if table.empty or "coverage_pct" not in table.columns:
            return None
        nominal = _safe_float(table["level_pct"].iloc[0], 90.0)
        labels = [f"J+{value}" for value in table["horizon"].tolist()]
        fig, axis = plt.subplots(figsize=(7.2, 3.9))
        bars = axis.bar(labels, table["coverage_pct"].astype(float), color="#4472c4", alpha=0.85)
        axis.axhline(nominal, color="#b3261e", lw=1.4, ls="--", label=f"nominal {nominal:.0f} %")
        axis.set_ylim(0.0, 105.0)
        axis.set_ylabel("Couverture réalisée (%)")
        axis.set_title("Couverture de l'intervalle de prévision par horizon")
        for bar, width in zip(bars, table["mean_width_mw"].astype(float), strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f"±{width / 2.0:.0f} MW",
                ha="center",
                fontsize=7.5,
            )
        axis.legend(fontsize=8)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ stability -------
    def backtest_stability(self, result: Any, *, name: str = "backtest_stability.png") -> Path | None:
        """MAPE per rolling-origin fold: is the error stable, or drifting?

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the backtest table is absent.
        """
        table = _table(result, "backtest")
        if table.empty or "mape_pct" not in table.columns:
            return None
        folds = table[table["fold"].astype(str) != "dispersion"]
        if folds.empty:
            return None
        spread = table[table["fold"].astype(str) == "dispersion"]
        labels = [str(value) for value in folds["fold"].tolist()]
        fig, axis = plt.subplots(figsize=(7.2, 3.6))
        axis.plot(labels, folds["mape_pct"].astype(float), marker="o", color=COLOR_FORECAST, lw=1.8)
        axis.axhline(
            float(folds["mape_pct"].astype(float).mean()),
            color=COLOR_NAIVE,
            ls="--",
            lw=1.1,
            label="MAPE moyen",
        )
        if not spread.empty:
            axis.set_title(
                "Backtest à origine glissante — dispersion inter-replis "
                f"{_safe_float(spread['mape_pct'].iloc[0]):.2f} point de %"
            )
        else:
            axis.set_title("Backtest à origine glissante")
        axis.set_xlabel("Repli chronologique")
        axis.set_ylabel("MAPE (%)")
        axis.legend(fontsize=8)
        return _save(fig, self.figures_dir / name)

    def residual_diagnostics(
        self, result: Any, *, name: str = "residual_diagnostics.png"
    ) -> Path | None:
        """Two panels: residual autocorrelation, and residuals over time coloured by horizon.

        Autocorrelated residuals mean the model leaves temporal structure on the table,
        and that a
        Gaussian interval computed from their standard deviation is too narrow.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no residual is available.
        """
        frame = _diagnostic_frame(result)
        if frame.empty or "residual" not in frame.columns:
            return None
        extras = getattr(result, "extras", None) or {}
        ac = extras.get("residual_autocorrelation") or {}

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
        if ac:
            lags = [int(str(key).split("_")[-1]) for key in ac]
            values = [_safe_float(ac[key]) for key in ac]
            threshold = 2.0 / np.sqrt(max(len(frame), 1))
            axes[0].bar([str(lag) for lag in lags], values, color="#4472c4")
            axes[0].axhline(threshold, color="#b3261e", ls="--", lw=1.0, label="±2/√n")
            axes[0].axhline(-threshold, color="#b3261e", ls="--", lw=1.0)
            axes[0].axhline(0.0, color="#333333", lw=0.8)
            axes[0].set_title("Autocorrélation des résidus")
            axes[0].set_xlabel("décalage (jours)")
            axes[0].set_ylabel("coefficient")
            axes[0].legend(fontsize=8)
        else:
            axes[0].set_title("Autocorrélation des résidus (indisponible)")

        horizon_column = _horizon_column(result, frame)
        if horizon_column is not None:
            scatter = axes[1].scatter(
                np.arange(len(frame)),
                frame["residual"].to_numpy(dtype="float64"),
                c=pd.to_numeric(frame[horizon_column], errors="coerce"),
                cmap=self.palette,
                s=11,
                alpha=0.65,
            )
            fig.colorbar(scatter, ax=axes[1], label="horizon (jours)", shrink=0.85)
        else:
            axes[1].scatter(
                np.arange(len(frame)), frame["residual"].to_numpy(dtype="float64"), s=11, alpha=0.6
            )
        axes[1].axhline(0.0, color="#333333", lw=1.0)
        axes[1].set_title("Résidus (prévision - observation)")
        axes[1].set_xlabel("observation (ordre chronologique)")
        axes[1].set_ylabel("MW")
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ where it fails --
    def error_by_regime(self, result: Any, *, name: str = "error_by_regime.png") -> Path | None:
        """MAPE per regime of the target day (none, cold snap, heatwave, industrial shutdown).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no regime column is available.
        """
        table = _table(result, "per_segment")
        if table.empty or "axis" not in table.columns:
            return None
        regime = table[table["axis"].isin(["event_type", "is_extreme_event"])]
        if regime.empty:
            return None
        regime = regime.sort_values("mape_pct", ascending=False)
        labels = [f"{row.axis}\n{row.segment}" for row in regime.itertuples()]
        fig, axis = plt.subplots(figsize=(7.8, 3.8))
        bars = axis.bar(labels, regime["mape_pct"].astype(float), color="#8064a2")
        axis.set_ylabel("MAPE (%)")
        axis.set_title("Erreur par régime du jour cible — là où la prévision vaut de l'argent")
        for bar, rows in zip(bars, regime["rows"].astype(int), strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"n={rows}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def bias_by_month(self, result: Any, *, name: str = "bias_by_month.png") -> Path | None:
        """Signed relative bias per month: a systematic under-forecast costs money in emergencies.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the monthly bias is unavailable.
        """
        extras = getattr(result, "extras", None) or {}
        bias = extras.get("bias_by_month") or {}
        if not bias:
            return None
        order = [
            month
            for month in (
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            )
            if month in bias
        ]
        values = [_safe_float(bias[month]) for month in order]
        colours = ["#b3261e" if value < 0 else "#2e7d32" for value in values]
        fig, axis = plt.subplots(figsize=(7.6, 3.4))
        axis.bar(order, values, color=colours)
        axis.axhline(0.0, color="#333333", lw=1.0)
        axis.set_ylabel("Biais relatif moyen (%)")
        axis.set_title("Biais par mois — négatif = sous-prévision systématique")
        return _save(fig, self.figures_dir / name)

    def error_breakdown(self, result: Any, *, name: str = "error_breakdown.png") -> Path | None:
        """MAPE for every segmentation axis, in one small-multiples figure.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no segment is available.
        """
        table = _table(result, "per_segment")
        if table.empty or "axis" not in table.columns:
            return None
        axes_names = [str(axis) for axis in table["axis"].unique()][:4]
        if not axes_names:
            return None
        fig, axes = plt.subplots(1, len(axes_names), figsize=(4.1 * len(axes_names), 3.6))
        axes_list = [axes] if len(axes_names) == 1 else list(axes)
        for axis, name_ in zip(axes_list, axes_names, strict=False):
            block = table[table["axis"] == name_].sort_values("mape_pct", ascending=False).head(10)
            axis.barh(block["segment"].astype(str), block["mape_pct"].astype(float), color="#4472c4")
            axis.invert_yaxis()
            axis.set_title(f"MAPE par {name_}", fontsize=9.5)
            axis.set_xlabel("MAPE (%)", fontsize=8)
            axis.tick_params(labelsize=7.5)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def worst_errors(self, result: Any, *, name: str = "worst_errors.png") -> Path | None:
        """The worst forecast days, labelled with their horizon and regime.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the error table is absent.
        """
        errors = _table(result, "errors")
        if errors.empty or "ape_pct" not in errors.columns:
            return None
        block = errors.head(TOP_ERRORS).iloc[::-1]
        labels: list[str] = []
        for row in block.itertuples():
            stamp = getattr(row, "origin_date", None) or getattr(row, "target_date", None)
            horizon = getattr(row, "horizon_days", None)
            event = getattr(row, "event_type", None)
            pieces = [str(pd.to_datetime(stamp).date()) if stamp is not None else str(len(labels))]
            if horizon is not None:
                pieces.append(f"J+{horizon}")
            if event is not None and str(event) != "none":
                pieces.append(str(event))
            labels.append(" · ".join(pieces))
        fig, axis = plt.subplots(figsize=(8.4, 4.2))
        axis.barh(labels, block["ape_pct"].astype(float), color="#c0504d")
        axis.set_xlabel("Erreur absolue en pourcentage (%)")
        axis.set_title(f"Les {len(block)} pires prévisions — à lire une par une avant de conclure")
        axis.tick_params(labelsize=7.5)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ model -----------
    def feature_importance(
        self, result: Any, *, name: str = "feature_importance.png", top_k: int = 14
    ) -> Path | None:
        """Top feature importances: what does the forecast actually lean on?

        Args:
            result: Evaluation result.
            name: Output file name.
            top_k: Number of features displayed.

        Returns:
            The written path, or ``None`` when the model exposes no importance.
        """
        table = _table(result, "feature_importance")
        if table.empty or "importance" not in table.columns:
            return None
        block = table.head(top_k).iloc[::-1]
        method = str(block["method"].iloc[0]) if "method" in block.columns else "native"
        fig, axis = plt.subplots(figsize=(7.4, 4.4))
        axis.barh(block["feature"].astype(str), block["importance"].astype(float), color="#4b7f52")
        axis.set_xlabel(f"Importance ({method})")
        axis.set_title("Variables qui portent la prévision")
        axis.tick_params(labelsize=8)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def metrics_bar(self, result: Any, *, name: str = "metrics_bar.png") -> Path | None:
        """Bar chart of the reported metrics (percent-scale metrics only, to stay readable).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no metric is available.
        """
        metrics = dict(getattr(result, "metrics", {}) or {})
        if not metrics:
            return None
        selected = {
            key: value
            for key, value in metrics.items()
            if key in {"mape", "smape", "mase", "interval_coverage"} and np.isfinite(_safe_float(value))
        }
        if not selected:
            return None
        fig, axis = plt.subplots(figsize=(6.4, 3.4))
        axis.bar(list(selected), [_safe_float(value) for value in selected.values()], color="#4472c4")
        axis.set_ylabel("valeur")
        axis.set_title("Métriques de prévision (MAPE et sMAPE en %, MASE sans unité)")
        for index, value in enumerate(selected.values()):
            axis.text(index, _safe_float(value), f"{_safe_float(value):.3f}", ha="center", va="bottom", fontsize=8)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ orchestration ---
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Render every available figure of a forecast evaluation.

        Args:
            result: Evaluation result.
            thresholds: Accepted for API symmetry with the classification plots; a
            forecast has no
                decision threshold to scan, so the argument is ignored.

        Returns:
            Mapping of figure name to written path (missing figures are skipped).
        """
        del thresholds  # aucune notion de seuil de décision en prévision
        produced: dict[str, Path] = {}
        renderers = (
            ("forecast_vs_actual", self.forecast_vs_actual),
            ("error_by_horizon", self.error_by_horizon),
            ("improvement_vs_baselines", self.improvement_vs_baselines),
            ("interval_coverage", self.interval_coverage),
            ("backtest_stability", self.backtest_stability),
            ("residual_diagnostics", self.residual_diagnostics),
            ("error_by_regime", self.error_by_regime),
            ("bias_by_month", self.bias_by_month),
            ("error_breakdown", self.error_breakdown),
            ("predicted_vs_actual", self.predicted_vs_actual),
            ("worst_errors", self.worst_errors),
            ("feature_importance", self.feature_importance),
            ("metrics_bar", self.metrics_bar),
        )
        for name, renderer in renderers:
            try:
                path = renderer(result)
            except Exception as error:  # noqa: BLE001 - une figure ne doit jamais casser un rapport
                logger.warning("Figure '{}' non générée : {}", name, error)
                continue
            if path is not None:
                produced[name] = path
        logger.info("{} figures de prévision générées dans {}", len(produced), self.figures_dir)
        return produced


__all__ = ["COLOR_ACTUAL", "COLOR_FORECAST", "ForecastingPlots", "TOLERANCE_PCT", "TOP_ERRORS"]
