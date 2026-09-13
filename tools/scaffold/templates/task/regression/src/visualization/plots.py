"""Figures used by the evaluation report and the notebooks.

Every function returns the path of the written PNG: figures are artefacts, not side effects.
The rendering backend is forced to ``Agg`` so that the code works headless (CI, containers).

The figure set is built for a **regression** target: it answers the three questions a reviewer
asks about a price/amount model — is it biased?, how wide are the errors?, and **where** are the
worst errors concentrated (which segment, which price range)?
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

#: Nombre de classes de prix utilisées pour les analyses par segment.
DEFAULT_BUCKETS = 8

#: Fourchette relative (± %) considérée comme « acceptable » par le métier.
TOLERANCE_PCT = 10.0


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

    ``EvaluationResult.predictions`` is a ``DataFrame`` (``y_true``, ``y_pred``, ``residual``,
    ``relative_error_pct``, ``within_tolerance``, ``price_bucket`` + context columns). Raw arrays
    are accepted as a fallback so that the figures also work with an ad-hoc result object.

    Args:
        result: Evaluation result (or any object exposing ``predictions`` / ``labels``).

    Returns:
        The diagnostic frame (possibly empty).
    """
    predictions = getattr(result, "predictions", None)
    if isinstance(predictions, pd.DataFrame):
        return predictions
    if predictions is None:
        return pd.DataFrame()
    labels = getattr(result, "labels", None)
    frame = pd.DataFrame({"y_pred": np.asarray(predictions, dtype="float64").ravel()})
    if labels is not None and np.ndim(labels) == 1 and len(np.asarray(labels)) == len(frame):
        frame["y_true"] = np.asarray(labels, dtype="float64").ravel()
    if "y_true" in frame.columns:
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["relative_error_pct"] = 100.0 * frame["residual"] / frame["y_true"]
        frame["within_tolerance"] = (
            frame["relative_error_pct"].abs() <= TOLERANCE_PCT
        ).astype("int64")
    return frame


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


class RegressionPlots:
    """Figure factory for a regression evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = "viridis") -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Seaborn/matplotlib palette for density and heat maps.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ diagnostics -----
    def predicted_vs_actual(self, result: Any, *, name: str = "predicted_vs_actual.png") -> Path | None:
        """Scatter predicted vs observed values, with the identity line and a tolerance band.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult` (or any object exposing
                ``predictions`` and ``labels``).
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        frame = _diagnostic_frame(result)
        if frame.empty or not {"y_true", "y_pred"} <= set(frame.columns):
            logger.debug("predicted_vs_actual ignoré : aucune prédiction exploitable")
            return None
        predictions = frame["y_pred"].to_numpy(dtype="float64")
        labels = frame["y_true"].to_numpy(dtype="float64")

        fig, axes = plt.subplots(figsize=(7.2, 6.4))
        axes.scatter(labels, predictions, s=14, alpha=0.45, edgecolors="none")
        low = float(min(labels.min(), predictions.min()))
        high = float(max(labels.max(), predictions.max()))
        span = np.linspace(low, high, 100)
        axes.plot(span, span, color="black", linewidth=1.6, label="prédiction parfaite")
        axes.plot(
            span,
            span * (1 + TOLERANCE_PCT / 100.0),
            color="grey",
            linestyle="--",
            linewidth=1.0,
            label=f"bande ± {TOLERANCE_PCT:.0f} %",
        )
        axes.plot(span, span * (1 - TOLERANCE_PCT / 100.0), color="grey", linestyle="--", linewidth=1.0)
        axes.set_xlabel("valeur observée")
        axes.set_ylabel("valeur prédite")
        axes.set_title("Prédit vs observé — régression")
        axes.legend(loc="upper left", fontsize=9)
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.set_aspect("equal", adjustable="box")
        return _save(fig, self.figures_dir / name)

    def residuals_vs_predicted(
        self, result: Any, *, name: str = "residuals_vs_predicted.png"
    ) -> Path | None:
        """Residuals against predicted values — the standard heteroscedasticity check.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        frame = _diagnostic_frame(result)
        if frame.empty or not {"y_pred", "residual"} <= set(frame.columns):
            return None
        predictions = frame["y_pred"].to_numpy(dtype="float64")
        residuals = frame["residual"].to_numpy(dtype="float64")

        fig, axes = plt.subplots(figsize=(7.6, 5.4))
        axes.axhline(0.0, color="black", linewidth=1.2)
        axes.scatter(predictions, residuals, s=13, alpha=0.4, edgecolors="none")

        # Moyenne glissante par classe de prédiction : révèle biais et hétéroscédasticité.
        frame = pd.DataFrame({"prediction": predictions, "residual": residuals})
        try:
            frame["bucket"] = pd.qcut(frame["prediction"], q=min(DEFAULT_BUCKETS, 8), duplicates="drop")
        except ValueError:
            frame["bucket"] = "all"
        summary = frame.groupby("bucket", observed=True)["residual"].mean()
        centres = frame.groupby("bucket", observed=True)["prediction"].median()
        if len(summary) > 1:
            axes.plot(
                centres.to_numpy(dtype="float64"),
                summary.to_numpy(dtype="float64"),
                color="crimson",
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="résidu moyen par classe",
            )
            axes.legend(loc="best", fontsize=9)
        axes.set_xlabel("valeur prédite")
        axes.set_ylabel("résidu (prédit − observé)")
        axes.set_title("Résidus vs prédiction — biais et hétéroscédasticité")
        return _save(fig, self.figures_dir / name)

    def error_distribution(self, result: Any, *, name: str = "error_distribution.png") -> Path | None:
        """Distribution of the signed relative error, with median and 95th percentile markers.

        Args:
            result: Evaluation result (expects ``extras["relative_error_pct"]``).
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        frame = _diagnostic_frame(result)
        extras = getattr(result, "extras", None) or {}
        relative = (
            frame["relative_error_pct"].to_numpy(dtype="float64")
            if "relative_error_pct" in frame.columns
            else extras.get("relative_error_pct")
        )
        if relative is None:
            return None
        values = pd.to_numeric(pd.Series(relative), errors="coerce").dropna().to_numpy(dtype="float64")
        if values.size == 0:
            return None

        fig, axes = plt.subplots(figsize=(7.6, 5.0))
        sns.histplot(values, bins=60, kde=True, ax=axes, color="steelblue")
        median = float(np.median(values))
        p95 = float(np.percentile(np.abs(values), 95))
        axes.axvline(0.0, color="black", linewidth=1.2, label="erreur nulle")
        axes.axvline(median, color="crimson", linestyle="--", linewidth=1.4, label=f"médiane {median:+.2f} %")
        axes.axvline(p95, color="darkorange", linestyle=":", linewidth=1.4, label=f"P95 |erreur| {p95:.2f} %")
        axes.axvline(-p95, color="darkorange", linestyle=":", linewidth=1.4)
        axes.set_xlabel("erreur relative signée (%)")
        axes.set_ylabel("nombre de biens")
        axes.set_title("Distribution de l'erreur relative")
        axes.legend(loc="best", fontsize=9)
        return _save(fig, self.figures_dir / name)

    def error_by_bucket(self, result: Any, *, name: str = "error_by_bucket.png") -> Path | None:
        """Box plot of the signed relative error per target bucket (bias by price range).

        Args:
            result: Evaluation result (expects ``extras["error_by_bucket"]``).
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        extras = getattr(result, "extras", None) or {}
        frame = extras.get("error_by_bucket")
        if frame is None or len(frame) == 0:
            return None
        data = frame.copy()
        data["bucket"] = data["bucket"].astype(str)

        fig, axes = plt.subplots(figsize=(8.4, 5.2))
        sns.boxplot(
            data=data,
            x="bucket",
            y="relative_error_pct",
            ax=axes,
            palette=self.palette,
            showfliers=False,
        )
        axes.axhline(0.0, color="black", linewidth=1.2)
        axes.axhline(TOLERANCE_PCT, color="grey", linestyle="--", linewidth=1.0)
        axes.axhline(-TOLERANCE_PCT, color="grey", linestyle="--", linewidth=1.0)
        axes.set_xlabel("classe de valeur observée")
        axes.set_ylabel("erreur relative signée (%)")
        axes.set_title("Erreur relative par classe de prix — biais segmenté")
        axes.tick_params(axis="x", rotation=25)
        return _save(fig, self.figures_dir / name)

    def coverage_by_bucket(self, result: Any, *, name: str = "coverage_by_bucket.png") -> Path | None:
        """Coverage of the business tolerance band per target bucket.

        Args:
            result: Evaluation result (expects ``extras["coverage_by_bucket"]``).
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        extras = getattr(result, "extras", None) or {}
        frame = extras.get("coverage_by_bucket")
        if frame is None or len(frame) == 0:
            return None
        data = frame.copy()
        data["bucket"] = data["bucket"].astype(str)
        data["coverage"] = data["coverage"].astype("float64") * 100.0

        fig, axes = plt.subplots(figsize=(8.4, 5.0))
        bars = axes.bar(data["bucket"], data["coverage"], color="steelblue", alpha=0.85)
        axes.axhline(75.0, color="crimson", linestyle="--", linewidth=1.4, label="cible métier (75 %)")
        for bar, value in zip(bars, data["coverage"], strict=False):
            axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.0,
                f"{value:.0f} %",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes.set_ylim(0.0, 105.0)
        axes.set_xlabel("classe de valeur observée")
        axes.set_ylabel(f"part dans la bande ± {TOLERANCE_PCT:.0f} % (%)")
        axes.set_title("Couverture de la fourchette par classe de prix")
        axes.legend(loc="lower left", fontsize=9)
        axes.tick_params(axis="x", rotation=25)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ synthèse --------
    def error_breakdown(self, result: Any, *, name: str = "error_breakdown.png") -> Path | None:
        """Horizontal bars comparing the error statistics (scale-aware reading).

        Args:
            result: Evaluation result (uses ``metrics`` and ``extras``).
            name: Output file name.

        Returns:
            The written path, or ``None`` when no metric is available.
        """
        metrics = dict(getattr(result, "metrics", None) or {})
        extras = getattr(result, "extras", None) or {}
        values = {
            "erreur médiane": _safe_float(extras.get("median_absolute_error")),
            "MAE": _safe_float(metrics.get("mae")),
            "RMSE": _safe_float(metrics.get("rmse")),
            "P95 |erreur|": _safe_float(extras.get("absolute_error_p95")),
            "erreur max": _safe_float(metrics.get("max_error")),
        }
        values = {key: value for key, value in values.items() if np.isfinite(value)}
        if not values:
            return None

        fig, axes = plt.subplots(figsize=(7.6, 4.4))
        axes.barh(list(values), list(values.values()), color="steelblue", alpha=0.85)
        axes.set_xscale("log")
        axes.set_xlabel("erreur absolue (échelle logarithmique)")
        axes.set_title("Hiérarchie des erreurs — la RMSE est tirée par la queue de distribution")
        for index, value in enumerate(values.values()):
            axes.text(value * 1.08, index, f"{value:,.0f}", va="center", fontsize=8)
        axes.invert_yaxis()
        return _save(fig, self.figures_dir / name)

    def metrics_bar(self, result: Any, *, name: str = "metrics_bar.png") -> Path | None:
        """Two-panel summary: absolute errors and scale-free scores.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no metric is available.
        """
        metrics = dict(getattr(result, "metrics", None) or {})
        absolute = {
            "RMSE": _safe_float(metrics.get("rmse")),
            "MAE": _safe_float(metrics.get("mae")),
            "max_error": _safe_float(metrics.get("max_error")),
        }
        relative = {
            "MAPE (%)": _safe_float(metrics.get("mape")) * 100.0,
            "sMAPE (%)": _safe_float(metrics.get("smape")) * 100.0,
            "R²": _safe_float(metrics.get("r2")),
        }
        absolute = {key: value for key, value in absolute.items() if np.isfinite(value)}
        relative = {key: value for key, value in relative.items() if np.isfinite(value)}
        if not absolute and not relative:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
        if absolute:
            axes[0].bar(list(absolute), list(absolute.values()), color="steelblue", alpha=0.85)
            axes[0].set_yscale("log")
            axes[0].set_ylabel("valeur (échelle logarithmique)")
            axes[0].set_title("Erreurs absolues")
            axes[0].tick_params(axis="x", rotation=15)
        if relative:
            colours = ["darkorange", "darkorange", "seagreen"][: len(relative)]
            axes[1].bar(list(relative), list(relative.values()), color=colours, alpha=0.85)
            axes[1].set_ylabel("valeur")
            axes[1].set_title("Scores sans unité")
            axes[1].tick_params(axis="x", rotation=15)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def feature_importance(
        self, result: Any, *, name: str = "feature_importance.png", top_k: int = 15
    ) -> Path | None:
        """Top-K feature importance bars.

        Args:
            result: Evaluation result exposing ``feature_importance``.
            name: Output file name.
            top_k: Number of features displayed.

        Returns:
            The written path, or ``None`` when importance is unavailable.
        """
        importance = getattr(result, "feature_importance", None)
        if importance is None or len(importance) == 0:
            logger.debug("feature_importance ignoré : aucune importance disponible")
            return None
        data = importance.head(int(top_k)).iloc[::-1]
        fig, axes = plt.subplots(figsize=(7.6, 5.2))
        axes.barh(data["feature"].astype(str), data["importance"].astype("float64"), color="teal", alpha=0.85)
        axes.set_xlabel("importance")
        axes.set_title(f"Top {min(int(top_k), len(data))} facteurs explicatifs")
        return _save(fig, self.figures_dir / name)

    def worst_errors(self, result: Any, *, name: str = "worst_errors.png", top_k: int = 10) -> Path | None:
        """Bar chart of the largest individual errors (support of the error analysis).

        Args:
            result: Evaluation result exposing ``errors``.
            name: Output file name.
            top_k: Number of rows displayed.

        Returns:
            The written path, or ``None`` when the error table is unavailable.
        """
        errors = getattr(result, "errors", None)
        if errors is None or len(errors) == 0:
            return None
        frame = errors.head(int(top_k)) if isinstance(errors, pd.DataFrame) else pd.DataFrame(errors)
        if "relative_error_pct" not in frame.columns or "row_index" not in frame.columns:
            return None
        frame = frame.iloc[::-1]
        labels = [f"ligne {int(index)}" for index in frame["row_index"]]
        colours = ["crimson" if value > 0 else "steelblue" for value in frame["relative_error_pct"]]

        fig, axes = plt.subplots(figsize=(7.6, 4.8))
        axes.barh(labels, frame["relative_error_pct"].astype("float64"), color=colours, alpha=0.85)
        axes.axvline(0.0, color="black", linewidth=1.0)
        axes.set_xlabel("erreur relative signée (%)")
        axes.set_title(f"Top {min(int(top_k), len(frame))} erreurs — rouge : sur-estimation")
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ orchestration ---
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Render every available figure for a regression evaluation.

        Args:
            result: Evaluation result.
            thresholds: Accepted for API symmetry with the classification plots; a regression has
                no decision threshold to scan, so the argument is ignored.

        Returns:
            Mapping of figure name to written path (missing figures are skipped).
        """
        del thresholds  # aucune notion de seuil de décision en régression
        produced: dict[str, Path] = {}
        renderers = (
            ("predicted_vs_actual", self.predicted_vs_actual),
            ("residuals_vs_predicted", self.residuals_vs_predicted),
            ("error_distribution", self.error_distribution),
            ("error_by_bucket", self.error_by_bucket),
            ("coverage_by_bucket", self.coverage_by_bucket),
            ("error_breakdown", self.error_breakdown),
            ("metrics_bar", self.metrics_bar),
            ("feature_importance", self.feature_importance),
            ("worst_errors", self.worst_errors),
        )
        for name, renderer in renderers:
            try:
                path = renderer(result)
            except Exception as error:  # noqa: BLE001 - une figure ne doit jamais casser un rapport
                logger.warning("Figure '{}' non générée : {}", name, error)
                continue
            if path is not None:
                produced[name] = path
        logger.info("{} figures de régression générées dans {}", len(produced), self.figures_dir)
        return produced


__all__ = ["DEFAULT_BUCKETS", "RegressionPlots", "TOLERANCE_PCT"]
