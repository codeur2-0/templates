"""Figures used by the evaluation report and the notebooks.

Every function returns the path of the written PNG: figures are artefacts, not side effects.
The rendering backend is forced to ``Agg`` so that the code works headless (CI, containers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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


class ClassificationPlots:
    """Figure factory for a classification evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = "viridis") -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Seaborn/matplotlib palette for heatmaps.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette

    # ------------------------------------------------------------------ matrices --------
    def confusion_matrix(self, result: Any, *, normalize: bool = True, name: str = "confusion_matrix.png") -> Path:
        """Plot the (normalised) confusion matrix.

        Args:
            result: :class:`src.evaluation.evaluator.EvaluationResult`.
            normalize: Show row percentages instead of raw counts.
            name: Output file name.

        Returns:
            The written path.
        """
        matrix = np.asarray(result.confusion_matrix, dtype="float64")
        counts = np.asarray(result.confusion_matrix)
        #: Libellé du mode d'affichage, extrait du titre pour rester sous la limite de longueur.
        mode = "lignes normalisées" if normalize else "effectifs"
        if normalize and matrix.sum() > 0:
            matrix = matrix / matrix.sum(axis=1, keepdims=True)

        fig, axis = plt.subplots(figsize=(6.2, 5.0))
        sns.heatmap(
            matrix,
            annot=counts if not normalize else np.round(matrix, 3),
            fmt="d" if not normalize else ".2f",
            cmap=self.palette,
            xticklabels=result.labels,
            yticklabels=result.labels,
            cbar=True,
            ax=axis,
            linewidths=0.6,
        )
        axis.set_xlabel("Classe prédite")
        axis.set_ylabel("Classe réelle")
        axis.set_title(f"Matrice de confusion ({mode}) — split {result.split}")
        return _save(fig, self.figures_dir / name)

    def per_class_metrics(self, per_class: pd.DataFrame, *, name: str = "per_class_metrics.png") -> Path:
        """Plot precision / recall / F1 per class.

        Args:
            per_class: DataFrame produced by the evaluator.
            name: Output file name.

        Returns:
            The written path.
        """
        frame = per_class.set_index("class")[["precision", "recall", "f1"]]
        fig, axis = plt.subplots(figsize=(7.5, 4.2))
        frame.plot(kind="bar", ax=axis, width=0.78)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Score")
        axis.set_xlabel("Classe")
        axis.set_title("Précision / Rappel / F1 par classe")
        axis.legend(title="Métrique", fontsize=9)
        axis.tick_params(axis="x", rotation=0)
        for container in axis.containers:
            # matplotlib type ``containers`` as ``Container`` while ``bar_label`` requires the
            # ``BarContainer`` produced by ``axis.bar`` : the cast is safe here.
            axis.bar_label(container, fmt="%.2f", fontsize=7, padding=1)  # type: ignore[arg-type]
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ curves ----------
    def roc_curve(self, result: Any, *, name: str = "roc_curve.png") -> Path | None:
        """Plot the ROC curve with the AUC in the legend.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when probabilities are unavailable.
        """
        curve = result.curves.get("roc")
        if not curve:
            logger.warning("ROC curve skipped: no probabilities in the evaluation result")
            return None
        auc = float(result.metrics.get("roc_auc", float("nan")))
        fig, axis = plt.subplots(figsize=(5.6, 5.0))
        axis.plot(curve["fpr"], curve["tpr"], lw=2.2, label=f"ROC AUC = {auc:.4f}", color="#2563eb")
        axis.plot([0, 1], [0, 1], ls="--", lw=1.2, color="grey", label="Hasard (AUC = 0.5)")
        axis.set_xlabel("Taux de faux positifs (FPR)")
        axis.set_ylabel("Taux de vrais positifs (TPR)")
        axis.set_title("Courbe ROC")
        axis.legend(loc="lower right", fontsize=9)
        return _save(fig, self.figures_dir / name)

    def precision_recall_curve(self, result: Any, *, name: str = "precision_recall_curve.png") -> Path | None:
        """Plot the precision-recall curve (more informative than ROC on données déséquilibrées).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when probabilities are unavailable.
        """
        curve = result.curves.get("precision_recall")
        if not curve:
            return None
        pr_auc = float(result.metrics.get("pr_auc", float("nan")))
        fig, axis = plt.subplots(figsize=(5.6, 5.0))
        axis.plot(curve["recall"], curve["precision"], lw=2.2, label=f"PR AUC = {pr_auc:.4f}", color="#16a34a")
        axis.set_xlabel("Rappel")
        axis.set_ylabel("Précision")
        axis.set_title("Courbe Précision-Rappel")
        axis.legend(loc="lower left", fontsize=9)
        return _save(fig, self.figures_dir / name)

    def calibration_curve(self, result: Any, *, name: str = "calibration_curve.png") -> Path | None:
        """Plot predicted probability vs observed frequency (calibration / fiabilité).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the curve is unavailable.
        """
        curve = result.curves.get("calibration")
        if not curve or not curve.get("mean_predicted"):
            return None
        fig, axis = plt.subplots(figsize=(5.6, 5.0))
        axis.plot(
            curve["mean_predicted"],
            curve["fraction_positive"],
            marker="o",
            lw=2.0,
            label="Modèle",
            color="#7c3aed",
        )
        axis.plot([0, 1], [0, 1], ls="--", lw=1.2, color="grey", label="Parfaitement calibré")
        axis.set_xlabel("Probabilité moyenne prédite")
        axis.set_ylabel("Fréquence observée de la classe positive")
        axis.set_title("Courbe de calibration")
        axis.legend(fontsize=9)
        return _save(fig, self.figures_dir / name)

    def score_distribution(self, result: Any, *, name: str = "score_distribution.png") -> Path | None:
        """Plot the distribution of the predicted score per true class.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        curve = result.curves.get("score_distribution")
        if not curve or not curve.get("positive"):
            return None
        fig, axis = plt.subplots(figsize=(6.6, 4.2))
        axis.hist(curve["negative"], bins=30, alpha=0.62, label="Classe 0 (réel)", color="#94a3b8", density=True)
        axis.hist(curve["positive"], bins=30, alpha=0.62, label="Classe 1 (réel)", color="#ef4444", density=True)
        axis.set_xlabel("Probabilité prédite de la classe positive")
        axis.set_ylabel("Densité")
        axis.set_title("Séparation des scores par classe réelle")
        axis.legend(fontsize=9)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ features --------
    def feature_importance(
        self, importance: pd.DataFrame, *, top_n: int = 15, name: str = "feature_importance.png"
    ) -> Path | None:
        """Plot the most important features.

        Args:
            importance: DataFrame ``feature | importance | method``.
            top_n: Number of features displayed.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no importance is available.
        """
        if importance is None or importance.empty:
            logger.warning("Feature importance skipped: the model exposes none")
            return None
        frame = importance.head(top_n).iloc[::-1]
        fig, axis = plt.subplots(figsize=(7.2, 0.36 * len(frame) + 1.6))
        axis.barh(frame["feature"], frame["importance"], color="#0ea5e9")
        axis.set_xlabel(f"Importance ({frame['method'].iloc[0]})")
        axis.set_title(f"Top {len(frame)} features")
        axis.tick_params(axis="y", labelsize=8)
        return _save(fig, self.figures_dir / name)

    def threshold_curve(
        self, thresholds: pd.DataFrame, *, name: str = "threshold_tradeoff.png"
    ) -> Path | None:
        """Plot precision / recall / flagged volume as a function of the decision threshold.

        Args:
            thresholds: DataFrame produced by ``Evaluator.threshold_analysis``.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the analysis is unavailable.
        """
        if thresholds is None or thresholds.empty:
            return None
        fig, left_axis = plt.subplots(figsize=(7.4, 4.4))
        left_axis.plot(thresholds["threshold"], thresholds["precision"], marker="o", label="Précision", color="#2563eb")
        left_axis.plot(thresholds["threshold"], thresholds["recall"], marker="s", label="Rappel", color="#dc2626")
        left_axis.plot(thresholds["threshold"], thresholds["f1"], marker="^", ls="--", label="F1", color="#16a34a")
        left_axis.set_xlabel("Seuil de décision")
        left_axis.set_ylabel("Score")
        left_axis.set_ylim(0.0, 1.05)

        right_axis = left_axis.twinx()
        right_axis.bar(
            thresholds["threshold"], thresholds["flagged_rate"], alpha=0.18, color="#64748b", label="Volume alerté"
        )
        right_axis.set_ylabel("Part de la population alertée")
        right_axis.grid(False)

        left_axis.set_title("Arbitrage précision / rappel / volume selon le seuil")
        lines, labels = left_axis.get_legend_handles_labels()
        extra_lines, extra_labels = right_axis.get_legend_handles_labels()
        left_axis.legend(lines + extra_lines, labels + extra_labels, fontsize=8, loc="center left")
        return _save(fig, self.figures_dir / name)

    def error_breakdown(
        self, predictions: pd.DataFrame, *, group_by: str | None = None, name: str = "error_breakdown.png"
    ) -> Path | None:
        """Plot the error rate, globally or per segment.

        Args:
            predictions: Row-level prediction frame (must contain ``is_error``).
            group_by: Optional categorical column of the prediction frame.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the frame is unusable.
        """
        if predictions.empty or "is_error" not in predictions.columns:
            return None
        if group_by and group_by in predictions.columns:
            frame = (
                predictions.groupby(group_by, observed=True)["is_error"]
                .agg(error_rate="mean", n="size")
                .sort_values("error_rate", ascending=False)
                .reset_index()
            )
            title = f"Taux d'erreur par {group_by}"
            label = str(group_by)
        else:
            frame = pd.DataFrame(
                {
                    "segment": ["erreur (FP+FN)", "correct"],
                    "error_rate": [float(predictions["is_error"].mean()), 1.0 - float(predictions["is_error"].mean())],
                    "n": [int(predictions["is_error"].sum()), int((1 - predictions["is_error"]).sum())],
                }
            )
            title = "Répartition des prédictions"
            label = "segment"

        fig, axis = plt.subplots(figsize=(7.0, 4.0))
        axis.bar(frame[label].astype(str), frame["error_rate"], color="#f97316")
        axis.set_ylabel("Taux")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        for index, row in frame.iterrows():
            axis.text(index, float(row["error_rate"]) + 0.01, f"n={int(row['n'])}", ha="center", fontsize=8)
        return _save(fig, self.figures_dir / name)

    def metrics_bar(
        self, metrics: dict[str, float], *, exclude: Sequence[str] = ("log_loss",), name: str = "metrics_summary.png"
    ) -> Path:
        """Plot the main metrics as a bar chart (used in reports and dashboards).

        Args:
            metrics: Metric name -> value.
            exclude: Metrics to hide (different scale or direction).
            name: Output file name.

        Returns:
            The written path.
        """
        filtered = {key: value for key, value in metrics.items() if key not in exclude and np.isfinite(value)}
        fig, axis = plt.subplots(figsize=(7.4, 3.8))
        axis.bar(list(filtered), list(filtered.values()), color="#0ea5e9")
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title("Métriques de l'évaluation")
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", fontsize=8)  # type: ignore[arg-type]
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ bundle ----------
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Generate every available figure for an evaluation result.

        Args:
            result: Evaluation result.
            thresholds: Optional threshold analysis table.

        Returns:
            Mapping of figure name to written path.
        """
        written: dict[str, Path] = {}
        candidates: list[tuple[str, Path | None]] = [
            ("confusion_matrix", self.confusion_matrix(result)),
            ("per_class_metrics", self.per_class_metrics(result.per_class)),
            ("roc_curve", self.roc_curve(result)),
            ("precision_recall_curve", self.precision_recall_curve(result)),
            ("calibration_curve", self.calibration_curve(result)),
            ("score_distribution", self.score_distribution(result)),
            ("feature_importance", self.feature_importance(result.feature_importance)),
            ("threshold_tradeoff", self.threshold_curve(thresholds) if thresholds is not None else None),
            ("error_breakdown", self.error_breakdown(result.predictions, group_by=_first_group(result.predictions))),
            ("metrics_summary", self.metrics_bar(result.metrics)),
        ]
        for name, path in candidates:
            if path is not None:
                written[name] = path
        logger.info("Figures generated: {}", sorted(written))
        return written


def _first_group(predictions: pd.DataFrame) -> str | None:
    """Pick the first categorical context column available for an error breakdown."""
    for column in ("contract_type", "internet_service", "payment_method", "region"):
        if column in predictions.columns:
            return column
    return None
