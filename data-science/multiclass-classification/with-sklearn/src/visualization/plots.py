"""Figures used by the multiclass evaluation report and the notebooks.

Every function returns the path of the written PNG: figures are artefacts, not side effects.
The rendering backend is forced to ``Agg`` so that the code works headless (CI, containers).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_DPI = 140

#: Colour of the model in every comparison figure (references are drawn in grey).
MODEL_COLOUR = "#005f73"


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


class MulticlassPlots:
    """Figure factory for a multiclass (diagnosis) evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = "viridis") -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Seaborn/matplotlib palette for heatmaps.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette

    # ------------------------------------------------------------------ confusion -------
    def confusion_matrix(
        self, result: Any, *, normalize: bool = True, name: str = "confusion_matrix.png"
    ) -> Path | None:
        """Plot the confusion matrix, rows normalised by the true class.

        A class without any row in the split keeps a row of zeros instead of dividing by zero.

        Args:
            result: :class:`src.evaluation.evaluator.EvaluationResult`.
            normalize: Show row shares instead of raw counts.
            name: Output file name.

        Returns:
            The written path, or ``None`` without confusion matrix.
        """
        if result.confusion_matrix is None:
            return None
        counts = np.asarray(result.confusion_matrix, dtype="float64")
        support = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, support, out=np.zeros_like(counts), where=support > 0)
        values = shares if normalize else counts
        size = max(5.2, 0.9 * len(result.labels) + 2.0)
        fig, axis = plt.subplots(figsize=(size + 1.0, size))
        sns.heatmap(
            values,
            annot=np.round(values, 2) if normalize else counts.astype(int),
            fmt=".2f" if normalize else "d",
            cmap=self.palette,
            xticklabels=result.labels,
            yticklabels=result.labels,
            vmin=0.0,
            vmax=1.0 if normalize else None,
            cbar=True,
            linewidths=0.6,
            ax=axis,
        )
        axis.set_xlabel("Mode prédit")
        axis.set_ylabel("Mode réel")
        mode = "part de la classe réelle" if normalize else "effectifs"
        axis.set_title(f"Matrice de confusion ({mode}) — split {result.split}")
        axis.tick_params(axis="x", rotation=30)
        return _save(fig, self.figures_dir / name)

    def per_class_metrics(self, result: Any, *, name: str = "per_class_metrics.png") -> Path | None:
        """Plot precision / recall / F1 per class; structural classes are flagged.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without per-class table.
        """
        per_class = result.per_class
        if per_class.empty:
            return None
        frame = per_class.set_index("class")[["precision", "recall", "f1"]]
        fig, axis = plt.subplots(figsize=(max(7.5, 1.3 * len(frame)), 4.2))
        frame.plot(kind="bar", ax=axis, width=0.8, color=["#94d2bd", "#0a9396", MODEL_COLOUR])
        axis.set_ylim(0.0, 1.08)
        axis.set_ylabel("Score")
        axis.set_xlabel("")
        axis.set_title("Précision / rappel / F1 par mode")
        axis.legend(fontsize=8, loc="upper right")
        axis.tick_params(axis="x", rotation=20)
        structural = (
            set(per_class.loc[per_class["structural"], "class"])
            if "structural" in per_class
            else set()
        )
        for position, label in enumerate(frame.index):
            if label in structural:
                axis.text(position, 1.02, "sans signal", ha="center", fontsize=7, color="#9b2226")
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ references ------
    def references(self, result: Any, *, name: str = "references.png") -> Path | None:
        """Plot the macro-F1 of the model against the floor, the current rule and the ceiling.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without references.
        """
        frame = result.references
        if frame.empty:
            return None
        frame = frame.dropna(subset=["f1_macro"])
        colours = [MODEL_COLOUR if label == "modèle" else "#adb5bd" for label in frame["référence"]]
        fig, axis = plt.subplots(figsize=(7.8, 0.55 * len(frame) + 1.8))
        axis.barh(frame["référence"][::-1], frame["f1_macro"][::-1], color=colours[::-1])
        for position, value in enumerate(frame["f1_macro"][::-1]):
            axis.text(float(value) + 0.01, position, f"{float(value):.3f}", va="center", fontsize=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("macro-F1 (test)")
        axis.set_title("Le modèle face au plancher, à la règle actuelle et au plafond")
        return _save(fig, self.figures_dir / name)

    def roc_ovr(self, result: Any, *, name: str = "roc_one_vs_rest.png") -> Path | None:
        """Plot one ROC curve per class (one-vs-rest), AUC in the legend.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without curves.
        """
        curves = (result.curves or {}).get("roc_ovr") or {}
        if not curves:
            return None
        fig, axis = plt.subplots(figsize=(6.4, 5.2))
        for label, points in curves.items():
            axis.plot(
                points["fpr"],
                points["tpr"],
                linewidth=1.5,
                label=f"{label} (AUC={points['auc']:.3f})",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#adb5bd", linewidth=1.0)
        axis.set_xlabel("Taux de faux positifs")
        axis.set_ylabel("Taux de vrais positifs")
        axis.set_title("Courbes ROC un-contre-tous")
        axis.legend(fontsize=7, loc="lower right")
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ calibration -----
    def reliability(self, result: Any, *, name: str = "reliability.png") -> Path | None:
        """Plot the top-label reliability diagram (confidence versus observed accuracy).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without calibration table.
        """
        table = result.calibration
        if table.empty:
            return None
        fig, axis = plt.subplots(figsize=(5.8, 5.0))
        axis.plot([0, 1], [0, 1], linestyle="--", color="#adb5bd", label="calibration parfaite")
        axis.plot(
            table["confidence"], table["accuracy"], marker="o", color=MODEL_COLOUR, label="modèle"
        )
        for _, row in table.iterrows():
            axis.annotate(
                str(int(row["count"])),
                (row["confidence"], row["accuracy"]),
                fontsize=7,
                xytext=(3, -9),
                textcoords="offset points",
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Confiance moyenne du mode retenu")
        axis.set_ylabel("Exactitude observée")
        axis.set_title(f"Diagramme de fiabilité — ECE = {result.ece:.3f}")
        axis.legend(fontsize=8, loc="upper left")
        return _save(fig, self.figures_dir / name)

    def confidence_histogram(
        self, result: Any, *, name: str = "confidence_distribution.png"
    ) -> Path | None:
        """Plot the confidence of correct versus wrong predictions, with the review threshold.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without predictions.
        """
        frame = result.predictions
        if frame.empty or "confidence" not in frame:
            return None
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        bins = np.linspace(0.0, 1.0, 26).tolist()
        axis.hist(
            frame.loc[frame["is_error"] == 0, "confidence"],
            bins=bins,
            alpha=0.75,
            color=MODEL_COLOUR,
            label="diagnostic correct",
        )
        axis.hist(
            frame.loc[frame["is_error"] == 1, "confidence"],
            bins=bins,
            alpha=0.75,
            color="#ee9b00",
            label="diagnostic erroné",
        )
        threshold = float(result.extras.get("review_threshold", float("nan")))
        if np.isfinite(threshold):
            axis.axvline(
                threshold,
                color="#9b2226",
                linestyle="--",
                label=f"seuil de revue ({threshold:.2f})",
            )
        axis.set_xlabel("Confiance du mode retenu")
        axis.set_ylabel("Alarmes")
        axis.set_title("Confiance des diagnostics corrects et erronés")
        axis.legend(fontsize=8)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ decision --------
    def abstention(self, result: Any, *, name: str = "abstention_tradeoff.png") -> Path | None:
        """Plot coverage, automated accuracy and cost per alarm against the review threshold.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without abstention table.
        """
        table = result.abstention
        if table.empty:
            return None
        fig, left = plt.subplots(figsize=(7.6, 4.0))
        left.plot(
            table["threshold"],
            table["coverage"],
            marker="o",
            color="#0a9396",
            label="couverture automatisée",
        )
        left.plot(
            table["threshold"],
            table["accuracy_automated"],
            marker="s",
            color=MODEL_COLOUR,
            label="exactitude automatisée",
        )
        left.set_ylim(0.0, 1.05)
        left.set_xlabel("Seuil de confiance sous lequel l'alarme part en revue")
        left.set_ylabel("Part")
        right = left.twinx()
        right.plot(
            table["threshold"],
            table["cost_per_alarm"],
            marker="^",
            color="#ee9b00",
            label="coût par alarme (EUR)",
        )
        right.set_ylabel("EUR par alarme")
        right.grid(False)
        threshold = float(result.extras.get("review_threshold", float("nan")))
        if np.isfinite(threshold):
            left.axvline(threshold, color="#9b2226", linestyle="--", linewidth=1.0)
        handles, labels = left.get_legend_handles_labels()
        extra_handles, extra_labels = right.get_legend_handles_labels()
        left.legend(handles + extra_handles, labels + extra_labels, fontsize=8, loc="lower left")
        left.set_title("Revue experte : couverture, exactitude et coût")
        return _save(fig, self.figures_dir / name)

    def decision_costs(self, result: Any, *, name: str = "decision_costs.png") -> Path | None:
        """Plot the cost per alarm of each decision policy.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` without decision table.
        """
        table = result.decision
        if table.empty:
            return None
        fig, axis = plt.subplots(figsize=(8.0, 0.6 * len(table) + 1.6))
        colours = [
            "#adb5bd" if "routage" in str(policy) else MODEL_COLOUR for policy in table["politique"]
        ]
        axis.barh(table["politique"][::-1], table["cost_per_alarm"][::-1], color=colours[::-1])
        for position, value in enumerate(table["cost_per_alarm"][::-1]):
            axis.text(float(value), position, f" {float(value):.0f} EUR", va="center", fontsize=8)
        axis.set_xlabel("Coût moyen par alarme (EUR)")
        axis.set_title("Coût des politiques de décision sur les mêmes alarmes")
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ misc ------------
    def feature_importance(
        self, importance: pd.DataFrame, *, top: int = 15, name: str = "feature_importance.png"
    ) -> Path | None:
        """Plot the top features.

        Args:
            importance: ``feature | importance | method`` frame.
            top: Number of features shown.
            name: Output file name.

        Returns:
            The written path, or ``None`` without importance.
        """
        if importance is None or importance.empty:
            return None
        frame = importance.head(top).iloc[::-1]
        fig, axis = plt.subplots(figsize=(7.6, 0.38 * len(frame) + 1.4))
        axis.barh(frame["feature"], frame["importance"], color="#0a9396")
        method = str(frame["method"].iloc[0]) if "method" in frame else "native"
        axis.set_xlabel(f"importance ({method})")
        axis.set_title(f"Top {len(frame)} des features")
        return _save(fig, self.figures_dir / name)

    def metrics_bar(
        self,
        metrics: dict[str, float],
        *,
        exclude: Sequence[str] = ("log_loss",),
        name: str = "metrics_summary.png",
    ) -> Path:
        """Plot the main metrics; the axis extends below zero when a metric is negative (MCC).

        Args:
            metrics: Metric name -> value.
            exclude: Metrics hidden (different scale or direction).
            name: Output file name.

        Returns:
            The written path.
        """
        filtered = {
            key: value
            for key, value in metrics.items()
            if key not in exclude and np.isfinite(value)
        }
        fig, axis = plt.subplots(figsize=(8.0, 3.8))
        axis.bar(list(filtered), list(filtered.values()), color=MODEL_COLOUR)
        lowest = min([0.0, *filtered.values()])
        axis.set_ylim(lowest - (0.05 if lowest < 0 else 0.0), 1.05)
        axis.axhline(0.0, color="#495057", linewidth=0.8)
        axis.set_ylabel("Score")
        axis.set_title("Métriques de l'évaluation")
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", fontsize=8)  # type: ignore[arg-type]
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ bundle ----------
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Generate every available figure for an evaluation result.

        Args:
            result: Evaluation result.
            thresholds: Unused in multiclass (no binary threshold); kept for API symmetry.

        Returns:
            Mapping of figure name to written path.
        """
        del thresholds
        candidates: list[tuple[str, Path | None]] = [
            ("confusion_matrix", self.confusion_matrix(result)),
            ("per_class_metrics", self.per_class_metrics(result)),
            ("references", self.references(result)),
            ("roc_one_vs_rest", self.roc_ovr(result)),
            ("reliability", self.reliability(result)),
            ("confidence_distribution", self.confidence_histogram(result)),
            ("abstention_tradeoff", self.abstention(result)),
            ("decision_costs", self.decision_costs(result)),
            ("feature_importance", self.feature_importance(result.feature_importance)),
            ("metrics_summary", self.metrics_bar(result.metrics)),
        ]
        written = {name: path for name, path in candidates if path is not None}
        logger.info("Figures generated: {}", sorted(written))
        return written
