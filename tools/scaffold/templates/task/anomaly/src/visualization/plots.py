"""Figures used by the anomaly-detection report and the notebooks.

Every function returns the path of the written PNG: figures are artefacts, not side effects.
The rendering backend is forced to ``Agg`` so that the code works headless (CI, containers).

The figure set answers the four questions a fraud lead asks about a detector:

1. **le score ordonne-t-il la fraude ?** — courbe précision-rappel (avec le plancher de
   prévalence) et courbe ROC ;
2. **que vaut-il au budget réel ?** — table d'arbitrage volume d'alertes → rappel/précision/lift,
   avec le budget retenu marqué ;
3. **où se concentre la fraude ?** — lift par décile de score et distribution des scores par
   population (fraude vs légitime) ;
4. **quels schémas sont couverts ?** — rappel par mode opératoire, le diagnostic qui évite
   qu'un bon score global masque un schéma en croissance non détecté.
"""

from __future__ import annotations

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

#: Couleur du détecteur (courbes principales).
MODEL_COLOR = "#0a9396"

#: Couleur du plancher aléatoire (prévalence) — la référence à battre.
BASELINE_COLOR = "#ae2012"

#: Couleur du budget retenu (ligne verticale des figures d'arbitrage).
BUDGET_COLOR = "#ee9b00"

#: Palette des modes opératoires de fraude.
SCHEME_PALETTE = "Set2"

#: Libellés lisibles des modes opératoires (le générateur produit des identifiants techniques).
SCHEME_LABELS: dict[str, str] = {
    "card_not_present": "Card not present",
    "account_takeover": "Prise de compte",
    "synthetic_identity": "Identité synthétique",
    "friendly_fraud": "Fraude amicale",
    "legitimate": "Légitime",
}


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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Coerce ``value`` to ``float`` without raising."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _curves(result: Any) -> dict[str, Any]:
    """Return the ``curves`` mapping of an evaluation result (empty when absent)."""
    curves = getattr(result, "curves", None)
    return dict(curves or {})


def _extras(result: Any) -> dict[str, Any]:
    """Return the ``extras`` mapping of an evaluation result (empty when absent)."""
    extras = getattr(result, "extras", None)
    return dict(extras or {})


def _metrics(result: Any) -> dict[str, float]:
    """Return the metric mapping of an evaluation result."""
    metrics = dict(getattr(result, "metrics", {}) or {})
    return {str(key): _safe_float(value) for key, value in metrics.items()}


def _frame(result: Any) -> pd.DataFrame:
    """Return the row-level prediction frame."""
    frame = getattr(result, "predictions", None)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


class AnomalyPlots:
    """Render the figures of an anomaly-detection evaluation."""

    def __init__(self, figures_dir: str | Path, *, dpi: int = DEFAULT_DPI) -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory where PNG files are written.
            dpi: Rendering resolution.
        """
        self.figures_dir = Path(figures_dir)
        self.dpi = int(dpi)

    # ------------------------------------------------------------------ courbes -----------
    def pr_curve(self, result: Any, *, name: str = "pr_curve.png") -> Path | None:
        """Precision-recall curve, with the prevalence floor and the operating point.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no curve point is available.
        """
        points = _curves(result).get("precision_recall") or []
        if not points:
            return None
        frame = pd.DataFrame(points)
        prevalence = _safe_float(_extras(result).get("prevalence"))
        budget_rate = _safe_float(_extras(result).get("budget_rate"), 0.02)

        fig, axis = plt.subplots(figsize=(6.4, 4.4))
        axis.plot(frame["recall"], frame["precision"], color=MODEL_COLOR, lw=2.0, label="Détecteur")
        if np.isfinite(prevalence):
            axis.axhline(
                prevalence,
                color=BASELINE_COLOR,
                ls="--",
                lw=1.4,
                label=f"Plancher aléatoire (prévalence = {prevalence:.3f})",
            )
        # Le point de fonctionnement réel : rappel et précision au budget d'investigation.
        metrics = _metrics(result)
        recall = _safe_float(metrics.get("recall_at_budget"))
        precision = _safe_float(metrics.get("precision_at_budget"))
        if np.isfinite(recall) and np.isfinite(precision):
            axis.scatter(
                [recall],
                [precision],
                color=BUDGET_COLOR,
                s=70,
                zorder=5,
                label=f"Budget {budget_rate:.0%} (rappel {recall:.2f})",
            )
        metrics = _metrics(result)
        pr_auc = _safe_float(metrics.get("pr_auc"))
        axis.set_xlabel("Rappel")
        axis.set_ylabel("Précision")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Courbe précision-rappel (PR AUC = {pr_auc:.3f})")
        axis.legend(loc="upper right", fontsize=8)
        return _save(fig, self.figures_dir / name)

    def roc_curve(self, result: Any, *, name: str = "roc_curve.png") -> Path | None:
        """ROC curve with the diagonal reference.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no curve point is available.
        """
        points = _curves(result).get("roc") or []
        if not points:
            return None
        frame = pd.DataFrame(points)
        roc_auc = _safe_float(_metrics(result).get("roc_auc"))
        fig, axis = plt.subplots(figsize=(6.0, 4.6))
        axis.plot(
            frame["fpr"],
            frame["tpr"],
            color=MODEL_COLOR,
            lw=2.0,
            label=f"Détecteur (AUC = {roc_auc:.3f})",
        )
        axis.plot(
            [0, 1], [0, 1], color=BASELINE_COLOR, ls="--", lw=1.2, label="Aléatoire (AUC = 0,500)"
        )
        axis.set_xlabel("Taux de faux positifs")
        axis.set_ylabel("Taux de vrais positifs")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Courbe ROC")
        axis.legend(loc="lower right", fontsize=8)
        return _save(fig, self.figures_dir / name)

    def score_distribution(
        self, result: Any, *, name: str = "score_distribution.png"
    ) -> Path | None:
        """Score histograms for frauds and legitimate transactions, with the alert threshold.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the distributions are empty.
        """
        distributions = _curves(result).get("score_distribution") or {}
        fraud = pd.DataFrame(distributions.get("fraud") or [])
        legitimate = pd.DataFrame(distributions.get("legitimate") or [])
        if fraud.empty and legitimate.empty:
            return None
        threshold = _safe_float(_extras(result).get("threshold"))

        fig, axis = plt.subplots(figsize=(7.4, 4.2))
        if not legitimate.empty:
            axis.plot(
                legitimate["bin_center"],
                legitimate["density"],
                color="#94a3b8",
                lw=2.0,
                label="Transactions légitimes",
            )
            axis.fill_between(
                legitimate["bin_center"], legitimate["density"], alpha=0.25, color="#94a3b8"
            )
        if not fraud.empty:
            axis.plot(
                fraud["bin_center"],
                fraud["density"],
                color=BASELINE_COLOR,
                lw=2.0,
                label="Fraudes confirmées",
            )
            axis.fill_between(
                fraud["bin_center"], fraud["density"], alpha=0.25, color=BASELINE_COLOR
            )
        if np.isfinite(threshold):
            budget = _safe_float(_extras(result).get("budget_rate"), 0.02)
            axis.axvline(
                threshold,
                color=BUDGET_COLOR,
                ls="--",
                lw=1.6,
                label=f"Seuil d'alerte (budget {budget:.0%})",
            )
        axis.set_yscale("log")
        axis.set_xlabel("Score d'anomalie")
        axis.set_ylabel("Densité (échelle log)")
        axis.set_title("Distribution des scores : la séparation fait la performance")
        axis.legend(fontsize=8)
        return _save(fig, self.figures_dir / name)

    def lift_by_decile(self, result: Any, *, name: str = "lift_by_decile.png") -> Path | None:
        """Fraud rate and lift per score decile, most anomalous first.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the lift table is empty.
        """
        rows = _curves(result).get("lift_by_decile") or []
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        fig, axis = plt.subplots(figsize=(7.0, 4.2))
        colors = [BUDGET_COLOR if int(row) >= 8 else MODEL_COLOR for row in frame["decile"]]
        axis.bar(range(len(frame)), frame["lift"], color=colors)
        axis.axhline(1.0, color=BASELINE_COLOR, ls="--", lw=1.2, label="Lift = 1 (aléatoire)")
        axis.set_xticks(range(len(frame)))
        axis.set_xticklabels([f"D{int(row)}" for row in frame["decile"]], fontsize=8)
        axis.set_xlabel("Décile de score (D9 = les 10 % les plus anormaux)")
        axis.set_ylabel("Lift sur la prévalence")
        axis.set_title("Concentration de la fraude par décile de score")
        axis.legend(fontsize=8)
        for index, value in enumerate(frame["lift"]):
            axis.text(
                index, float(value), f"{float(value):.0f}x", ha="center", va="bottom", fontsize=7
            )
        return _save(fig, self.figures_dir / name)

    def budget_tradeoff(self, result: Any, *, name: str = "budget_tradeoff.png") -> Path | None:
        """Recall and precision as a function of the investigation budget.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the trade-off table is empty.
        """
        rows = _curves(result).get("budget_tradeoff") or []
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        budget_rate = _safe_float(_extras(result).get("budget_rate"), 0.02)

        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.plot(
            frame["budget_rate"] * 100,
            frame["recall"],
            color=MODEL_COLOR,
            lw=2.0,
            marker="o",
            ms=4,
            label="Rappel (fraude capturée)",
        )
        axis.plot(
            frame["budget_rate"] * 100,
            frame["precision"],
            color=BASELINE_COLOR,
            lw=2.0,
            marker="s",
            ms=4,
            label="Précision (part de vraies fraudes)",
        )
        second = axis.twinx()
        second.plot(
            frame["budget_rate"] * 100,
            frame["lift"],
            color="#005f73",
            lw=1.4,
            ls=":",
            label="Lift (échelle de droite)",
        )
        second.set_ylabel("Lift", color="#005f73")
        second.grid(False)
        axis.axvline(
            budget_rate * 100,
            color=BUDGET_COLOR,
            ls="--",
            lw=1.8,
            label=f"Budget retenu ({budget_rate:.0%})",
        )
        axis.set_xlabel("Volume d'alertes (% du flux)")
        axis.set_ylabel("Rappel / précision")
        axis.set_ylim(0.0, 1.02)
        axis.set_title("Arbitrage capacité d'analyse ↔ fraude capturée")
        lines, labels = axis.get_legend_handles_labels()
        extra_lines, extra_labels = second.get_legend_handles_labels()
        axis.legend(lines + extra_lines, labels + extra_labels, fontsize=8, loc="center right")
        return _save(fig, self.figures_dir / name)

    def scheme_recall(self, result: Any, *, name: str = "scheme_recall.png") -> Path | None:
        """Recall at budget per fraud scheme (coverage of each modus operandi).

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no scheme breakdown exists.
        """
        table = getattr(result, "per_scheme", None)
        if not isinstance(table, pd.DataFrame) or table.empty:
            return None
        frame = table.sort_values("recall_at_budget", ascending=True)
        labels = [SCHEME_LABELS.get(str(value), str(value)) for value in frame["fraud_scheme"]]

        fig, axis = plt.subplots(figsize=(7.0, 3.8))
        palette = sns.color_palette(SCHEME_PALETTE, n_colors=len(frame))
        bars = axis.barh(labels, frame["recall_at_budget"], color=palette)
        for bar, count in zip(bars, frame["frauds"], strict=True):
            axis.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{float(bar.get_width()):.2f}  (n={int(count)})",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(0.0, 1.15)
        axis.set_xlabel("Rappel au budget")
        axis.set_title("Couverture par mode opératoire")
        return _save(fig, self.figures_dir / name)

    def error_quadrant(self, result: Any, *, name: str = "error_quadrant.png") -> Path | None:
        """Scatter the scored rows by outcome (TP / FP / FN / TN) against their rank.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the prediction frame is empty.
        """
        frame = _frame(result)
        if frame.empty or "outcome" not in frame.columns:
            return None
        sample = (
            frame.sample(n=min(len(frame), 3000), random_state=0) if len(frame) > 3000 else frame
        )
        colors = {
            "true_positive": "#0a9396",
            "false_positive": "#ee9b00",
            "false_negative": "#ae2012",
            "true_negative": "#cbd5e1",
        }
        fig, axis = plt.subplots(figsize=(7.0, 4.2))
        for outcome, color in colors.items():
            subset = sample[sample["outcome"] == outcome]
            if subset.empty:
                continue
            axis.scatter(
                subset["rank"] / max(len(frame), 1),
                subset["score"],
                s=9 if outcome == "true_negative" else 22,
                color=color,
                alpha=0.85 if outcome != "true_negative" else 0.35,
                label=outcome.replace("_", " "),
            )
        axis.set_yscale("symlog")
        axis.set_xlabel("Rang normalisé (0 = score le plus élevé)")
        axis.set_ylabel("Score d'anomalie (échelle symlog)")
        axis.set_title("Lecture des erreurs : fraudes manquées et fausses alertes")
        axis.legend(fontsize=8, markerscale=1.6)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ batch -------------
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Render every available figure and return their paths.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            thresholds: Accepted for API symmetry with the classification figures; the anomaly
                threshold is carried by ``result.extras`` and already drawn on the figures.

        Returns:
            Mapping of figure name to written path (unrenderable figures are absent).
        """
        del thresholds  # le seuil d'alerte est porté par `extras`, pas par une table séparée
        candidates = (
            ("pr_curve", self.pr_curve),
            ("roc_curve", self.roc_curve),
            ("score_distribution", self.score_distribution),
            ("lift_by_decile", self.lift_by_decile),
            ("budget_tradeoff", self.budget_tradeoff),
            ("scheme_recall", self.scheme_recall),
            ("error_quadrant", self.error_quadrant),
        )
        written: dict[str, Path] = {}
        for figure_name, renderer in candidates:
            try:
                path = renderer(result)
            except (ValueError, KeyError, TypeError) as error:  # données insuffisantes
                logger.warning("Figure '{}' non générée : {}", figure_name, error)
                continue
            if path is not None:
                written[figure_name] = path
        logger.info("Figures générées : {}", sorted(written))
        return written


__all__ = ["AnomalyPlots"]
