"""Figures of the retrieval report.

Five figures carry the whole diagnosis, and each one answers a question an engineer actually
asks:

``recall_curve``      at which cut-off does the model find what it is going to find? and where
                      do the trivial references sit?
``score_separation``  does the retrieval score separate answerable questions from the ones the
                      corpus cannot answer? (if not, no threshold can make abstention work)
``latency``           what does a request cost, and is the tail acceptable?
``segments``          on which difficulty / intent does the system collapse?
``error_profile``     among the retrieved passages, which explicability feature distinguishes a
                      relevant passage from a near miss?

Every function returns the path of the written figure and never raises on missing data: a figure
that cannot be drawn is skipped with a warning, because a report must still be produced when a
segment is empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # rendu hors écran : les figures sont écrites, jamais affichées
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Palette used by every figure (consistent across the repository).
PALETTE = {"model": "#1f77b4", "random": "#d62728", "corpus_order": "#7f7f7f", "accent": "#2ca02c"}

#: Label of every baseline in the figures.
BASELINE_LABELS = {
    "random": "Tirage aléatoire",
    "corpus_order": "Ordre du corpus",
    "popularity": "Popularité",
}


def _finish(figure: plt.Figure, path: Path, *, dpi: int = 130) -> Path:
    """Save a figure and close it.

    Args:
        figure: Figure to persist.
        path: Destination file.
        dpi: Resolution.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    logger.debug("Figure written: {}", path)
    return path


def plot_recall_curve(
    metrics: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
    path: str | Path,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Path | None:
    """Plot recall@k of the model against the trivial references.

    Args:
        metrics: Model metrics (``recall_at_1``, ...).
        baselines: Baselines metrics, keyed by baseline name.
        path: Destination file.
        ks: Cut-offs to plot.

    Returns:
        The written path, or ``None`` when no metric is available.
    """
    series = {
        "Modèle": [metrics.get(f"recall_at_{k}", np.nan) for k in ks],
        **{
            BASELINE_LABELS.get(name, name): [values.get(f"recall_at_{k}", np.nan) for k in ks]
            for name, values in baselines.items()
        },
    }
    if all(np.all(np.isnan(values)) for values in series.values()):
        logger.warning("No recall metric available: skipping the recall curve")
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for label, values in series.items():
        axis.plot(ks, values, marker="o", label=label, color=PALETTE.get(label.lower()))
    axis.set_xlabel("k (nombre de passages retournés)")
    axis.set_ylabel("Recall@k (documents pertinents retrouvés)")
    axis.set_title("Rappel par coupure : le modèle face aux références triviales")
    axis.set_xticks(list(ks))
    axis.grid(alpha=0.3)
    axis.legend()
    return _finish(figure, Path(path))


def plot_score_separation(
    per_question: pd.DataFrame, path: str | Path, *, threshold: float | None = None
) -> Path | None:
    """Plot the retrieval score distribution, answerable versus unanswerable questions.

    Args:
        per_question: Per-question table produced by the evaluator.
        path: Destination file.
        threshold: Abstention threshold, drawn as a vertical line when provided.

    Returns:
        The written path, or ``None`` when the column is missing.
    """
    if "top_score" not in per_question.columns or per_question.empty:
        logger.warning("No score available: skipping the separation figure")
        return None
    frame = per_question.assign(
        groupe=np.where(per_question["answer_type"] == "unanswerable", "Hors corpus", "Répondable")
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    sns.histplot(
        data=frame,
        x="top_score",
        hue="groupe",
        bins=24,
        stat="density",
        common_norm=False,
        alpha=0.55,
        ax=axis,
    )
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", label=f"Seuil ({threshold:.3f})")
        axis.legend()
    axis.set_xlabel("Score du meilleur passage retrouvé")
    axis.set_ylabel("Densité")
    axis.set_title("Séparation des questions répondables et hors corpus")
    return _finish(figure, Path(path))


def plot_latency(per_question: pd.DataFrame, path: str | Path) -> Path | None:
    """Plot the latency distribution with its p50 and p95.

    Args:
        per_question: Per-question table produced by the evaluator.
        path: Destination file.

    Returns:
        The written path, or ``None`` when the column is missing.
    """
    if "latency_ms" not in per_question.columns or per_question.empty:
        logger.warning("No latency available: skipping the latency figure")
        return None
    values = per_question["latency_ms"].astype(float)
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    sns.histplot(values, bins=24, color=PALETTE["model"], ax=axis)
    for quantile, colour in ((0.50, PALETTE["accent"]), (0.95, PALETTE["random"])):
        axis.axvline(
            float(values.quantile(quantile)),
            color=colour,
            linestyle="--",
            label=f"p{int(quantile * 100)} = {values.quantile(quantile):.0f} ms",
        )
    axis.set_xlabel("Latence d'une réponse complète (ms)")
    axis.set_ylabel("Nombre de questions")
    axis.set_title("Latence de bout en bout (retrieval + génération)")
    axis.legend()
    return _finish(figure, Path(path))


def plot_segments(
    segments: Mapping[str, Mapping[str, float]], path: str | Path, *, metric: str = "recall_at_5"
) -> Path | None:
    """Plot one metric per segment value.

    Args:
        segments: Segment metrics, keyed by ``column=value``.
        path: Destination file.
        metric: Metric to plot.

    Returns:
        The written path, or ``None`` when no segment carries the metric.
    """
    rows = [
        {"segment": key, "valeur": values[metric]}
        for key, values in segments.items()
        if metric in values and values[metric] == values[metric]
    ]
    if not rows:
        logger.warning("No segment metric '{}': skipping the segment figure", metric)
        return None
    frame = pd.DataFrame(rows).sort_values("valeur")
    figure, axis = plt.subplots(figsize=(7.8, max(3.0, 0.32 * len(frame))))
    sns.barplot(data=frame, y="segment", x="valeur", color=PALETTE["model"], ax=axis)
    axis.set_xlabel(metric)
    axis.set_ylabel("")
    axis.set_title(f"{metric} par segment")
    axis.grid(axis="x", alpha=0.3)
    return _finish(figure, Path(path))


def plot_error_profile(
    profiles: pd.DataFrame, path: str | Path, *, feature: str = "idf_overlap"
) -> Path | None:
    """Compare the explicability features of the relevant and irrelevant passages.

    Args:
        profiles: One row per retrieved passage, with a ``relevant`` flag and the features
            computed by :class:`src.features.build_features.ChunkFeatureBuilder`.
        path: Destination file.
        feature: Feature to compare.

    Returns:
        The written path, or ``None`` when the data is missing.
    """
    if profiles.empty or feature not in profiles.columns or "relevant" not in profiles.columns:
        logger.warning("No error profile available: skipping the figure")
        return None
    frame = profiles.assign(
        verdict=np.where(profiles["relevant"].astype(bool), "Pertinent", "Non pertinent")
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    sns.boxplot(data=frame, x="verdict", y=feature, ax=axis, palette="Set2")
    axis.set_xlabel("")
    axis.set_ylabel(feature)
    axis.set_title(f"Profil d'erreur : {feature} des passages retrouvés")
    axis.grid(axis="y", alpha=0.3)
    return _finish(figure, Path(path))


__all__ = [
    "BASELINE_LABELS",
    "PALETTE",
    "plot_error_profile",
    "plot_latency",
    "plot_recall_curve",
    "plot_score_separation",
    "plot_segments",
]
