"""Figures de lecture des résultats : par type, par niveau de confiance, par surface réservée.

Quatre figures, et chacune répond à une question qu'un tableau de chiffres ne rend pas lisible :

* :func:`plot_per_label` — quel type est sacrifié ? Une F1 macro moyenne cache toujours un type en
  dessous, et la barre l'affiche avec son support ;
* :func:`plot_entity_counts` — le corpus est-il équilibré ? Un type rare porte une F1 instable, et
la
  figure le dit avant qu'on ne commente le score ;
* :func:`plot_confidence` — la confiance publiée vaut-elle ce qu'elle annonce ? La précision
observée
  par niveau est tracée **avec** la diagonale, donc une confiance systématiquement trop haute se
  voit ;
* :func:`plot_holdout` — qu'a appris le modèle, et qu'a-t-il recopié ? Le rappel sur les surfaces
  réservées est tracé à côté de celui des surfaces déjà vues à l'entraînement.

Toutes les figures sont écrites en PNG, sans écran (backend ``Agg``), et retournent leur chemin :
c'est ce qui permet au constructeur de rapport de les citer dans un tableau markdown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Les figures sont écrites, jamais affichées : le projet tourne sans écran.

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Colour of the project (a single accent keeps the figures readable).
ACCENT = "#1f5f8b"
ALERT = "#c0392b"


def plot_per_label(per_label: pd.DataFrame, path: str | Path) -> Path:
    """Draw the precision / recall / F1 of every entity type.

    Args:
        per_label: Table produced by :func:`~src.training.metrics.per_label_frame`.
        path: Destination PNG.

    Returns:
        The written path.
    """
    frame = per_label[per_label["label"] != "micro"].copy()
    figure, axes = plt.subplots(figsize=(7.2, 0.7 * max(len(frame), 2) + 1.6))
    positions = range(len(frame))
    height = 0.26
    axes.barh(
        [p - height for p in positions],
        frame["precision"],
        height,
        label="Précision",
        color="#9ecae1",
    )
    axes.barh(list(positions), frame["recall"], height, label="Rappel", color="#6baed6")
    axes.barh([p + height for p in positions], frame["f1"], height, label="F1", color=ACCENT)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(
        [f"{row.label} ({int(row.support)})" for row in frame.itertuples(index=False)]
    )
    axes.set_xlim(0.0, 1.05)
    axes.set_xlabel("Score (support entre parenthèses)")
    axes.set_title("Performance par type d'entité")
    axes.legend(loc="lower right", frameon=False)
    axes.grid(axis="x", alpha=0.25)
    return _save(figure, path)


def plot_entity_counts(distribution: pd.DataFrame, path: str | Path) -> Path:
    """Draw the number of mentions per type and per split.

    Args:
        distribution: Table produced by
        :meth:`~src.data.loaders.EntityCorpusLoader.entity_distribution`.
        path: Destination PNG.

    Returns:
        The written path.
    """
    pivot = distribution.pivot_table(
        index="label", columns="split", values="n_mentions", aggfunc="sum", fill_value=0
    )
    order = ["train", "val", "calibration", "test"]
    columns = [name for name in order if name in pivot.columns]
    figure, axes = plt.subplots(figsize=(7.2, 4.0))
    pivot[columns].plot(
        kind="barh", ax=axes, width=0.78, color=["#6baed6", "#9ecae1", "#cfd9e3", ACCENT]
    )
    axes.set_xlabel("Nombre de mentions")
    axes.set_ylabel("")
    axes.set_title("Corpus annoté : mentions par type et par split")
    axes.legend(title="split", frameon=False)
    axes.grid(axis="x", alpha=0.25)
    return _save(figure, path)


def plot_confidence(confidence: pd.DataFrame, path: str | Path) -> Path:
    """Draw the observed precision against the confidence level.

    Args:
        confidence: Table produced by :func:`~src.training.metrics.confidence_table`.
        path: Destination PNG.

    Returns:
        The written path.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    centres = [_centre(bucket) for bucket in confidence["bucket"]]
    axes.plot(
        centres, confidence["precision"], marker="o", color=ACCENT, label="Précision observée"
    )
    axes.plot(
        [0.0, 1.0], [0.0, 1.0], linestyle="--", color="#888888", label="Confiance = précision"
    )
    for centre, row in zip(centres, confidence.itertuples(index=False), strict=True):
        if int(row.n_mentions) == 0:
            axes.annotate("aucune mention", (centre, 0.02), fontsize=7, color=ALERT)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel("Niveau de corroboration")
    axes.set_ylabel("Précision observée")
    axes.set_title("La confiance publiée vaut-elle ce qu'elle annonce ?")
    axes.legend(frameon=False, loc="upper left")
    axes.grid(alpha=0.25)
    return _save(figure, path)


def plot_holdout(holdout: pd.DataFrame, path: str | Path) -> Path:
    """Draw the recall on reserved surfaces against the recall on surfaces seen at training time.

    Args:
        holdout: Table produced by :meth:`~src.evaluation.evaluator.EntityEvaluator._holdout`
            (``group``, ``n_entities``, ``recall``).
        path: Destination PNG.

    Returns:
        The written path.
    """
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    colours = [
        ALERT if row.group == "reservee" else ACCENT for row in holdout.itertuples(index=False)
    ]
    bars = axes.bar(
        [row.group for row in holdout.itertuples(index=False)],
        holdout["recall"],
        color=colours,
        width=0.55,
    )
    for bar, row in zip(bars, holdout.itertuples(index=False), strict=True):
        axes.annotate(
            f"{row.recall:.2f}\n({int(row.n_entities)} mentions)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes.set_ylim(0.0, 1.15)
    axes.set_ylabel("Rappel")
    axes.set_title("Surfaces réservées aux splits d'évaluation ou vues au train")
    axes.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def plot_lengths(documents: pd.DataFrame, path: str | Path, *, column: str = "n_tokens") -> Path:
    """Draw the distribution of message lengths, split by editorial style.

    Args:
        documents: Message table.
        path: Destination PNG.
        column: Length column to plot (``n_tokens`` by default).

    Returns:
        The written path.
    """
    figure, axes = plt.subplots(figsize=(6.8, 3.8))
    styles = sorted({str(value) for value in documents["style"]}) if "style" in documents else [""]
    for position, style in enumerate(styles):
        subset = documents[documents["style"].astype(str) == style] if style else documents
        axes.hist(
            subset[column].astype(float),
            bins=24,
            alpha=0.6,
            label=f"{style} ({len(subset)})" if style else f"messages ({len(subset)})",
            color=[ACCENT, "#9ecae1", "#6baed6"][position % 3],
        )
    axes.set_xlabel("Longueur du message (tokens)")
    axes.set_ylabel("Messages")
    axes.set_title("Longueur des messages par style rédactionnel")
    axes.legend(frameon=False)
    axes.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def _centre(bucket: str) -> float:
    """Return the centre of a bucket label such as ``[0.4; 0.6]``.

    Args:
        bucket: Bucket label produced by the confidence table.

    Returns:
        The centre of the bucket, 0.5 when the label cannot be parsed.
    """
    try:
        low, high = bucket.strip("[]").split(";")
        return (float(low) + float(high)) / 2.0
    except ValueError:  # pragma: no cover - étiquette inattendue
        return 0.5


def _save(figure: plt.Figure, path: str | Path) -> Path:
    """Write a figure as PNG and close it.

    Args:
        figure: Matplotlib figure.
        path: Destination path.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    logger.debug("Figure written: {}", destination)
    return destination


__all__ = [
    "plot_confidence",
    "plot_entity_counts",
    "plot_holdout",
    "plot_lengths",
    "plot_per_label",
]
