"""Figures de diagnostic du classifieur de texte.

Quatre figures, chacune répondant à une question que les chiffres agrégés ne tranchent pas :

* **matrice de confusion** — quelles classes se confondent, et dans quel sens (un ``produit``
  prédit là où la référence est ``livraison`` ne se répare pas de la même façon que l'inverse) ;
* **F1 par classe** — la classe que le modèle n'apprend pas, avec son effectif : une F1 de 0,50
  sur 30 documents et la même sur 300 ne racontent pas la même histoire ;
* **confiance vs exactitude** — la courbe de calibration : un modèle qui annonce 0,9 et se trompe
  deux fois sur cinq n'est pas utilisable en production, quel que soit son F1 ;
* **longueur des textes** — la distribution des tokens par classe, pour voir si une classe ne se
  distingue que par sa longueur (une fuite que le vocabulaire ne montrerait pas).

Les figures sont écrites dans ``artifacts/figures`` et référencées par le rapport : une figure non
écrite n'est pas référencée, donc jamais un lien mort.
"""

from __future__ import annotations

import itertools

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Colour used for the reference curves of every figure.
ACCENT = "#1f4e79"


def plot_confusion(
    confusion: pd.DataFrame, path: str | Path, *, title: str = "Matrice de confusion (test)"
) -> Path:
    """Draw the confusion matrix as a heat map.

    Args:
        confusion: Confusion matrix (rows = reference, columns = prediction).
        path: Destination PNG file.
        title: Figure title.

    Returns:
        The written path.
    """
    labels = [str(label) for label in confusion.columns]
    matrix = confusion.to_numpy(dtype="float64")
    figure, axis = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.1 * len(labels) + 2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Prédiction du modèle")
    axis.set_ylabel("Catégorie de référence")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value:
                axis.text(
                    column,
                    row,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "black",
                    fontsize=9,
                )
    figure.colorbar(image, ax=axis, shrink=0.8, label="tickets")
    figure.tight_layout()
    return _save(figure, path)


def plot_per_class(per_class: pd.DataFrame, path: str | Path) -> Path:
    """Draw the F1 per class, with the support annotated.

    Args:
        per_class: Table with ``class``, ``f1`` and ``support`` columns.
        path: Destination PNG file.

    Returns:
        The written path.
    """
    frame = per_class.sort_values("f1")
    figure, axis = plt.subplots(figsize=(8, 0.6 * len(frame) + 2.5))
    positions = np.arange(len(frame))
    axis.barh(positions, frame["f1"], color=ACCENT)
    axis.set_yticks(positions, [str(label) for label in frame["class"]])
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("F1 (test)")
    axis.set_title("F1 par catégorie — l'effectif est annoté")
    for position, (score, support) in enumerate(zip(frame["f1"], frame["support"], strict=True)):
        axis.text(
            min(float(score) + 0.02, 0.95),
            position,
            f"{float(score):.2f}  (n={int(support)})",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    return _save(figure, path)


def plot_calibration(predictions: pd.DataFrame, path: str | Path, *, bins: int = 10) -> Path:
    """Draw the reliability curve of the classifier.

    Args:
        predictions: Prediction table with ``confidence`` and ``correct`` columns.
        path: Destination PNG file.
        bins: Number of confidence buckets.

    Returns:
        The written path.
    """
    confidences = predictions["confidence"].to_numpy(dtype="float64")
    correctness = predictions["correct"].to_numpy(dtype="float64")
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres: list[float] = []
    observed: list[float] = []
    for lower, upper in itertools.pairwise(edges):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        centres.append(float(confidences[mask].mean()))
        observed.append(float(correctness[mask].mean()))
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="calibration parfaite")
    axis.plot(centres, observed, marker="o", color=ACCENT, label="observé")
    axis.set_xlabel("Confiance annoncée")
    axis.set_ylabel("Exactitude observée")
    axis.set_title("Calibration — annoncer 0,9 et se tromper 1 fois sur 3")
    axis.legend()
    figure.tight_layout()
    return _save(figure, path)


def plot_lengths(documents: pd.DataFrame, path: str | Path, *, column: str = "n_tokens") -> Path:
    """Draw the length distribution of the documents, per class.

    Args:
        documents: Labelled corpus with the length column.
        path: Destination PNG file.
        column: Length column to plot.

    Returns:
        The written path.
    """
    classes = documents["label"].astype(str)
    order = sorted(classes.unique())
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(
        [documents.loc[classes == label, column].to_numpy(dtype="float64") for label in order],
        tick_labels=order,
        showmeans=True,
    )
    axis.set_ylabel("tokens par ticket")
    axis.set_title("Longueur des textes par catégorie — une classe qui ne se distingue que par là")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    return _save(figure, path)


def _save(figure: "plt.Figure", path: str | Path) -> Path:
    """Write a figure and close it (matplotlib keeps every open figure in memory)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    logger.debug("Figure written: {}", destination)
    return destination


__all__ = ["plot_calibration", "plot_confusion", "plot_lengths", "plot_per_class"]
