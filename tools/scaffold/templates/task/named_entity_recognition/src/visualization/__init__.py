"""Couche de visualisation de la reconnaissance d'entités.

Cinq figures, toutes écrites dans ``artifacts/figures`` par le rapport : la F1 par type, la
répartition des mentions par type et par split, la précision observée par niveau de confiance, le
rappel sur les surfaces réservées et la longueur des messages par style rédactionnel.
"""

from src.visualization.plots import (
    plot_confidence,
    plot_entity_counts,
    plot_holdout,
    plot_lengths,
    plot_per_label,
)

__all__ = [
    "plot_confidence",
    "plot_entity_counts",
    "plot_holdout",
    "plot_lengths",
    "plot_per_label",
]
