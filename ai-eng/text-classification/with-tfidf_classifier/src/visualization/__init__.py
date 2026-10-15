"""Couche de visualisation de la classification de texte.

Quatre figures, toutes écrites dans ``artifacts/figures`` par le rapport : la matrice de
confusion, la F1 par classe, la courbe de calibration et la distribution des longueurs.
"""

from src.visualization.plots import plot_calibration, plot_confusion, plot_lengths, plot_per_class

__all__ = ["plot_calibration", "plot_confusion", "plot_lengths", "plot_per_class"]
