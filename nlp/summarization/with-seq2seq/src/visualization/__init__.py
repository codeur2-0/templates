"""Figures du résumé : ROUGE par stratégie, couverture par type, coûts de sortie.

Cinq figures, toutes dessinées à partir des tables publiées par l'évaluation ; une figure impossible
(table vide) est ignorée plutôt que remplacée par un graphique trompeur.
"""

from src.visualization.plots import (
    STRATEGY_COLOURS,
    figure_names,
    plot_all,
    plot_compression_vs_coverage,
    plot_coverage_by_type,
    plot_output_costs,
    plot_segment_metrics,
    plot_strategy_rouge,
)

__all__ = [
    "STRATEGY_COLOURS",
    "figure_names",
    "plot_all",
    "plot_compression_vs_coverage",
    "plot_coverage_by_type",
    "plot_output_costs",
    "plot_segment_metrics",
    "plot_strategy_rouge",
]
