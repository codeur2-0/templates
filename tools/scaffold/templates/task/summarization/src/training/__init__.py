"""Entraînement : ajustement, comparaison des algorithmes, métriques de fidélité.

* :mod:`src.training.trainer` — ajuste la stratégie servie, la compare aux algorithmes déclarés sur un
  échantillon déterministe de validation, puis persiste l'artefact, sa fiche de modèle et la
  configuration résolue ;
* :mod:`src.training.metrics` — couverture des faits par type, valeurs non supportées, longueurs,
  latences, lecture du verdict contractuel.
"""

from src.training.metrics import (
    aggregate_coverage,
    coverage_frame,
    coverage_scores,
    length_stats,
    per_strategy_frame,
    segment_frame,
    unsupported_values,
    verdict_from_metrics,
)
from src.training.trainer import (
    COMPARISON_DOCUMENTS,
    SummaryTrainer,
    SummaryTrainingOutcome,
    default_model_file,
)

__all__ = [
    "COMPARISON_DOCUMENTS",
    "SummaryTrainer",
    "SummaryTrainingOutcome",
    "aggregate_coverage",
    "coverage_frame",
    "coverage_scores",
    "default_model_file",
    "length_stats",
    "per_strategy_frame",
    "segment_frame",
    "unsupported_values",
    "verdict_from_metrics",
]
