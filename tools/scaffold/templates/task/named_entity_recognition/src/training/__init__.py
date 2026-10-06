"""Couche d'entraînement de la reconnaissance d'entités : métriques et boucle.

Deux responsabilités, séparées pour rester testables indépendamment :

* :mod:`src.training.metrics` — les métriques **au niveau entité** (un span est correct s'il a le
bon
  type *et* les bonnes bornes), la F1 partielle qui mesure le coût de cette exigence, la lecture
  par
  type, les erreurs classées et la table de précision par niveau de confiance. Aucune dépendance
  aux
  modèles, donc testable sur des tuples ;
* :mod:`src.training.trainer` — la boucle : découpage par colonne ``split``, ajustement de
  l'extracteur sur les annotations du train, mesure sur la validation, callbacks, latence mesurée
  message par message (c'est la latence du service, pas celle d'un lot).

Le socle fournit déjà :class:`~src.training.callbacks.BaseCallback` et
:class:`~src.training.callbacks.CallbackContext` ; le trainer les pilote sans les connaître.
"""

from src.training.metrics import (
    Span,
    confidence_gap,
    confidence_table,
    counts_by_label,
    describe_metrics,
    entity_scores,
    error_frame,
    flatten_metrics,
    latency_stats,
    partial_scores,
    per_label_frame,
    percentile,
    precision_recall_f1,
    span_sets,
    trivial_floor,
)
from src.training.trainer import EntityTrainer, EntityTrainingOutcome, score_arrays

__all__ = [
    "EntityTrainer",
    "EntityTrainingOutcome",
    "Span",
    "confidence_gap",
    "confidence_table",
    "counts_by_label",
    "describe_metrics",
    "entity_scores",
    "error_frame",
    "flatten_metrics",
    "latency_stats",
    "partial_scores",
    "per_label_frame",
    "percentile",
    "precision_recall_f1",
    "score_arrays",
    "span_sets",
    "trivial_floor",
]
