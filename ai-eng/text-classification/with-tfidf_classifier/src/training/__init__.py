"""Couche d'entraînement de la classification de texte : métriques, boucle, seuils.

Deux responsabilités, séparées pour rester testables indépendamment :

* :mod:`src.training.metrics` — le registre des métriques et les tables dérivées (par classe,
  matrice de confusion, références triviales). Aucune dépendance aux modèles ;
* :mod:`src.training.trainer` — la boucle : découpage par colonne ``split``, ajustement,
  mesure sur la validation, callbacks, archivage des erreurs les plus confiantes.

Le socle fournit déjà :class:`~src.training.callbacks.BaseCallback` et
:class:`~src.training.callbacks.CallbackContext` ; le trainer les pilote sans les connaître.
"""

from src.training.metrics import (
    CLASSIFICATION_METRICS,
    METRIC_DESCRIPTIONS,
    METRICS_BY_TASK,
    classification_metrics,
    confusion_matrix_frame,
    describe_metrics,
    expected_calibration_error,
    flatten_metrics,
    majority_baseline,
    metrics_for_task,
    per_class_frame,
    percentile,
    stratified_baseline,
)
from src.training.trainer import ClassificationOutcome, ClassificationTrainer

__all__ = [
    "CLASSIFICATION_METRICS",
    "METRICS_BY_TASK",
    "METRIC_DESCRIPTIONS",
    "ClassificationOutcome",
    "ClassificationTrainer",
    "classification_metrics",
    "confusion_matrix_frame",
    "describe_metrics",
    "expected_calibration_error",
    "flatten_metrics",
    "majority_baseline",
    "metrics_for_task",
    "per_class_frame",
    "percentile",
    "stratified_baseline",
]
