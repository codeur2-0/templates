"""Couche d'évaluation de la reconnaissance d'entités.

* :mod:`src.evaluation.evaluator` — mesure un extracteur ajusté sur un split, message par message
  (la latence publiée est donc une latence de service), ventile les métriques par type et par
  segment, compare les **surfaces réservées** aux surfaces vues à l'entraînement et mesure deux
  références sur les mêmes lignes : le plancher trivial et la couche de règles ;
* :mod:`src.evaluation.reports` — rend le rapport Markdown, ses tables CSV et ses figures.

Le split de test n'est mesuré qu'une fois, par le pipeline d'évaluation : c'est ce qui rend les
chiffres du README comparables d'une exécution à l'autre.
"""

from src.evaluation.evaluator import SEGMENT_COLUMNS, EntityEvaluator, EvaluationResult
from src.evaluation.reports import ERROR_KINDS, MAIN_METRICS, ReportBuilder, ReportBundle

__all__ = [
    "ERROR_KINDS",
    "MAIN_METRICS",
    "SEGMENT_COLUMNS",
    "EntityEvaluator",
    "EvaluationResult",
    "ReportBuilder",
    "ReportBundle",
]
