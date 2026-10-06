"""Couche d'évaluation de la classification de texte.

* :mod:`src.evaluation.evaluator` — mesure un classifieur ajusté sur un split, texte par texte
  (la latence publiée est donc une latence de service), et ventile les métriques par segment ;
* :mod:`src.evaluation.reports` — rend le rapport Markdown, ses tables CSV et ses figures.

Le split de test n'est mesuré qu'une fois, par le pipeline d'évaluation : c'est ce qui rend les
chiffres du README comparables d'une exécution à l'autre.
"""

from src.evaluation.evaluator import SEGMENT_COLUMNS, ClassificationEvaluator, EvaluationResult
from src.evaluation.reports import ReportBuilder, ReportBundle

__all__ = [
    "SEGMENT_COLUMNS",
    "ClassificationEvaluator",
    "EvaluationResult",
    "ReportBuilder",
    "ReportBundle",
]
