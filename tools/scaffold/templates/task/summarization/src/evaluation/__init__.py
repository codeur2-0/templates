"""Mesure du résumé : ROUGE, fidélité aux faits, ventilation, rapport.

* :mod:`src.evaluation.rouge` — ROUGE-1/2/L sur la meilleure des deux références de chaque document ;
* :mod:`src.evaluation.evaluator` — mesure une stratégie sur un split, compare les stratégies sur les
  **mêmes lignes**, ventile par segment et détecte les valeurs non supportées ;
* :mod:`src.evaluation.reports` — rend le rapport Markdown, ses tables CSV et ses figures.

Le split de test n'est mesuré qu'une fois, par le pipeline d'évaluation : c'est ce qui rend les
chiffres du README comparables d'une exécution à l'autre.
"""

from src.evaluation.evaluator import SEGMENT_COLUMNS, SummaryEvaluation, SummaryEvaluator
from src.evaluation.reports import ReportBuilder, ReportBundle, report_sections
from src.evaluation.rouge import (
    DEFAULT_ORDERS,
    RougeScore,
    RougeScores,
    corpus_rouge,
    rouge_n,
    rouge_scores,
    tokenize_for_rouge,
)

__all__ = [
    "DEFAULT_ORDERS",
    "SEGMENT_COLUMNS",
    "ReportBuilder",
    "ReportBundle",
    "RougeScore",
    "RougeScores",
    "SummaryEvaluation",
    "SummaryEvaluator",
    "corpus_rouge",
    "report_sections",
    "rouge_n",
    "rouge_scores",
    "tokenize_for_rouge",
]
