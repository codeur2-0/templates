"""Evaluation layer of the retrieval task."""

from src.evaluation.evaluator import EvaluationResult, Evaluator, query_relevance
from src.evaluation.reports import ReportBuilder, ReportBundle

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "ReportBuilder",
    "ReportBundle",
    "query_relevance",
]
