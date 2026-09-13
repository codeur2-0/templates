"""Couche évaluation : métriques, diagnostics par segment et rapports."""

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.reports import ReportBuilder

__all__ = ["EvaluationResult", "Evaluator", "ReportBuilder"]
