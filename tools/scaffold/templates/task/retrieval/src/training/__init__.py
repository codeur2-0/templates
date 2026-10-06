"""Training layer of the text modality: metrics and index building."""

from src.training.losses_metrics import (
    METRIC_DESCRIPTIONS,
    METRICS_BY_TASK,
    abstention_accuracy,
    answer_exact_match,
    answer_f1,
    citation_precision,
    citation_recall,
    describe_metrics,
    flatten_metrics,
    metrics_for_task,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
)
from src.training.trainer import Trainer, TrainingOutcome

__all__ = [
    "METRICS_BY_TASK",
    "METRIC_DESCRIPTIONS",
    "Trainer",
    "TrainingOutcome",
    "abstention_accuracy",
    "answer_exact_match",
    "answer_f1",
    "citation_precision",
    "citation_recall",
    "describe_metrics",
    "flatten_metrics",
    "metrics_for_task",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "retrieval_metrics",
]
