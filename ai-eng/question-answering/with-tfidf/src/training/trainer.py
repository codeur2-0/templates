"""Index building, monitored by the shared callbacks.

The trainer is deliberately thin. It does not know how a corpus is chunked nor how passages are
scored — the model owns that — it owns the *protocol*:

1. fire ``on_train_begin`` with the resolved parameters,
2. hand the corpus (and the validation questions) to the model,
3. measure the validation questions with the same metric definitions as the final evaluation,
4. fire ``on_epoch_end`` so that early stopping / threshold callbacks see the numbers,
5. warn when the validation score misses the contractual minimum,
6. return a :class:`TrainingOutcome` carrying the fit result, the metrics and the passages.

Monitoring on the *validation* questions (never the test ones) is what keeps the reported test
score honest: it is measured exactly once, by the evaluation pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base import BaseModel, FitResult
from src.training.callbacks import BaseCallback, CallbackContext
from src.training.losses_metrics import (
    abstention_accuracy,
    answer_f1,
    citation_precision,
    ndcg_at_k,
    percentile,
    recall_at_k,
    retrieval_metrics,
)
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Cut-offs reported by the validation metrics.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class TrainingOutcome:
    """Everything one index build produced.

    Attributes:
        fit_result: Result returned by the model.
        metrics: Validation metrics (document-level recall, answer grounding, latency).
        history: Metric history, epoch by epoch (one entry for a single-shot index build).
        n_validation_questions: Number of questions used for monitoring.
        n_val_questions_scored: Number of questions where the ranking metric was applicable.
        chunks_path: Path of the persisted passage table, when written.
    """

    fit_result: FitResult
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_validation_questions: int = 0
    n_val_questions_scored: int = 0
    chunks_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the run."""
        return {
            "fit_result": self.fit_result.to_dict(),
            "metrics": dict(self.metrics),
            "history": {key: list(values) for key, values in self.history.items()},
            "n_validation_questions": self.n_validation_questions,
            "n_val_questions_scored": self.n_val_questions_scored,
            "chunks_path": None if self.chunks_path is None else str(self.chunks_path),
        }


class Trainer:
    """Build the index and monitor it on the validation questions."""

    def __init__(
        self,
        model: BaseModel,
        *,
        config: Mapping[str, Any],
        paths: ProjectPaths | None = None,
        metric_names: Sequence[str] | None = None,
        ks: Sequence[int] = DEFAULT_KS,
        min_primary_metric: float | None = None,
        primary_metric: str = "recall_at_5",
        answer_k: int = 5,
        chunks_per_document: int = 3,
        callbacks: Sequence[BaseCallback] | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            model: Model to fit (any :class:`BaseModel` implementation).
            config: ``train`` node of the configuration.
            paths: Project filesystem layout.
            metric_names: Metrics requested by the manifest (used to decide whether the answer
                metrics must be computed at all: a pure retriever answers nothing).
            ks: Ranking cut-offs.
            min_primary_metric: Contractual minimum of the primary metric (a warning, not a hard
                failure: the report is what decides compliance).
            primary_metric: Name of the primary metric in ``metrics``.
            answer_k: Number of passages retrieved when computing the answer metrics.
            chunks_per_document: Passages retrieved per document of the monitored ranking (a
                document is counted once, whatever the number of its passages in the top-k).
            callbacks: Callbacks to fire.
        """
        self.model = model
        self.config = dict(config)
        self.paths = paths or ProjectPaths.from_root()
        self.metric_names = tuple(metric_names or ())
        self.ks = tuple(int(k) for k in ks)
        self.min_primary_metric = min_primary_metric
        self.primary_metric = primary_metric
        self.answer_k = int(answer_k)
        self.chunks_per_document = max(int(chunks_per_document), 1)
        self.callbacks: list[BaseCallback] = list(callbacks or [])
        self.context = CallbackContext(
            model_name=type(model).__name__,
            params=dict(model.params),
            epochs=1,
        )

    def run(
        self,
        documents: pd.DataFrame,
        queries: pd.DataFrame | None = None,
        *,
        calibration: pd.DataFrame | None = None,
    ) -> TrainingOutcome:
        """Fit the model, measure it on the validation questions and return the outcome.

        Args:
            documents: Reference corpus.
            queries: Validation questions (``None`` disables the monitored metrics).
            calibration: Calibration questions, handed to the model so that it can tune its own
                decision threshold without touching the validation or test splits.

        Returns:
            The :class:`TrainingOutcome`.
        """
        self._fire("on_train_begin")
        fit_result = self.model.fit(
            documents,
            calibration if calibration is not None and not calibration.empty else queries,
            callbacks=self.callbacks,
            context=self.context,
        )
        validation, n_scored = self._validate(queries)
        metrics = {**{f"val_{key}": value for key, value in validation.items()}}
        metrics.update(
            {
                f"index_{key}": float(value)
                for key, value in fit_result.extra.items()
                if isinstance(value, (int, float))
            }
        )
        self.context.logs = dict(metrics)
        self._fire("on_epoch_end")
        self._fire("on_train_end")
        self._warn_on_threshold(metrics)
        history = {key: list(values) for key, values in self.context.history.items()}
        logger.info(
            "Index built | chunks={} | documents={} | {:.2f}s",
            fit_result.n_chunks,
            fit_result.n_documents,
            fit_result.duration_seconds,
        )
        return TrainingOutcome(
            fit_result=fit_result,
            metrics=metrics,
            history=history,
            n_validation_questions=0 if queries is None else len(queries),
            n_val_questions_scored=n_scored,
        )

    # ------------------------------------------------------------------ interne -----------
    def _validate(self, queries: pd.DataFrame | None) -> tuple[dict[str, float], int]:
        """Measure the fitted model on the validation questions.

        The monitoring metric is *document-level* recall: a retrieved passage counts as found
        when its document is annotated relevant. It is chunking-independent, cheap, and it is
        exactly what the notebooks compare across chunk sizes. The final evaluation measures the
        sharper passage-level and span-level metrics (see ``src/evaluation/evaluator.py``).

        Args:
            queries: Validation questions, or ``None``.

        Returns:
            The metrics and the number of questions where the ranking metric was applicable.
        """
        if queries is None or queries.empty:
            return {}, 0
        answerable = queries.loc[queries["answer_type"] != "unanswerable"]
        rankings: list[list[str]] = []
        relevances: list[set[str]] = []
        latitudes: list[float] = []
        f1_scores: list[float] = []
        citation_scores: list[float] = []
        decisions: list[float] = []
        window = max(self.ks) * self.chunks_per_document
        for record in answerable.itertuples(index=False):
            passages = self.model.retrieve(str(record.question), window)
            rankings.append(
                list(dict.fromkeys(passage.doc_id for passage in passages))[: max(self.ks)]
            )
            relevances.append({item for item in str(record.gold_doc_ids).split(",") if item})
        metrics = retrieval_metrics(rankings, relevances, ks=self.ks) if rankings else {}
        keep = {
            key: value
            for key, value in metrics.items()
            if key.startswith(("recall_at_", "mrr", "ndcg_at_", "n_scored_"))
        }
        if self._needs_answer_metrics():
            for record in queries.itertuples(index=False):
                answer = self.model.answer(
                    str(record.question), self.answer_k, query_id=str(record.query_id)
                )
                latitudes.append(answer.latency_ms)
                should_abstain = str(record.answer_type) == "unanswerable"
                decisions.append(abstention_accuracy(answer.abstained, should_abstain))
                if not should_abstain:
                    f1_scores.append(answer_f1(answer.text, str(record.reference_answer)))
                    relevant_chunks = {passage.chunk_id for passage in answer.passages}
                    citation_scores.append(
                        citation_precision(answer.cited_chunk_ids, relevant_chunks)
                    )
            keep["answer_f1"] = _safe_mean(f1_scores)
            keep["citation_precision"] = _safe_mean(citation_scores)
            keep["abstention_accuracy"] = _safe_mean(decisions)
            keep["latency_p50_ms"] = percentile(latitudes, 0.50)
            keep["latency_p95_ms"] = percentile(latitudes, 0.95)
        return keep, int(metrics.get("n_scored_recall_at_5", 0.0))

    def _needs_answer_metrics(self) -> bool:
        """Whether the manifest asks for answer / grounding metrics."""
        return any(name in self.metric_names for name in ("answer_f1", "citation_precision")) or (
            "answer_f1" in self.config.get("metrics", {})
        )

    def _warn_on_threshold(self, metrics: Mapping[str, float]) -> None:
        """Log a warning when the primary metric misses its contractual minimum."""
        if self.min_primary_metric is None:
            return
        value = metrics.get(f"val_{self.primary_metric}")
        if value is None:
            return
        if value < float(self.min_primary_metric):
            logger.warning(
                "Validation '{}' = {:.4f} is below the contractual minimum {:.4f}: the report "
                "will state it rather than hide it",
                self.primary_metric,
                float(value),
                float(self.min_primary_metric),
            )

    def _fire(self, hook: str) -> None:
        """Call one hook on every callback."""
        for callback in self.callbacks:
            getattr(callback, hook)(self.context)


def _safe_mean(values: Sequence[float]) -> float:
    """Mean of the finite values, NaN when there is none."""
    finite = [float(value) for value in values if value == value]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


__all__ = ["DEFAULT_KS", "Trainer", "TrainingOutcome", "ndcg_at_k", "recall_at_k"]
