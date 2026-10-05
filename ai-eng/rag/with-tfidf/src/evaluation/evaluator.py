"""Evaluation of a retrieval-augmented generation system.

The evaluator answers three questions, in this order, and never mixes them up:

1. **Does it find the right passages?** — document-level ranking metrics (recall@k, MRR,
   nDCG@k), computed per question then averaged, compared to two trivial references (a random
   ranking and the corpus order). A recall@5 of 0.8 means nothing until it is compared to the
   0.14 a random ranking reaches on the same questions.
2. **Do the answers stand on the retrieved passages?** — citation precision/recall, answer F1
   and exact match against the reference wording, and the abstention decision on the questions
   the corpus cannot answer (``hors_corpus``).
3. **Where does it fail?** — the same metrics per difficulty, per intent and per question family,
   plus the rank of the first relevant passage, which separates "the passage is not in the index"
   from "the passage is ranked 7th".

Relevance is a *judgement*, so it is stated explicitly: a retrieved passage is relevant when its
document is annotated relevant **and** it overlaps the annotated answer span (at least half of
the span's content words). The overlap rule is what prevents an unanswerable question from
scoring a point by citing any passage of the document that happens to talk about the topic.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.transformers import tokenize
from src.training.losses_metrics import (
    RETRIEVAL_METRICS,
    abstention_accuracy,
    answer_exact_match,
    answer_f1,
    citation_precision,
    citation_recall,
    flatten_metrics,
    percentile,
    retrieval_metrics,
)
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Fraction of the annotated answer span that a passage must contain to count as relevant.
DEFAULT_SPAN_OVERLAP = 0.5

#: Separator used when a question has several answering extracts (multi-document questions).
SPAN_SEPARATOR = "||"

#: Segments reported by the evaluation (column -> values are discovered dynamically).
SEGMENT_COLUMNS: tuple[str, ...] = ("difficulty", "intent", "answer_type")


def _content_tokens(text: str) -> set[str]:
    """Return the content tokens of a text (used for the span-overlap judgement)."""
    return {token for token in tokenize(text) if len(token) > 3}


def query_relevance(
    query: Mapping[str, Any],
    passages: Sequence[Any],
    *,
    span_overlap: float = DEFAULT_SPAN_OVERLAP,
) -> set[str]:
    """Identify which retrieved passages really answer a question.

    Args:
        query: Annotated question (``gold_doc_ids``, ``gold_span``, ``answer_type``).
        passages: Retrieved passages (objects exposing ``chunk_id``, ``doc_id`` and ``text``).
        span_overlap: Share of the answer span's content words a passage must contain.

    Returns:
        The identifiers of the relevant passages (empty for an unanswerable question).
    """
    if str(query.get("answer_type")) == "unanswerable":
        return set()
    gold_documents = {item for item in str(query.get("gold_doc_ids", "")).split(",") if item}
    spans = [
        _content_tokens(span)
        for span in str(query.get("gold_span", "")).split(SPAN_SEPARATOR)
        if span.strip()
    ]
    relevant: set[str] = set()
    for passage in passages:
        if passage.doc_id not in gold_documents:
            continue
        if not spans:
            relevant.add(passage.chunk_id)
            continue
        passage_tokens = _content_tokens(passage.text)
        if any(
            len(passage_tokens & span_tokens) / len(span_tokens) >= span_overlap
            for span_tokens in spans
        ):
            relevant.add(passage.chunk_id)
    return relevant


@dataclass
class EvaluationResult:
    """Everything one evaluation run produced.

    Attributes:
        metrics: Aggregated metrics (primary and secondary of the manifest).
        per_question: One row per question (rank of the first relevant passage, metrics, decision).
        baselines: Metrics of the trivial references, keyed by reference name.
        segments: Metrics per segment value (``difficulty``, ``intent``, ``answer_type``).
        ks: Cut-offs used.
        n_questions: Number of evaluated questions.
        n_answerable: Number of questions that have an answer in the corpus.
        notes: Human readable remarks attached to the run (used by the report).
    """

    metrics: dict[str, float]
    per_question: pd.DataFrame
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    segments: dict[str, dict[str, float]] = field(default_factory=dict)
    ks: tuple[int, ...] = (1, 3, 5, 10)
    n_questions: int = 0
    n_answerable: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload (the per-question detail is excluded by design)."""
        return {
            "metrics": dict(self.metrics),
            "baselines": {name: dict(values) for name, values in self.baselines.items()},
            "segments": {name: dict(values) for name, values in self.segments.items()},
            "ks": list(self.ks),
            "n_questions": int(self.n_questions),
            "n_answerable": int(self.n_answerable),
            "notes": list(self.notes),
        }


class Evaluator:
    """Measure a fitted model on annotated questions."""

    def __init__(
        self,
        model: Any,
        *,
        config: Mapping[str, Any] | None = None,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        ks: Sequence[int] = (1, 3, 5, 10),
        answer_k: int = 5,
        span_overlap: float = DEFAULT_SPAN_OVERLAP,
        chunks_per_document: int = 3,
        random_state: int = 42,
    ) -> None:
        """Configure the evaluator.

        Args:
            model: Fitted model (any object implementing the :class:`BaseModel` contract).
            config: Full application configuration (read for the seed and the generator metadata).
            metrics_config: ``metrics`` node (primary metric, contractual minimum).
            paths: Project filesystem layout.
            ks: Ranking cut-offs.
            answer_k: Number of passages retrieved before generating an answer.
            span_overlap: Span-overlap threshold used by :func:`query_relevance`.
            chunks_per_document: Number of passages retrieved per document of the final ranking.
                A document is presented once in a search result: retrieving ``k`` passages and
                calling the list a "top-k documents" would let a single document occupy three
                slots, which understates the recall of every system. The evaluator therefore
                retrieves ``k * chunks_per_document`` passages and deduplicates the documents.
            random_state: Seed of the random baseline.
        """
        self.model = model
        self.config = dict(config or {})
        self.metrics_config = dict(metrics_config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.ks = tuple(int(k) for k in ks)
        self.answer_k = int(answer_k)
        self.span_overlap = float(span_overlap)
        self.chunks_per_document = max(int(chunks_per_document), 1)
        self.random_state = int(random_state)

    @property
    def primary_metric(self) -> str:
        """Name of the primary metric declared by the manifest."""
        return str(self.metrics_config.get("primary", "recall_at_5"))

    @property
    def min_primary(self) -> float | None:
        """Contractual minimum of the primary metric, when declared."""
        value = self.metrics_config.get("min_primary")
        return None if value is None else float(value)

    def evaluate(
        self, queries: pd.DataFrame, documents: pd.DataFrame | None = None
    ) -> EvaluationResult:
        """Evaluate the model on a set of annotated questions.

        Args:
            queries: Annotated questions (the test split, ideally).
            documents: Reference corpus, used by the baselines and the explicability features.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            ValueError: When the question frame is empty.
        """
        if queries.empty:
            msg = "No question to evaluate: the test split is empty"
            raise ValueError(msg)

        rows: list[dict[str, Any]] = []
        doc_rankings: list[list[str]] = []
        doc_relevances: list[set[str]] = []
        chunk_rankings: list[list[str]] = []
        chunk_relevances: list[set[str]] = []
        wants_answers = self._wants_answer_metrics()
        for record in queries.to_dict(orient="records"):
            retrieved = self.model.retrieve(
                str(record["question"]), max(self.ks) * self.chunks_per_document
            )
            relevant_chunks = query_relevance(record, retrieved, span_overlap=self.span_overlap)
            gold_documents = {
                item for item in str(record.get("gold_doc_ids", "")).split(",") if item
            }
            document_ranking = list(dict.fromkeys(passage.doc_id for passage in retrieved))[
                : max(self.ks)
            ]
            doc_rankings.append(document_ranking)
            doc_relevances.append(
                set() if record.get("answer_type") == "unanswerable" else gold_documents
            )
            chunk_rankings.append([passage.chunk_id for passage in retrieved])
            chunk_relevances.append(relevant_chunks)

            row: dict[str, Any] = {
                "query_id": str(record.get("query_id", "")),
                "question": str(record.get("question", "")),
                "difficulty": str(record.get("difficulty", "")),
                "intent": str(record.get("intent", "")),
                "answer_type": str(record.get("answer_type", "")),
                "n_gold_docs": int(record.get("n_gold_docs", 0) or 0),
                "n_relevant_retrieved": len(relevant_chunks),
                "first_relevant_rank": _first_rank(document_ranking, gold_documents),
                "top_score": round(max((passage.score for passage in retrieved), default=0.0), 6),
                "n_retrieved": len(retrieved),
            }
            row.update(self._row_metrics(document_ranking, gold_documents, relevant_chunks))
            row.update(
                self._answer_row(record, retrieved, relevant_chunks) if wants_answers else {}
            )
            rows.append(row)

        per_question = pd.DataFrame(rows)
        metrics = retrieval_metrics(doc_rankings, doc_relevances, ks=self.ks)
        metrics = dict(metrics.items())
        metrics.update(self._passage_metrics(chunk_rankings, chunk_relevances))
        if wants_answers:
            metrics.update(_aggregate_answer_metrics(per_question))
        metrics["n_questions"] = float(len(queries))
        metrics["n_answerable"] = float((queries["answer_type"] != "unanswerable").sum())
        metrics["n_indexed_chunks"] = float(self._n_chunks())

        baselines = self._baselines(queries, documents)
        segments = self._segments(per_question)
        notes = self._notes(metrics)
        logger.info(
            "Evaluation finished | {} questions | {}={} | random={}",
            len(queries),
            self.primary_metric,
            _fmt(metrics.get(self.primary_metric)),
            _fmt(baselines.get("random", {}).get(self.primary_metric)),
        )
        return EvaluationResult(
            metrics=flatten_metrics(metrics),
            per_question=per_question,
            baselines={name: flatten_metrics(values) for name, values in baselines.items()},
            segments={name: flatten_metrics(values) for name, values in segments.items()},
            ks=self.ks,
            n_questions=len(queries),
            n_answerable=int((queries["answer_type"] != "unanswerable").sum()),
            notes=notes,
        )

    # ------------------------------------------------------------------ interne -----------
    def _wants_answer_metrics(self) -> bool:
        """Whether answer / grounding metrics were requested by the manifest."""
        declared = {
            str(name)
            for name in [
                self.metrics_config.get("primary", ""),
                *list(self.metrics_config.get("secondary", []) or []),
            ]
        }
        return bool(getattr(self.model, "supports_generation", True)) and bool(
            declared
            & {"answer_f1", "answer_exact_match", "citation_precision", "abstention_accuracy"}
        )

    def _row_metrics(
        self, ranking: Sequence[str], relevant: set[str], relevant_chunks: set[str]
    ) -> dict[str, float]:
        """Compute the per-question ranking metrics."""
        values: dict[str, float] = {}
        for k in self.ks:
            values[f"recall_at_{k}"] = _ratio(len(set(ranking[:k]) & relevant), len(relevant))
            values[f"precision_at_{k}"] = _ratio(
                len(set(ranking[:k]) & relevant), max(len(ranking[:k]), 1)
            )
            values[f"ndcg_at_{k}"] = _ndcg(ranking, relevant, k)
        values["mrr"] = _reciprocal_rank(ranking, relevant)
        values["n_relevant_retrieved"] = float(len(relevant_chunks))
        return values

    def _answer_row(
        self, record: Mapping[str, Any], retrieved: Sequence[Any], relevant_chunks: set[str]
    ) -> dict[str, float]:
        """Answer the question and measure the grounding of the produced answer."""
        answer = self.model.answer(
            str(record["question"]),
            self.answer_k,
            query_id=str(record.get("query_id", "")),
        )
        should_abstain = str(record.get("answer_type")) == "unanswerable"
        return {
            "answer_f1": answer_f1(answer.text, str(record.get("reference_answer", ""))),
            "answer_exact_match": answer_exact_match(
                answer.text, str(record.get("reference_answer", ""))
            ),
            "citation_precision": _nan_to_float(
                citation_precision(answer.cited_chunk_ids, relevant_chunks)
            ),
            "citation_recall": _nan_to_float(
                citation_recall(answer.cited_chunk_ids, relevant_chunks)
            ),
            "abstained": float(bool(answer.abstained)),
            "should_abstain": float(should_abstain),
            "abstention_correct": abstention_accuracy(answer.abstained, should_abstain),
            "n_citations": float(len(answer.cited_chunk_ids)),
            "latency_ms": float(answer.latency_ms),
        }

    def _passage_metrics(
        self, rankings: Sequence[Sequence[str]], relevances: Sequence[set[str]]
    ) -> dict[str, float]:
        """Passage-level precision and hit-rate, computed on the questions with gold passages."""
        usable = [
            (ranking, relevant)
            for ranking, relevant in zip(rankings, relevances, strict=True)
            if relevant
        ]
        metrics: dict[str, float] = {}
        for k in self.ks:
            if not usable:
                metrics[f"passage_precision_at_{k}"] = float("nan")
                metrics[f"passage_hit_rate_at_{k}"] = float("nan")
                continue
            precisions = [
                len(set(ranking[:k]) & relevant) / max(len(ranking[:k]), 1)
                for ranking, relevant in usable
            ]
            hits = [1.0 if set(ranking[:k]) & relevant else 0.0 for ranking, relevant in usable]
            metrics[f"passage_precision_at_{k}"] = float(np.mean(precisions))
            metrics[f"passage_hit_rate_at_{k}"] = float(np.mean(hits))
        metrics["n_questions_with_gold_passage"] = float(len(usable))
        return metrics

    def _baselines(
        self, queries: pd.DataFrame, documents: pd.DataFrame | None
    ) -> dict[str, dict[str, float]]:
        """Measure two trivial references on the same questions.

        ``random`` ranks the documents at random (seeded), ``corpus_order`` ranks them by
        identifier — the "no model at all" reference. Both use the same relevance judgements as
        the model, which is what makes the comparison fair.
        """
        generator = random.Random(self.random_state)
        corpus_documents = list(documents["doc_id"].astype(str)) if documents is not None else []
        random_rankings: list[list[str]] = []
        order_rankings: list[list[str]] = []
        relevances: list[set[str]] = []
        for record in queries.to_dict(orient="records"):
            gold = {item for item in str(record.get("gold_doc_ids", "")).split(",") if item}
            relevant = set() if record.get("answer_type") == "unanswerable" else gold
            relevances.append(relevant)
            pool = corpus_documents or sorted(gold)
            shuffled = list(pool)
            generator.shuffle(shuffled)
            random_rankings.append(shuffled)
            order_rankings.append(sorted(pool))
        return {
            "random": retrieval_metrics(random_rankings, relevances, ks=self.ks),
            "corpus_order": retrieval_metrics(order_rankings, relevances, ks=self.ks),
        }

    def _segments(self, per_question: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Break the metrics down by segment (difficulty, intent, answer type)."""
        segments: dict[str, dict[str, float]] = {}
        for column in SEGMENT_COLUMNS:
            if column not in per_question.columns:
                continue
            for value, group in per_question.groupby(column, dropna=False):
                key = f"{column}={value}"
                segments[key] = {
                    "n_questions": float(len(group)),
                    "recall_at_5": float(group["recall_at_5"].mean(skipna=True)),
                    "mrr": float(group["mrr"].mean(skipna=True)),
                    "mean_first_relevant_rank": float(
                        group["first_relevant_rank"].replace(0, np.nan).mean(skipna=True)
                    ),
                }
                if "answer_f1" in group.columns:
                    segments[key]["answer_f1"] = float(group["answer_f1"].mean(skipna=True))
                    segments[key]["abstention_rate"] = float(group["abstained"].mean(skipna=True))
        return segments

    def _notes(self, metrics: Mapping[str, float]) -> list[str]:
        """Build the human readable remarks attached to the run."""
        notes: list[str] = [
            "La pertinence d'un passage est jugée par le chevauchement avec l'extrait annoté "
            f"(seuil {self.span_overlap:.0%} des mots de contenu) et non par sa seule appartenance "
            "au bon document.",
        ]
        minimum = self.min_primary
        value = metrics.get(self.primary_metric)
        if minimum is not None and value is not None and value == value:
            verdict = "atteint" if float(value) >= minimum else "non atteint"
            notes.append(
                f"Objectif contractuel {self.primary_metric} >= {minimum:.3f} : {verdict} "
                f"(mesuré {float(value):.4f})."
            )
        unanswerable = metrics.get("abstention_accuracy")
        if unanswerable is not None and unanswerable == unanswerable:
            notes.append(
                f"Décision réponse/abstention correcte sur {float(unanswerable):.1%} des questions."
            )
        return notes

    def _n_chunks(self) -> int:
        """Number of indexed passages, when the model exposes it."""
        chunks = getattr(self.model, "chunks", None)
        try:
            return len(chunks) if chunks is not None else 0
        except TypeError:  # pragma: no cover - defensive
            return 0


def _ratio(numerator: int, denominator: int) -> float:
    """Safe ratio, NaN when the denominator is zero (metric not applicable)."""
    return float("nan") if denominator == 0 else numerator / denominator


def _nan_to_float(value: float) -> float:
    """Keep NaN as NaN: pandas propagates it, and the aggregate excludes it explicitly."""
    return float(value)


def _first_rank(ranking: Sequence[str], relevant: set[str]) -> int:
    """Rank of the first relevant document (0 when none was found)."""
    for position, identifier in enumerate(ranking, start=1):
        if identifier in relevant:
            return position
    return 0


def _reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document (NaN without relevant document)."""
    if not relevant:
        return float("nan")
    rank = _first_rank(ranking, relevant)
    return 0.0 if rank == 0 else 1.0 / rank


def _ndcg(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant:
        return float("nan")
    gains = [
        1.0 / np.log2(position + 1)
        for position, identifier in enumerate(ranking[:k], start=1)
        if identifier in relevant
    ]
    ideal = sum(1.0 / np.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return float(sum(gains) / ideal) if ideal else float("nan")


def _mean_or_nan(values: pd.Series) -> float:
    """Mean of a column, NaN when the selection is empty (the metric is then not applicable)."""
    return float(values.mean(skipna=True)) if len(values) else float("nan")


def _aggregate_answer_metrics(per_question: pd.DataFrame) -> dict[str, float]:
    """Aggregate the answer-level columns of the per-question table."""
    if "answer_f1" not in per_question.columns:
        return {}
    answerable = per_question.loc[per_question["should_abstain"] == 0]
    answerable_rows = per_question.index[per_question["should_abstain"] == 0]
    should_abstain = per_question.index[per_question["should_abstain"] == 1]
    latency = per_question["latency_ms"]
    return {
        "answer_f1": float(answerable["answer_f1"].mean(skipna=True)),
        "answer_exact_match": float(answerable["answer_exact_match"].mean(skipna=True)),
        "citation_precision": float(answerable["citation_precision"].mean(skipna=True)),
        "citation_recall": float(answerable["citation_recall"].mean(skipna=True)),
        "abstention_accuracy": float(per_question["abstention_correct"].mean(skipna=True)),
        "abstention_recall": _mean_or_nan(per_question.loc[should_abstain, "abstained"]),
        "answer_coverage": _mean_or_nan(1.0 - per_question.loc[answerable_rows, "abstained"]),
        "abstention_balanced_accuracy": 0.5
        * (
            _mean_or_nan(per_question.loc[should_abstain, "abstained"])
            + _mean_or_nan(1.0 - per_question.loc[answerable_rows, "abstained"])
        ),
        "abstention_rate": float(per_question["abstained"].mean(skipna=True)),
        "false_answer_rate": float(
            per_question.loc[per_question["should_abstain"] == 1, "abstained"].eq(0).mean()
        ),
        "mean_citations": float(per_question["n_citations"].mean(skipna=True)),
        "latency_p50_ms": percentile(latency, 0.50),
        "latency_p95_ms": percentile(latency, 0.95),
    }


def _fmt(value: float | None) -> str:
    """Format a metric for the logs."""
    return "n/a" if value is None or value != value else f"{value:.4f}"


__all__ = [
    "DEFAULT_SPAN_OVERLAP",
    "SPAN_SEPARATOR",
    "RETRIEVAL_METRICS",
    "EvaluationResult",
    "Evaluator",
    "SEGMENT_COLUMNS",
    "query_relevance",
]
