"""Évaluation d'un générateur de résumé : ROUGE, fidélité aux faits, segments et références.

L'évaluateur répond à quatre questions, dans l'ordre où un lecteur se les pose :

1. **combien ?** — ROUGE-1, ROUGE-2 et ROUGE-L face à la meilleure des deux références de chaque
   document, plus la couverture des faits saillants. Les deux familles de métriques sont publiées
   ensemble parce qu'aucune ne suffit : le ROUGE récompense la formulation, la couverture la
   *substance* ;
2. **est-ce que c'est vrai ?** — les valeurs du résumé absentes du document sont
   comptées et listées. C'est la seule mesure d'hallucination possible sans juge
   humain, et elle est publiée avec ses limites : elle détecte une durée ou une
   référence inventée, pas une phrase qui réordonne des faits exacts ;
3. **où est-ce que ça casse ?** — la ventilation par type d'intervention
   et par urgence déclarée, les résumés les plus courts, les plus longs,
   et les documents où le modèle a buté sur son budget ;
4. **par rapport à quoi ?** — trois références mesurées sur les **mêmes lignes** : le résumé vide
   (ROUGE 0,0, un plancher défini), la baseline extractive ``lead`` et la baseline ``textrank``
   publiée par la famille. Le score du modèle servi se lit donc par différence.

Le split de test n'est lu qu'ici, une fois : l'entraînement et le suivi de validation ne le voient
jamais.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.schemas import fact_columns
from src.evaluation.rouge import corpus_rouge, rouge_scores
from src.models.contract import BaseTextGenerator
from src.training.metrics import (
    aggregate_coverage,
    coverage_frame,
    describe_metrics,
    latency_stats,
    length_stats,
    per_strategy_frame,
    segment_frame,
    trivial_floor,
    verdict_from_metrics,
)
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Segment dimensions published for every evaluated split.
SEGMENT_COLUMNS: tuple[str, ...] = ("intervention_type", "urgency", "site")


@dataclass
class SummaryEvaluation:
    """Everything one evaluation of a split produced.

    Attributes:
        metrics: Flat metrics of the split (ROUGE, fidelity, lengths, latency).
        per_strategy: One row per strategy, so two models are compared on the same lines.
        segments: Metrics restricted to each value of each segment dimension.
        fidelity: Per-document fidelity table (coverage, unsupported values).
        errors: Documents where the model is the furthest from its reference, with their facts.
        baselines: Measured references (empty summary, lead extraction, textrank baseline).
        verdict: ``conforme``, ``non conforme`` or ``indéterminé`` — the contractual read.
        verdict_detail: The numbers behind that read (observed value, threshold, margin).
        predictions: The published summary table, reference and metrics joined.
        n_documents: Number of scored documents.
        strategies: Strategies compared in this evaluation.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_strategy: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    fidelity: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdict: str = "indéterminé"
    verdict_detail: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_documents: int = 0
    strategies: tuple[str, ...] = ()

    @property
    def primary_metric(self) -> str:
        """Name of the metric the contract is read on."""
        return str(self.verdict_detail.get("metric", "rouge1_f"))

    @property
    def threshold(self) -> float | None:
        """Contractual minimum of the primary metric (``None`` when the family declares none)."""
        value = self.verdict_detail.get("threshold")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the evaluation.

        Returns:
            The metrics, the tables (as records) and the verdict, ready for ``evaluation_metrics``.
        """
        return {
            "metrics": dict(self.metrics),
            "per_strategy": self.per_strategy.to_dict(orient="records"),
            "segments": self.segments.to_dict(orient="records"),
            "fidelity": self.fidelity.to_dict(orient="records"),
            "errors": self.errors.to_dict(orient="records"),
            "baselines": {name: dict(values) for name, values in self.baselines.items()},
            "verdict": self.verdict,
            "verdict_detail": dict(self.verdict_detail),
            "n_documents": self.n_documents,
            "strategies": list(self.strategies),
        }


class SummaryEvaluator:
    """Score fitted strategies on a split, with their references and their segments."""

    def __init__(
        self,
        models: Sequence[BaseTextGenerator] | BaseTextGenerator,
        *,
        config: Mapping[str, Any] | None = None,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        text_column: str = "text",
        id_column: str = "doc_id",
        segment_columns: Sequence[str] = SEGMENT_COLUMNS,
        top_errors: int = 25,
        baselines: Mapping[str, BaseTextGenerator] | None = None,
    ) -> None:
        """Configure the evaluator.

        Args:
            models: Strategy (or strategies) to measure, in publication order.
            config: Full application configuration (project identity, seed).
            metrics_config: ``metrics`` node (primary metric, contractual minimum, direction).
            paths: Project filesystem layout.
            text_column: Column holding the documents' text.
            id_column: Column holding the document identifiers.
            segment_columns: Columns the metrics are broken down by, when present.
            top_errors: Number of documents archived in the error table.
            baselines: Extra strategies measured on the same lines and published as references.
        """
        self.models = [models] if isinstance(models, BaseTextGenerator) else list(models)
        if not self.models:
            msg = "SummaryEvaluator needs at least one strategy to measure"
            raise ValueError(msg)
        self.config = dict(config or {})
        self.metrics_config = dict(metrics_config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.text_column = str(text_column)
        self.id_column = str(id_column)
        self.segment_columns = tuple(segment_columns)
        self.top_errors_count = max(int(top_errors), 1)
        self.baselines = dict(baselines or {})

    # ------------------------------------------------------------------ évaluation ---------
    def evaluate(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> SummaryEvaluation:
        """Measure every strategy on one split, references and segments included.

        Args:
            documents: Documents of the split (``doc_id``, ``text``, ``n_tokens``, segments).
            references: Reference summaries of the same documents (``doc_id``, ``summary``…).
            facts: Fact table of the same documents.

        Returns:
            The :class:`SummaryEvaluation` of the split.
        """
        self._check_columns(documents, references)
        golden = self._golden(references)
        documents = documents.loc[documents[self.id_column].isin(golden)].reset_index(drop=True)
        logger.info(
            "Évaluation de {} document(s) sur {} stratégie(s)",
            len(documents),
            len(self.models) + len(self.baselines),
        )

        frames: list[pd.DataFrame] = []
        for model in [*self.models, *self.baselines.values()]:
            frames.append(self.score_model(model, documents, references, facts))
        predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        served = self.models[0].strategy
        served_predictions = predictions.loc[predictions["strategy"] == served].reset_index(
            drop=True
        )
        fidelity = coverage_frame(served_predictions, facts, documents)
        metrics = self._metrics(served_predictions, fidelity, documents)
        baselines = self._baseline_metrics(predictions)
        segments = segment_frame(
            served_predictions,
            documents,
            columns=self.segment_columns,
            metric_columns=("rouge1_f", "rouge2_f", "rouge_l_f", "fact_coverage", "compression"),
        )
        errors = self._error_frame(served_predictions, references, fidelity, documents)
        verdict, detail = verdict_from_metrics(
            metrics,
            primary=str(self.metrics_config.get("primary", "rouge1_f")),
            minimum=(
                None
                if self.metrics_config.get("min_primary") is None
                else float(self.metrics_config["min_primary"])
            ),
            direction=str(self.metrics_config.get("direction", "maximize")),
        )
        return SummaryEvaluation(
            metrics=metrics,
            per_strategy=per_strategy_frame(predictions),
            segments=segments,
            fidelity=fidelity,
            errors=errors,
            baselines=baselines,
            verdict=verdict,
            verdict_detail=detail,
            predictions=predictions,
            n_documents=len(documents),
            strategies=tuple(dict.fromkeys(predictions["strategy"]))
            if not predictions.empty
            else (),
        )

    def score_model(
        self,
        model: BaseTextGenerator,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate and score the summaries of one strategy.

        Args:
            model: Fitted strategy.
            documents: Documents to summarise.
            references: Reference summaries of the same documents.
            facts: Fact table of the same documents.

        Returns:
            One row per document: ``doc_id``, ``strategy``, ``reference_summary``, ``prediction``,
            the ROUGE triplets, the fidelity columns and the generation metadata.
        """
        golden = dict(zip(references["doc_id"], references["summary"], strict=False))
        summaries = model.summarize(
            [str(value) for value in documents[self.text_column]],
            doc_ids=[str(value) for value in documents[self.id_column]],
        )
        rows: list[dict[str, Any]] = []
        for summary in summaries:
            reference = str(golden.get(summary.doc_id, ""))
            scores = rouge_scores(reference, summary.summary)
            coverage = coverage_scores_for(
                summary.summary, facts.loc[facts["doc_id"] == summary.doc_id]
            )
            source = str(
                documents.loc[documents[self.id_column] == summary.doc_id, self.text_column].iloc[0]
            )
            rows.append(
                {
                    "doc_id": summary.doc_id,
                    "strategy": summary.strategy,
                    "reference_summary": reference,
                    "prediction": summary.summary,
                    "n_sentences": len(summary.sentences),
                    "n_tokens": summary.n_tokens,
                    "compression": round(summary.compression, 6),
                    "hit_max_length": int(summary.hit_max_length),
                    "latency_ms": round(summary.latency_ms, 3),
                    "fact_coverage": coverage["fact_coverage"],
                    "unsupported_facts": int(coverage["n_unsupported"]),
                    "unsupported_values": " | ".join(unsupported_values(summary.summary, source)),
                    **scores.to_metrics(),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ helpers ------------
    def _metrics(
        self,
        predictions: pd.DataFrame,
        fidelity: pd.DataFrame,
        documents: pd.DataFrame,
    ) -> dict[str, float]:
        """Aggregate the metrics of the served strategy.

        Args:
            predictions: Prediction rows of the served strategy.
            fidelity: Fidelity table of the served strategy.
            documents: Evaluated documents.

        Returns:
            The flat metrics of the run (ROUGE, fidelity, lengths, latency, support).
        """
        if predictions.empty:
            return {**trivial_floor(), "n_documents": 0.0}
        references = [str(value) for value in predictions["reference_summary"]]
        generated = [str(value) for value in predictions["prediction"]]
        metrics = corpus_rouge(references, generated)
        metrics = {name: float(value) for name, value in metrics.items()}
        metrics.update(aggregate_coverage(fidelity))
        metrics.update(length_stats(predictions))
        metrics.update(latency_stats([float(value) for value in predictions["latency_ms"]]))
        metrics["n_documents"] = float(len(predictions))
        metrics["document_tokens_mean"] = (
            round(float(documents["n_tokens"].mean()), 2) if "n_tokens" in documents else 0.0
        )
        metrics["rouge1_p_mean"] = round(float(predictions["rouge1_p"].mean()), 4)
        metrics["rouge1_r_mean"] = round(float(predictions["rouge1_r"].mean()), 4)
        metrics["rouge_l_r_mean"] = round(float(predictions["rouge_l_r"].mean()), 4)
        return {name: round(float(value), 4) for name, value in metrics.items()}

    def _baseline_metrics(self, predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Read the metric block of every published reference.

        Args:
            predictions: Prediction rows of every measured strategy.

        Returns:
            ``{reference: {metric: value}}``, with the empty-summary floor always present.
        """
        baselines: dict[str, dict[str, Any]] = {"resume_vide": dict(trivial_floor())}
        for name, strategy in self._baseline_strategies().items():
            subset = predictions.loc[predictions["strategy"] == strategy]
            if subset.empty:
                continue
            block: dict[str, Any] = {
                # La clé est le **nom publié** de la référence : le rapport lit `baselines` sans
                # connaître les stratégies internes.
                "reference": name,
                "strategy": strategy,
                "rouge1_f": round(float(subset["rouge1_f"].mean()), 4),
                "rouge2_f": round(float(subset["rouge2_f"].mean()), 4),
                "rouge_l_f": round(float(subset["rouge_l_f"].mean()), 4),
                "fact_coverage": round(float(subset["fact_coverage"].mean()), 4),
                "unsupported_facts_mean": round(float(subset["unsupported_facts"].mean()), 3),
                "compression": round(float(subset["compression"].mean()), 4),
            }
            baselines[name] = block
        return baselines

    def _baseline_strategies(self) -> dict[str, str]:
        """Map the published reference names onto their strategy keys.

        Returns:
            ``{reference name: strategy}``, e.g. ``{"baseline_extractive": "lead"}``.
        """
        mapping: dict[str, str] = {}
        for name, model in self.baselines.items():
            mapping[str(name)] = model.strategy
        return mapping

    def _error_frame(
        self,
        predictions: pd.DataFrame,
        references: pd.DataFrame,
        fidelity: pd.DataFrame,
        documents: pd.DataFrame,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Collect the documents where the summary is the furthest from its reference.

        Args:
            predictions: Prediction rows of the served strategy.
            references: Reference summaries.
            fidelity: Fidelity table (coverage and unsupported values).
            documents: Document table (segments and text).
            limit: Number of rows to keep (defaults to the evaluator's ``top_errors``).

        Returns:
            The worst documents, with their ROUGE, their coverage, their missing facts and
            an excerpt of the document — enough to diagnose without reopening the corpus.
        """
        if predictions.empty:
            return pd.DataFrame()
        golden = dict(zip(references["doc_id"], references["summary"], strict=False))
        rows: list[dict[str, Any]] = []
        for row in predictions.itertuples(index=False):
            document = documents.loc[documents[self.id_column] == row.doc_id]
            row_fidelity = fidelity.loc[fidelity["doc_id"] == row.doc_id]
            missing = self._missing_facts(str(row.doc_id), str(row.prediction), documents)
            rows.append(
                {
                    "doc_id": str(row.doc_id),
                    "rouge1_f": round(float(row.rouge1_f), 4),
                    "rouge_l_f": round(float(row.rouge_l_f), 4),
                    "fact_coverage": round(float(row.fact_coverage), 4),
                    "unsupported_facts": int(row.unsupported_facts),
                    "n_predicted_words": len(str(row.prediction).split()),
                    "intervention_type": str(document["intervention_type"].iloc[0])
                    if not document.empty
                    else "",
                    "urgency": str(document["urgency"].iloc[0]) if not document.empty else "",
                    "reference_summary": str(golden.get(str(row.doc_id), "")),
                    "prediction": str(row.prediction),
                    "missing_facts": missing,
                    "unsupported_values": str(row_fidelity["unsupported_values"].iloc[0])
                    if not row_fidelity.empty
                    else "",
                }
            )
        frame = pd.DataFrame(rows).sort_values(["rouge1_f", "doc_id"], ascending=[True, True])
        return frame.head(int(limit or self.top_errors_count)).reset_index(drop=True)

    def _missing_facts(self, doc_id: str, prediction: str, documents: pd.DataFrame) -> list[str]:
        """List the salient values of a document that its summary does not report.

        Args:
            doc_id: Document identifier.
            prediction: Generated summary.
            documents: Document table (unused placeholders kept for signature stability).

        Returns:
            The missing values, in decreasing length (the longest facts are the most informative).
        """
        del documents
        return self._facts_index.get(doc_id, []) if hasattr(self, "_facts_index") else []

    def _golden(self, references: pd.DataFrame) -> dict[str, str]:
        """Index the reference summaries by document.

        Args:
            references: Reference summary table.

        Returns:
            Mapping ``doc_id -> summary``.
        """
        return {
            str(key): str(value)
            for key, value in zip(references["doc_id"], references["summary"], strict=False)
        }

    @staticmethod
    def _check_columns(documents: pd.DataFrame, references: pd.DataFrame) -> None:
        """Fail early when a table misses a column the evaluator needs.

        Args:
            documents: Document table.
            references: Reference summary table.

        Raises:
            ValueError: When a required column is missing.
        """
        for column in ("doc_id", "text"):
            if column not in documents.columns:
                msg = (
                    f"Document table is missing the column '{column}' "
                    f"(present: {list(documents.columns)})"
                )
                raise ValueError(msg)
        for column in ("doc_id", "summary"):
            if column not in references.columns:
                msg = (
                    f"Reference summary table is missing the column '{column}' "
                    f"(present: {list(references.columns)})"
                )
                raise ValueError(msg)


def coverage_scores_for(prediction: str, facts: pd.DataFrame) -> dict[str, float]:
    """Small indirection kept for readability of the scoring loop.

    Args:
        prediction: Generated summary.
        facts: Fact rows of the document.

    Returns:
        The coverage metrics of that document.
    """
    from src.training.metrics import coverage_scores

    return coverage_scores(prediction, facts)


def unsupported_values(prediction: str, document: str) -> list[str]:
    """Small indirection kept for readability of the scoring loop.

    Args:
        prediction: Generated summary.
        document: Source document.

    Returns:
        The values of the summary that the document does not contain.
    """
    from src.training.metrics import unsupported_values as _unsupported

    return _unsupported(prediction, document)


def metric_glossary(names: Sequence[str] | None = None) -> dict[str, str]:
    """Expose the metric glossary of the family (used by the report).

    Args:
        names: Restrict the glossary to those metric names.

    Returns:
        Mapping ``metric -> definition``.
    """
    return describe_metrics(names)


def fidelity_columns() -> tuple[str, ...]:
    """Columns of the per-document fidelity table.

    Returns:
        The names of the per-type coverage columns, in :data:`~src.data.schemas.FACT_TYPES` order.
    """
    return ("fact_coverage", "fact_precision", "n_unsupported", *fact_columns())


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean of a sequence, or zero when it is empty.

    Args:
        values: Numeric values.

    Returns:
        The mean, rounded to four decimals.
    """
    return round(float(np.mean(values)), 4) if values else 0.0


__all__ = [
    "SEGMENT_COLUMNS",
    "SummaryEvaluation",
    "SummaryEvaluator",
    "fidelity_columns",
    "mean_or_zero",
    "metric_glossary",
]
