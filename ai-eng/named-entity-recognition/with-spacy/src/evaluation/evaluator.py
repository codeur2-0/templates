"""Évaluation d'un extracteur d'entités : les métriques, leurs segments et leurs références.

L'évaluateur répond à quatre questions, dans l'ordre où un lecteur se les pose :

1. **combien ?** — les scores micro et macro (:func:`~src.training.metrics.entity_scores`), plus
la
   F1 partielle qui dit ce que coûte l'exigence de bornes exactes ;
2. **où est-ce que ça casse ?** — la table par type, les erreurs classées (mentions manquées,
   inventées, bornes décalées) et les **segments** : le style rédactionnel et le canal d'arrivée.
   Un
   score global qui cache un effondrement sur les messages abrégés n'est pas un résultat ;
3. **par rapport à quoi ?** — deux références mesurées sur les **mêmes lignes** : le plancher
trivial
   (un système qui n'annote rien obtient 0,0) et la **couche de règles** apprise sur le train puis
   évaluée sur le même split. Le gain du modèle se lit donc par différence, pas par intuition ;
4. **peut-on s'en servir ?** — la latence chaude par message, la table de précision par niveau de
   confiance, et la part de mentions dont la surface était **réservée** aux splits d'évaluation :
   c'est la seule mesure qui distingue « le modèle a appris » de « le modèle a recopié une liste
   ».

Le split de test n'est lu qu'ici, une fois : l'entraînement et le suivi de validation ne le voient
jamais.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.schemas import ENTITY_LABELS
from src.inference.extraction import extract_one_by_one
from src.models.contract import BaseEntityTagger
from src.training.metrics import (
    confidence_gap,
    confidence_table,
    entity_scores,
    error_frame,
    latency_stats,
    partial_scores,
    per_label_frame,
    trivial_floor,
)
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Segment dimensions published for every evaluated split.
SEGMENT_COLUMNS: tuple[str, ...] = ("style", "canal")


@dataclass
class EvaluationResult:
    """Everything one evaluation of a split produced.

    Attributes:
        metrics: Flat metrics of the split (entity scores, partial scores, latency, confidence).
        per_label: Precision / recall / F1 / support per entity type.
        segments: Metrics restricted to each value of each segment dimension.
        holdout: Recall on reserved surfaces versus surfaces already seen at training time.
        confidence: Precision observed at every confidence level (a table, not a probability).
        errors: Mistakes classified (missed, invented, shifted boundaries).
        baselines: Measured references (trivial floor, declared rule layer).
        verdict: ``conforme``, ``non conforme`` or ``indéterminé`` — the contractual read.
        verdict_detail: The numbers behind that read (observed value, threshold, margin).
        predictions: The published mention table, with its verdict when the reference is known.
        n_documents: Number of scored messages.
        n_entities: Number of reference entities.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_label: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    verdict: str = "indéterminé"
    verdict_detail: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_documents: int = 0
    n_entities: int = 0

    @property
    def primary_metric(self) -> str:
        """Name of the metric the contract is read on."""
        return str(self.verdict_detail.get("metric", "entity_f1"))

    @property
    def threshold(self) -> float | None:
        """Contractual minimum of the primary metric (``None`` when the family declares none)."""
        value = self.verdict_detail.get("threshold")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the evaluation."""
        return {
            "metrics": dict(self.metrics),
            "per_label": self.per_label.to_dict(orient="records"),
            "segments": self.segments.to_dict(orient="records"),
            "holdout": self.holdout.to_dict(orient="records"),
            "confidence": self.confidence.to_dict(orient="records"),
            "errors": self.errors.to_dict(orient="records"),
            "baselines": {name: dict(values) for name, values in self.baselines.items()},
            "verdict": self.verdict,
            "verdict_detail": dict(self.verdict_detail),
            "n_documents": self.n_documents,
            "n_entities": self.n_entities,
        }


class EntityEvaluator:
    """Score a fitted extractor on a labelled split, with its references and its segments."""

    def __init__(
        self,
        model: BaseEntityTagger,
        *,
        config: Mapping[str, Any],
        paths: ProjectPaths | None = None,
        text_column: str = "text",
        target_column: str = "label",
        id_column: str = "msg_id",
        split_column: str = "split",
        primary_metric: str = "entity_f1",
        min_primary_metric: float | None = None,
        measure_baselines: bool = True,
    ) -> None:
        """Configure the evaluator.

        Args:
            model: Fitted extractor.
            config: Full application configuration (``metrics`` and ``train`` nodes are read).
            paths: Project filesystem layout.
            text_column: Column holding the text.
            target_column: Column holding the entity type of the annotations.
            id_column: Join key between the two tables.
            split_column: Column holding the split names.
            primary_metric: Metric the contract is read on.
            min_primary_metric: Contractual minimum of the primary metric (``None`` disables it).
            measure_baselines: Whether to measure the declared references on the same split.
        """
        self.model = model
        self.config = dict(config)
        self.paths = paths or ProjectPaths.from_root()
        self.text_column = str(text_column)
        self.target_column = str(target_column)
        self.id_column = str(id_column)
        self.split_column = str(split_column)
        self.primary_metric = str(primary_metric)
        self.min_primary_metric = min_primary_metric
        self.measure_baselines = bool(measure_baselines)

    # ------------------------------------------------------------------ évaluation --------
    def evaluate(self, documents: pd.DataFrame, spans: pd.DataFrame) -> EvaluationResult:
        """Score a labelled split.

        Args:
            documents: Messages of the split.
            spans: Reference annotations of the split.

        Returns:
            The :class:`EvaluationResult` of the split.
        """
        label_column = str(self.config.get("data", {}).get("target") or self.target_column)
        predictions, latencies = extract_one_by_one(
            self.model,
            documents,
            text_column=self.text_column,
            id_column=self.id_column,
        )
        gold = self._gold_sets(documents, spans)
        predicted = self._predicted_sets(documents, predictions, gold)
        labels = list(self.model.labels or ENTITY_LABELS)
        holdout = self._holdout(documents, spans, predictions)
        metrics = entity_scores(gold, predicted, labels=labels)
        metrics.update(partial_scores(gold, predicted, labels=labels))
        metrics.update(latency_stats(latencies))
        metrics.update(_holdout_metrics(holdout))
        metrics["n_predicted_entities"] = float(len(predictions))
        metrics["mean_confidence"] = (
            round(float(predictions["confidence"].astype(float).mean()), 4)
            if not predictions.empty
            else 0.0
        )
        metrics["share_from_rules"] = (
            round(float((predictions["source"].astype(str) == "regle").mean()), 4)
            if not predictions.empty
            else 0.0
        )
        metrics["confidence_gap"] = confidence_gap(predictions, spans)
        metrics = {key: float(value) for key, value in metrics.items()}
        published = self._with_reference(predictions, spans, label_column=label_column)
        detail = self._verdict(metrics)
        result = EvaluationResult(
            metrics=metrics,
            per_label=per_label_frame(gold, predicted, labels=labels),
            segments=self._segments(documents, spans, predictions),
            holdout=holdout,
            confidence=confidence_table(predictions, spans),
            errors=error_frame(documents, spans, predictions, limit=25),
            baselines=self._baselines(documents, spans) if self.measure_baselines else {},
            verdict=str(detail["verdict"]),
            verdict_detail=detail,
            predictions=published,
            n_documents=len(documents),
            n_entities=len(spans),
        )
        logger.info(
            "Evaluation | {} messages | {} entités | entity_f1={} | macro_f1={} | {}",
            result.n_documents,
            result.n_entities,
            _format(metrics.get("entity_f1")),
            _format(metrics.get("macro_f1")),
            detail.get("message", ""),
        )
        return result

    # ------------------------------------------------------------------ segments ----------
    def _segments(
        self, documents: pd.DataFrame, spans: pd.DataFrame, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Restrict the entity metrics to each value of each segment dimension.

        Args:
            documents: Messages of the split.
            spans: Reference annotations.
            predictions: Published mention table.

        Returns:
            One row per (segment, value) with the metrics computed on the messages concerned.
        """
        rows: list[dict[str, Any]] = []
        gold_by_message = self._spans_of(spans)
        for column in SEGMENT_COLUMNS:
            if column not in documents.columns:
                continue
            for value in sorted({str(item) for item in documents[column]}):
                selected = documents[documents[column].astype(str) == value]
                identifiers = [str(item) for item in selected[self.id_column]]
                gold = [gold_by_message.get(identifier, set()) for identifier in identifiers]
                predicted = self._predicted_sets_from_frame(predictions, identifiers)
                if not any(gold):
                    continue
                scores = entity_scores(
                    gold, predicted, labels=list(self.model.labels or ENTITY_LABELS)
                )
                scores.update(partial_scores(gold, predicted))
                rows.append(
                    {
                        "segment": column,
                        "value": value,
                        "n_documents": len(selected),
                        "n_entities": int(sum(len(item) for item in gold)),
                        "entity_f1": scores["entity_f1"],
                        "entity_precision": scores["entity_precision"],
                        "entity_recall": scores["entity_recall"],
                        "partial_f1": scores["partial_f1"],
                        "boundary_errors": scores["boundary_errors"],
                    }
                )
        return pd.DataFrame(rows)

    def _holdout(
        self, documents: pd.DataFrame, spans: pd.DataFrame, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare the recall on reserved surfaces with the recall on surfaces seen before.

        Args:
            documents: Messages of the split.
            spans: Reference annotations (``holdout`` column).
            predictions: Published mention table.

        Returns:
            One row per group (``reservee`` / ``vue_au_train``) with the reference count, the
            matched
            ones and the resulting recall. C'est le seul endroit où la dégradation des surfaces
            réservées est chiffrée.
        """
        if spans.empty or "holdout" not in spans.columns:
            return pd.DataFrame(columns=["group", "n_entities", "matched", "recall"])
        if "split" in documents.columns:
            keep = set(documents[self.id_column].astype(str))
            subset = spans[spans[self.id_column].astype(str).isin(keep)]
        else:
            subset = spans
        predicted_keys = {
            (str(row.msg_id), int(row.start), int(row.end), str(row.label))
            for row in predictions.itertuples(index=False)
        }
        rows: list[dict[str, Any]] = []
        for flag, name in ((True, "reservee"), (False, "vue_au_train")):
            group = subset[subset["holdout"].astype(bool) == flag]
            matched = sum(
                1
                for row in group.itertuples(index=False)
                if (str(row.msg_id), int(row.start), int(row.end), str(row.label)) in predicted_keys
            )
            rows.append(
                {
                    "group": name,
                    "n_entities": len(group),
                    "matched": int(matched),
                    "recall": round(matched / len(group), 4) if len(group) else 0.0,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ références --------
    def _baselines(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        """Measure the declared references on the same lines: the trivial floor and the rules.

        Args:
            documents: Messages of the evaluated split.
            spans: Reference annotations of the split.

        Returns:
            A mapping with ``aucune_entite`` (the trivial floor) and ``regles`` (the declared rule
            layer, learned on the **training** annotations then scored here).
        """
        baselines: dict[str, dict[str, float]] = {"aucune_entite": trivial_floor()}
        try:
            from src.models import build_model

            rules = build_model(self.config, algorithm="gazetteer", params={})
            train_documents = self._training_documents(documents)
            train_spans = self._training_annotations(spans)
            if train_documents.empty or train_spans.empty:
                logger.warning(
                    "Couche de règles non mesurée : le train n'est pas joignable depuis ce split"
                )
                return baselines
            rules.fit(train_documents, train_spans)
            predictions, _ = extract_one_by_one(
                rules, documents, text_column=self.text_column, id_column=self.id_column
            )
            gold = self._gold_sets(documents, spans)
            predicted = self._predicted_sets_from_frame(
                predictions, [str(item) for item in documents[self.id_column]]
            )
            scores = entity_scores(gold, predicted, labels=list(self.model.labels or ENTITY_LABELS))
            scores.update(partial_scores(gold, predicted))
            scores["n_predicted_entities"] = float(len(predictions))
            baselines["regles"] = {key: float(value) for key, value in scores.items()}
        except (ValueError, ImportError) as error:  # pragma: no cover - configuration dégradée
            logger.warning("Couche de règles non mesurée : {}", error)
        return baselines

    # ------------------------------------------------------------------ verdict -----------
    def _verdict(self, metrics: Mapping[str, float]) -> dict[str, Any]:
        """Read the primary metric against its contractual minimum.

        Args:
            metrics: Metrics of the evaluated split.

        Returns:
            A mapping with the observed value, the threshold, the verdict and the sentence
            published
            in the report.
        """
        observed = metrics.get(self.primary_metric)
        if observed is None:
            return {
                "metric": self.primary_metric,
                "observed": None,
                "threshold": self.min_primary_metric,
                "passed": False,
                "verdict": "indéterminé",
                "message": f"Métrique '{self.primary_metric}' absente du rapport.",
            }
        threshold = self.min_primary_metric
        passed = threshold is None or float(observed) >= float(threshold)
        verdict = "conforme" if passed else "non conforme"
        if threshold is None:
            message = f"{self.primary_metric}={observed:.4f} (aucun seuil contractuel déclaré)."
        elif passed:
            message = (
                f"{self.primary_metric}={observed:.4f} ≥ {float(threshold):.4f} : "
                "le critère de la famille est satisfait."
            )
        else:
            message = (
                f"{self.primary_metric}={observed:.4f} < {float(threshold):.4f} : "
                "le critère de la famille n'est pas satisfait."
            )
        return {
            "metric": self.primary_metric,
            "observed": float(observed),
            "threshold": None if threshold is None else float(threshold),
            "passed": bool(passed),
            "verdict": verdict,
            "message": message,
        }

    # ------------------------------------------------------------------ interne -----------
    def _spans_of(self, spans: pd.DataFrame) -> dict[str, set[tuple[int, int, str]]]:
        """Index the reference spans by message."""
        grouped: dict[str, set[tuple[int, int, str]]] = {}
        for row in spans.itertuples(index=False):
            grouped.setdefault(str(row.msg_id), set()).add(
                (int(row.start), int(row.end), str(row.label))
            )
        return grouped

    def _gold_sets(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> list[set[tuple[int, int, str]]]:
        """Return the reference spans, one set per message of the split (order preserved)."""
        grouped = self._spans_of(spans)
        return [grouped.get(str(item), set()) for item in documents[self.id_column]]

    def _predicted_sets_from_frame(
        self, predictions: pd.DataFrame, identifiers: Sequence[str]
    ) -> list[set[tuple[int, int, str]]]:
        """Return the predicted spans, one set per identifier (order preserved)."""
        grouped: dict[str, set[tuple[int, int, str]]] = {}
        if not predictions.empty:
            for row in predictions.itertuples(index=False):
                grouped.setdefault(str(row.msg_id), set()).add(
                    (int(row.start), int(row.end), str(row.label))
                )
        return [grouped.get(str(item), set()) for item in identifiers]

    def _predicted_sets(
        self,
        documents: pd.DataFrame,
        predictions: pd.DataFrame,
        gold: Sequence[set[tuple[int, int, str]]],
    ) -> list[set[tuple[int, int, str]]]:
        """Return the predicted spans, keeping only the types the corpus declares.

        Args:
            documents: Messages of the split.
            predictions: Published mention table.
            gold: Reference spans, one set per message (used to bound the labels).

        Returns:
            One set per message, aligned with ``gold``.
        """
        declared = {span[2] for spans in gold for span in spans} or set(ENTITY_LABELS)
        grouped: dict[str, set[tuple[int, int, str]]] = {}
        if not predictions.empty:
            for row in predictions.itertuples(index=False):
                label = str(row.label)
                if label not in declared:
                    continue
                grouped.setdefault(str(row.msg_id), set()).add(
                    (int(row.start), int(row.end), label)
                )
        return [grouped.get(str(item), set()) for item in documents[self.id_column]]

    def _with_reference(
        self, predictions: pd.DataFrame, spans: pd.DataFrame, *, label_column: str
    ) -> pd.DataFrame:
        """Add the reference verdict to a mention table.

        Args:
            predictions: Published mention table.
            spans: Reference annotations.
            label_column: Column holding the entity type of the reference.

        Returns:
            The mention table with ``expected_label`` and ``correct`` columns.
        """
        if predictions.empty:
            return predictions
        expected = {
            (str(row.msg_id), int(row.start), int(row.end)): str(getattr(row, label_column))
            for row in spans.itertuples(index=False)
        }
        frame = predictions.copy()
        keys = zip(frame["msg_id"], frame["start"], frame["end"], strict=True)
        frame["expected_label"] = [expected.get((str(m), int(s), int(e))) for m, s, e in keys]
        frame["correct"] = [
            int(str(label) == str(known)) if known is not None else 0
            for label, known in zip(frame["label"], frame["expected_label"], strict=True)
        ]
        return frame

    def _training_documents(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Return the training messages of the corpus (for the rule baseline)."""
        from src.data.loaders import EntityCorpusLoader

        try:
            corpus = EntityCorpusLoader(self.paths).split("train")
        except (FileNotFoundError, ValueError):
            return pd.DataFrame()
        return corpus if not corpus.empty else pd.DataFrame()

    def _training_annotations(self, spans: pd.DataFrame) -> pd.DataFrame:
        """Return the training annotations of the corpus (for the rule baseline)."""
        from src.data.loaders import EntityCorpusLoader

        try:
            loader = EntityCorpusLoader(self.paths)
            documents, annotations = loader.load_corpus()
        except (FileNotFoundError, ValueError):
            return pd.DataFrame()
        return loader.split_annotations("train", documents=documents, spans=annotations)


def _holdout_metrics(holdout: pd.DataFrame) -> dict[str, float]:
    """Turn the reserved-surface comparison into two headline metrics.

    Args:
        holdout: Table produced by :meth:`EntityEvaluator._holdout`.

    Returns:
        ``holdout_recall`` (reserved surfaces) and ``seen_surface_recall`` (surfaces already seen
        at
        training time). Les deux sont publiées côte à côte : c'est leur écart qui dit si le
        système
        apprend la forme d'un nom ou recopie une liste.
    """
    if holdout.empty:
        return {}
    recalls = {str(row.group): float(row.recall) for row in holdout.itertuples(index=False)}
    metrics: dict[str, float] = {}
    if "reservee" in recalls:
        metrics["holdout_recall"] = recalls["reservee"]
    if "vue_au_train" in recalls:
        metrics["seen_surface_recall"] = recalls["vue_au_train"]
    if {"reservee", "vue_au_train"} <= set(recalls):
        metrics["holdout_gap"] = round(recalls["vue_au_train"] - recalls["reservee"], 4)
    return metrics


def _format(value: float | None) -> str:
    """Format a metric for the logs."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["SEGMENT_COLUMNS", "EntityEvaluator", "EvaluationResult"]
