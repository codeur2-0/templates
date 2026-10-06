"""Entraînement d'un extracteur d'entités, monitoré par les callbacks du socle.

Le déroulé est volontairement le même que celui des autres tâches — c'est ce qui rend les projets
comparables — mais il porte sur **deux tables** et il mesure au niveau entité :

1. le *trainer* découpe les messages par leur colonne ``split`` (jamais à la main, jamais au
hasard :
   le découpage est écrit par le générateur et archivé avec le corpus) et ne garde que les
   annotations des messages retenus ;
2. il appelle :meth:`src.models.contract.BaseEntityTagger.fit` sur les messages et les annotations
   d'entraînement ;
3. il mesure le split de **validation** mention par mention : un span est correct si le type *et*
les
   bornes sont exacts, et les scores sont préfixés ``val_`` ;
4. il déclenche ``on_epoch_end`` (historique, seuil, arrêt anticipé) puis avertit si la métrique
   principale passe sous le contrat ;
5. il rend un :class:`EntityTrainingOutcome` qui porte le résultat d'ajustement, les métriques, la
   table par type et les erreurs les plus instructives.

Les prédictions du split de validation sont produites **message par message**, comme un service
les
reçoit : la latence publiée est donc un temps de réponse, pas un débit de lot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.schemas import ENTITY_LABELS
from src.inference.extraction import extract_one_by_one
from src.models.base import FitResult
from src.models.contract import BaseEntityTagger
from src.training.callbacks import BaseCallback, CallbackContext
from src.training.metrics import (
    confidence_gap,
    entity_scores,
    error_frame,
    latency_stats,
    partial_scores,
    per_label_frame,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EntityTrainingOutcome:
    """What one training run produced.

    Attributes:
        fit_result: Result returned by the model.
        metrics: Training metrics (plain names) plus the validation ones (``val_`` prefixed).
        history: Per-epoch training history (empty for a rule-only model, which has no epochs).
        n_train_documents: Number of messages used for training.
        n_val_documents: Number of validation messages monitored.
        n_train_entities: Number of annotated entities seen during training.
        n_val_entities: Number of reference entities of the validation split.
        per_label: Per-type precision / recall / F1 on the validation split.
        top_errors: Most instructive mistakes of the validation split.
    """

    fit_result: FitResult
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_train_documents: int = 0
    n_val_documents: int = 0
    n_train_entities: int = 0
    n_val_entities: int = 0
    per_label: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_errors: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the run."""
        return {
            "fit_result": self.fit_result.to_dict(),
            "metrics": dict(self.metrics),
            "history": {key: list(values) for key, values in self.history.items()},
            "n_train_documents": self.n_train_documents,
            "n_val_documents": self.n_val_documents,
            "n_train_entities": self.n_train_entities,
            "n_val_entities": self.n_val_entities,
            "per_label": self.per_label.to_dict(orient="records"),
            "top_errors": self.top_errors.to_dict(orient="records"),
        }


class EntityTrainer:
    """Fit an entity extractor and monitor it on the validation split."""

    def __init__(
        self,
        model: BaseEntityTagger,
        *,
        config: Mapping[str, Any],
        text_column: str = "text",
        target_column: str = "label",
        id_column: str = "msg_id",
        split_column: str = "split",
        metric_names: Sequence[str] | None = None,
        primary_metric: str = "entity_f1",
        min_primary_metric: float | None = None,
        callbacks: Sequence[BaseCallback] | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            model: Extractor to fit.
            config: ``train`` node of the configuration.
            text_column: Column holding the messages' text.
            target_column: Column holding the entity type in the annotation table.
            id_column: Join key between the two tables.
            split_column: Column holding the split names (``train``, ``val``, ``test``).
            metric_names: Metrics requested by the manifest (documentation: every metric of the
            task
                is computed anyway, they are cheap).
            primary_metric: Name of the metric the contract is read on.
            min_primary_metric: Contractual minimum of the primary metric (a warning, not a hard
                failure: the evaluation report is what decides compliance).
            callbacks: Callbacks to fire.
        """
        self.model = model
        self.config = dict(config)
        self.text_column = str(text_column)
        self.target_column = str(target_column)
        self.id_column = str(id_column)
        self.split_column = str(split_column)
        self.metric_names = tuple(metric_names or ())
        self.primary_metric = str(primary_metric)
        self.min_primary_metric = min_primary_metric
        self.callbacks = list(callbacks or [])

    def run(
        self,
        documents: pd.DataFrame,
        spans: pd.DataFrame,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> EntityTrainingOutcome:
        """Fit the model on ``train`` messages and measure the validation ones.

        Args:
            documents: Annotated corpus (the trainer keeps the ``train`` messages).
            spans: Annotations of the corpus.
            validation: Optional validation pair (defaults to the ``val`` split of ``documents``).

        Returns:
            The :class:`EntityTrainingOutcome` of the run.

        Raises:
            ValueError: When the training split holds no annotated entity.
        """
        train_documents = self.split(documents, "train")
        train_spans = self.annotations_of(train_documents, spans)
        if train_spans.empty:
            msg = (
                "No annotated entity in the training split: nothing to learn "
                f"({len(train_documents)} messages, {len(spans)} annotations in the corpus)"
            )
            raise ValueError(msg)
        val_documents, val_spans = (
            validation if validation is not None else self.validation_split(documents, spans)
        )
        context = CallbackContext(
            model_name=type(self.model).__name__,
            params=self.model._effective_params(),
            epochs=int(self.config.get("epochs", 1)),
        )
        self._fire("on_train_begin", context)
        fit_result = self.model.fit(
            train_documents,
            train_spans,
            callbacks=self.callbacks,
            context=context,
        )
        metrics = dict(fit_result.metrics)
        per_label = pd.DataFrame()
        top_errors = pd.DataFrame()
        if not val_documents.empty:
            per_label, top_errors, validation_metrics = self.evaluate(val_documents, val_spans)
            metrics.update(validation_metrics)
        history = {key: list(values) for key, values in fit_result.history.items()}
        history.setdefault("epoch", list(range(1, len(fit_result.history.get("loss", [])) + 1)))
        context.logs = dict(metrics)
        context.best_score = metrics.get(f"val_{self.primary_metric}")
        context.best_epoch = max(len(history.get("loss", [])) - 1, 0)
        self._fire("on_epoch_end", context)
        self._warn_on_threshold(metrics)
        self._fire("on_train_end", context)
        logger.info(
            "Training done | {} | train={} messages | val={} messages | val_{}={}",
            self.model.summary(),
            len(train_documents),
            len(val_documents),
            self.primary_metric,
            _format(metrics.get(f"val_{self.primary_metric}")),
        )
        return EntityTrainingOutcome(
            fit_result=fit_result,
            metrics=metrics,
            history=history,
            n_train_documents=len(train_documents),
            n_val_documents=len(val_documents),
            n_train_entities=len(train_spans),
            n_val_entities=len(val_spans),
            per_label=per_label,
            top_errors=top_errors,
        )

    def evaluate(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        """Score a split and return the metrics, the per-type table and the worst mistakes.

        Args:
            documents: Messages of the split.
            spans: Reference annotations of the split.

        Returns:
            The per-type frame, the mistake table, and the metrics prefixed with ``val_``.
        """
        if documents.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
        predicted_frame, latencies = self.predict_one_by_one(documents)
        gold = self._gold_sets(documents, spans)
        predicted = self._predicted_sets(documents, predicted_frame)
        scores = self._score(gold, predicted, predicted_frame, spans, latencies)
        metrics = {f"val_{key}": float(value) for key, value in scores.items()}
        per_label = per_label_frame(
            gold, predicted, labels=list(self.model.labels or ENTITY_LABELS)
        )
        errors = error_frame(documents, spans, predicted_frame, limit=25)
        return per_label, errors, metrics

    def predict_one_by_one(self, documents: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
        """Extract the mentions of a frame, one message at a time, and time each call.

        Args:
            documents: Messages to annotate.

        Returns:
            The prediction table and the latency of every message, in milliseconds.
        """
        return extract_one_by_one(
            self.model,
            documents,
            text_column=self.text_column,
            id_column=self.id_column,
        )

    def split(self, documents: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return the messages belonging to a split.

        Args:
            documents: Annotated corpus.
            name: Split name (``train``, ``val``, ``calibration``, ``test``).

        Returns:
            The matching messages (the whole frame when it carries no split column — a
            single-split
            corpus stays legitimate for a quick experiment).
        """
        if self.split_column not in documents.columns:
            return documents
        return documents[documents[self.split_column].astype(str) == name].reset_index(drop=True)

    def annotations_of(self, documents: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
        """Return the annotations of the given messages.

        Args:
            documents: Messages selected by a split.
            spans: Annotation table of the corpus.

        Returns:
            The annotations of those messages, sorted by message and offset.

        Raises:
            ValueError: When the frame is not an annotation table (the join key or the offsets are
                missing), so the caller learns what is wrong instead of reading a ``KeyError``.
        """
        required = (self.id_column, "start", "end")
        missing = [name for name in required if name not in spans.columns]
        if missing:
            msg = f"Annotation table is missing the columns {missing}: {sorted(spans.columns)}"
            raise ValueError(msg)
        keep = set(documents[self.id_column].astype(str))
        subset = spans[spans[self.id_column].astype(str).isin(keep)]
        return subset.sort_values([self.id_column, "start", "end"]).reset_index(drop=True)

    def validation_split(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the validation pair, falling back to the whole corpus when it has no split.

        Args:
            documents: Annotated corpus.
            spans: Annotations of the corpus.

        Returns:
            The validation messages and their annotations.
        """
        if self.split_column not in documents.columns:
            return documents, spans
        selected = self.split(documents, "val")
        return selected, self.annotations_of(selected, spans)

    # ------------------------------------------------------------------ interne -----------
    def _score(
        self,
        gold: list[set[tuple[int, int, str]]],
        predicted: list[set[tuple[int, int, str]]],
        predicted_frame: pd.DataFrame,
        spans: pd.DataFrame,
        latencies: Sequence[float],
    ) -> dict[str, float]:
        """Assemble every entity metric of a scored split."""
        scores = entity_scores(gold, predicted, labels=list(self.model.labels or ENTITY_LABELS))
        scores.update(
            partial_scores(gold, predicted, labels=list(self.model.labels or ENTITY_LABELS))
        )
        scores.update(latency_stats(latencies))
        if not predicted_frame.empty:
            scores["mean_confidence"] = round(
                float(predicted_frame["confidence"].astype(float).mean()), 4
            )
            scores["share_from_rules"] = round(
                float((predicted_frame["source"].astype(str) == "regle").mean()), 4
            )
        else:
            scores["mean_confidence"] = 0.0
            scores["share_from_rules"] = 0.0
        scores["confidence_gap"] = confidence_gap(predicted_frame, spans)
        return scores

    def _gold_sets(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> list[set[tuple[int, int, str]]]:
        """Return the reference spans, one set per message of the split (order preserved)."""
        grouped: dict[str, set[tuple[int, int, str]]] = {}
        for row in spans.itertuples(index=False):
            grouped.setdefault(str(getattr(row, self.id_column)), set()).add(
                (int(row.start), int(row.end), str(getattr(row, self.target_column)))
            )
        return [grouped.get(str(identifier), set()) for identifier in documents[self.id_column]]

    def _predicted_sets(
        self, documents: pd.DataFrame, predicted_frame: pd.DataFrame
    ) -> list[set[tuple[int, int, str]]]:
        """Return the predicted spans, one set per message of the split (order preserved)."""
        grouped: dict[str, set[tuple[int, int, str]]] = {}
        if not predicted_frame.empty:
            for row in predicted_frame.itertuples(index=False):
                grouped.setdefault(str(row.msg_id), set()).add(
                    (int(row.start), int(row.end), str(row.label))
                )
        return [grouped.get(str(identifier), set()) for identifier in documents[self.id_column]]

    def _fire(self, hook: str, context: CallbackContext) -> None:
        """Call a hook on every callback, in order."""
        for callback in self.callbacks:
            getattr(callback, hook)(context)

    def _warn_on_threshold(self, metrics: Mapping[str, float]) -> None:
        """Warn when the primary metric misses the contractual minimum."""
        if self.min_primary_metric is None:
            return
        observed = metrics.get(f"val_{self.primary_metric}")
        if observed is None:
            logger.warning(
                "Metric '{}' not computed: the contractual minimum cannot be checked",
                self.primary_metric,
            )
        elif float(observed) < float(self.min_primary_metric):
            logger.warning(
                "Primary metric {}={} is below the contractual minimum {}",
                self.primary_metric,
                _format(float(observed)),
                self.min_primary_metric,
            )


def _format(value: float | None) -> str:
    """Format a metric for the logs."""
    return "n/a" if value is None else f"{value:.4f}"


def score_arrays(
    gold: Sequence[set[tuple[int, int, str]]], predicted: Sequence[set[tuple[int, int, str]]]
) -> dict[str, float]:
    """Score two aligned arrays of span sets (used by the notebooks).

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message.

    Returns:
        The entity-level scores and the partial ones, flattened.
    """
    scores = entity_scores(gold, predicted)
    scores.update(partial_scores(gold, predicted))
    return {key: float(value) for key, value in scores.items()}


__all__ = ["EntityTrainer", "EntityTrainingOutcome", "score_arrays"]
