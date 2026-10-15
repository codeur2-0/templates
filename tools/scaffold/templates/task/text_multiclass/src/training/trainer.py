"""Entraînement supervisé d'un classifieur de texte, monitoré par les callbacks du socle.

Le déroulé est volontairement le même que celui de la modalité tabulaire — c'est ce qui rend les
deux projets comparables — mais il ne suppose rien de la représentation :

1. le *trainer* découpe le frame par sa colonne ``split`` (jamais à la main, jamais au hasard :
   le découpage est une donnée, il est écrit par le générateur et archivé avec le corpus) ;
2. il appelle :meth:`src.models.contract.BaseTextClassifier.fit` sur les lignes d'entraînement ;
3. il mesure le split de **validation** avec les métriques du projet et préfixe ses scores de
   ``val_`` ;
4. il déclenche ``on_epoch_end`` pour que l'early stopping et le contrôle de seuil voient les
   nombres, puis journalise un avertissement si la métrique principale passe sous le contrat ;
5. il rend un :class:`ClassificationOutcome` qui porte le résultat d'ajustement, les métriques et
   les erreurs les plus coûteuses du split de validation.

Le split de test n'apparaît nulle part ici : il est mesuré une fois, par le pipeline d'évaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import FitResult
from src.models.contract import BaseTextClassifier
from src.training.callbacks import BaseCallback, CallbackContext
from src.training.metrics import classification_metrics, per_class_frame
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationOutcome:
    """What one training run produced.

    Attributes:
        fit_result: Result returned by the model.
        metrics: Training metrics (plain names) plus the validation ones (``val_`` prefixed).
        history: Metric history, epoch by epoch (one entry for a single-shot fit).
        n_train_documents: Number of documents used for training.
        n_val_documents: Number of validation documents monitored.
        per_class: Per-class precision / recall / F1 on the validation split.
        top_errors: Most confident mistakes of the validation split.
    """

    fit_result: FitResult
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_train_documents: int = 0
    n_val_documents: int = 0
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_errors: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the run."""
        return {
            "fit_result": self.fit_result.to_dict(),
            "metrics": dict(self.metrics),
            "history": {key: list(values) for key, values in self.history.items()},
            "n_train_documents": self.n_train_documents,
            "n_val_documents": self.n_val_documents,
            "per_class": self.per_class.to_dict(orient="records"),
            "top_errors": self.top_errors.to_dict(orient="records"),
        }


class ClassificationTrainer:
    """Fit a classifier and monitor it on the validation split."""

    def __init__(
        self,
        model: BaseTextClassifier,
        *,
        config: Mapping[str, Any],
        text_column: str = "text",
        target_column: str = "label",
        split_column: str = "split",
        metric_names: Sequence[str] | None = None,
        primary_metric: str = "macro_f1",
        min_primary_metric: float | None = None,
        callbacks: Sequence[BaseCallback] | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            model: Classifier to fit.
            config: ``train`` node of the configuration.
            text_column: Column holding the documents' text.
            target_column: Column holding the labels.
            split_column: Column holding the split names (``train``, ``val``, ``test``).
            metric_names: Metrics requested by the manifest (documentation: every metric of the
                task is computed anyway, they are cheap).
            primary_metric: Name of the metric the contract is read on.
            min_primary_metric: Contractual minimum of the primary metric (a warning, not a hard
                failure: the report is what decides compliance).
            callbacks: Callbacks to fire.
        """
        self.model = model
        self.config = dict(config)
        self.text_column = str(text_column)
        self.target_column = str(target_column)
        self.split_column = str(split_column)
        self.metric_names = tuple(metric_names or ())
        self.primary_metric = str(primary_metric)
        self.min_primary_metric = min_primary_metric
        self.callbacks = list(callbacks or [])

    def run(
        self, documents: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> ClassificationOutcome:
        """Fit the model on ``train`` rows and measure the validation rows.

        Args:
            documents: Labelled corpus (the trainer keeps the ``train`` rows).
            validation: Validation frame (defaults to the ``val`` rows of ``documents``).

        Returns:
            The :class:`ClassificationOutcome` of the run.
        """
        train = self.split(documents, "train")
        val = validation if validation is not None else self.split(documents, "val")
        context = CallbackContext(
            model_name=type(self.model).__name__,
            params=self.model._effective_params(),
            epochs=int(self.config.get("epochs", 1)),
        )
        self._fire("on_train_begin", context)
        fit_result = self.model.fit(
            train, target=self.target_column, callbacks=self.callbacks, context=context
        )
        metrics = dict(fit_result.metrics)
        per_class = pd.DataFrame()
        top_errors = pd.DataFrame()
        if val is not None and not val.empty:
            per_class, top_errors, validation_metrics = self.evaluate(val)
            metrics.update(validation_metrics)
        context.logs = dict(metrics)
        context.best_score = metrics.get(f"val_{self.primary_metric}")
        context.best_epoch = 0
        self._fire("on_epoch_end", context)
        self._warn_on_threshold(metrics)
        self._fire("on_train_end", context)
        logger.info(
            "Training done | {} | train={} docs | val={} docs | val_{}={}",
            self.model.summary(),
            len(train),
            len(val),
            self.primary_metric,
            _format(metrics.get(f"val_{self.primary_metric}")),
        )
        return ClassificationOutcome(
            fit_result=fit_result,
            metrics=metrics,
            history={key: [value] for key, value in metrics.items()},
            n_train_documents=int(len(train)),
            n_val_documents=int(len(val)),
            per_class=per_class,
            top_errors=top_errors,
        )

    def evaluate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        """Score a frame and return the metrics, the per-class table and the worst mistakes.

        Args:
            frame: Labelled frame to score.

        Returns:
            The per-class frame, the most confident errors, and the metrics prefixed with ``val_``.
        """
        if frame.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
        truth = frame[self.target_column].astype(str).to_numpy()
        probabilities = self.model.predict_proba(frame[self.text_column].astype(str).tolist())
        labels = self.model.labels
        predictions = np.asarray([labels[int(index)] for index in np.argmax(probabilities, axis=1)])
        confidences = probabilities.max(axis=1)
        scores = classification_metrics(
            truth, predictions, confidences=confidences, labels=list(labels)
        )
        metrics = {f"val_{key}": float(value) for key, value in scores.items()}
        metrics["val_latency_p50_ms"] = 0.0
        per_class = per_class_frame(
            pd.Series(truth), pd.Series(predictions), labels=list(labels)
        )
        errors = frame.loc[:, [self.text_column, self.target_column]].copy()
        errors["prediction"] = predictions
        errors["confidence"] = confidences
        mistakes = errors[errors[self.target_column].astype(str) != errors["prediction"]]
        mistakes = mistakes.sort_values("confidence", ascending=False).head(20)
        return per_class, mistakes.reset_index(drop=True), metrics

    def split(self, documents: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return the rows belonging to a split.

        Args:
            documents: Labelled corpus.
            name: Split name (``train``, ``val``, ``calibration``, ``test``).

        Returns:
            The matching rows (the whole frame when it carries no split column — a single-split
            corpus stays legitimate for a quick experiment).
        """
        if self.split_column not in documents.columns:
            return documents
        return documents[documents[self.split_column].astype(str) == name]

    # ------------------------------------------------------------------ interne -----------
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


__all__ = ["ClassificationOutcome", "ClassificationTrainer"]
