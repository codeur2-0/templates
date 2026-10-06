"""Mesure d'un classifieur sur le split de test, références triviales comprises.

L'évaluateur ne réentraîne rien, ne règle rien et ne choisit rien : il applique le modèle aux
lignes du split de test, calcule les métriques, ventile les erreurs par segment et compare le
résultat à ce qu'un **tirage** obtiendrait.

Trois lectures sont publiées côte à côte, et c'est délibéré :

* la **moyenne** (``macro_f1``, ``accuracy``) ;
* la **ventilation** (par canal, par style rédactionnel, par classe) : une moyenne qui cache une
  classe jamais détectée est un chiffre trompeur ;
* les **pires erreurs**, avec leur confiance : une erreur commise à 0,95 de confiance n'a pas la
  même conséquence qu'une erreur à 0,40, et le rapport doit pouvoir les distinguer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.contract import BaseTextClassifier
from src.training.metrics import (
    classification_metrics,
    confusion_matrix_frame,
    majority_baseline,
    per_class_frame,
    percentile,
    stratified_baseline,
)
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Segment columns the evaluator breaks the metrics down by, when the corpus carries them.
SEGMENT_COLUMNS: tuple[str, ...] = ("source", "style", "priority")


@dataclass
class EvaluationResult:
    """Everything one evaluation run produced.

    Attributes:
        metrics: Metrics of the run (classification metrics, latencies, coverage).
        baselines: Metrics of the trivial references (majority class, stratified draw).
        per_class: Precision / recall / F1 / support per class.
        confusion: Confusion matrix (rows = reference, columns = prediction).
        segments: Metrics per segment value (channel, editorial style, priority).
        top_errors: Most confident mistakes, with the text and the expected label.
        predictions: One row per evaluated document (text, expected, predicted, confidence).
        verdict: ``conforme``, ``non conforme`` or ``indéterminé``.
        primary_metric: Name of the contractual metric the verdict is read on.
        threshold: Contractual minimum of the primary metric.
        n_documents: Number of evaluated documents.
        n_classes: Number of classes involved.
    """

    metrics: dict[str, float]
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: str = "indéterminé"
    primary_metric: str = "macro_f1"
    threshold: float | None = None
    n_documents: int = 0
    n_classes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (tables included, as records)."""
        return {
            "metrics": dict(self.metrics),
            "baselines": {name: dict(values) for name, values in self.baselines.items()},
            "per_class": self.per_class.to_dict(orient="records"),
            "confusion": {
                "labels": [str(label) for label in self.confusion.columns],
                "matrix": self.confusion.to_numpy().tolist(),
            },
            "segments": self.segments.to_dict(orient="records"),
            "top_errors": self.top_errors.to_dict(orient="records"),
            "verdict": self.verdict,
            "primary_metric": self.primary_metric,
            "threshold": self.threshold,
            "n_documents": self.n_documents,
            "n_classes": self.n_classes,
        }


class ClassificationEvaluator:
    """Apply a trained classifier to a labelled split and measure the result."""

    def __init__(
        self,
        model: BaseTextClassifier,
        *,
        config: Mapping[str, Any] | None = None,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        text_column: str = "text",
        target_column: str = "label",
        segment_columns: Sequence[str] = SEGMENT_COLUMNS,
        top_errors: int = 25,
    ) -> None:
        """Configure the evaluator.

        Args:
            model: Trained classifier to measure.
            config: Full application configuration (project identity, seed).
            metrics_config: ``metrics`` node (primary metric, contractual minimum, direction).
            paths: Project filesystem layout.
            text_column: Column holding the documents' text.
            target_column: Column holding the reference labels.
            segment_columns: Columns the metrics are broken down by, when present.
            top_errors: Number of mistakes archived in the error table.
        """
        self.model = model
        self.config = dict(config or {})
        self.metrics_config = dict(metrics_config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.text_column = str(text_column)
        self.target_column = str(target_column)
        self.segment_columns = tuple(segment_columns)
        self.top_errors_count = max(int(top_errors), 1)

    def evaluate(self, frame: pd.DataFrame) -> EvaluationResult:
        """Measure the classifier on a labelled frame.

        Args:
            frame: Labelled documents (usually the test split).

        Returns:
            The :class:`EvaluationResult` of the run.

        Raises:
            ValueError: When the frame is empty, or lacks the text or label column.
        """
        column = self.target_column
        if frame.empty:
            msg = "Cannot evaluate an empty frame: check the split sizes of the generator"
            raise ValueError(msg)
        for required in (self.text_column, column):
            if required not in frame.columns:
                msg = (
                    f"Column '{required}' missing from the evaluated frame: {sorted(frame.columns)}"
                )
                raise ValueError(msg)

        truth = frame[column].astype(str).to_numpy()
        texts = frame[self.text_column].astype(str).tolist()
        probabilities, latencies = self._predict_with_latency(texts)
        labels = self.model.labels
        predictions = np.asarray([labels[int(index)] for index in np.argmax(probabilities, axis=1)])
        confidences = probabilities.max(axis=1)

        metrics = classification_metrics(
            truth, predictions, confidences=confidences, labels=list(labels)
        )
        metrics["latency_p50_ms"] = percentile(latencies, 0.50)
        metrics["latency_p95_ms"] = percentile(latencies, 0.95)
        metrics["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["coverage"] = 1.0

        predictions_frame = pd.DataFrame(
            {
                "doc_id": frame.get("doc_id", pd.Series(range(len(frame)))).astype(str).to_numpy(),
                "text": texts,
                "expected_label": truth,
                "prediction": predictions,
                "confidence": confidences,
                "correct": (truth == predictions).astype("int64"),
                "latency_ms": np.asarray(latencies, dtype="float64"),
            }
        )
        segments = self._segments(frame, predictions_frame)
        top_errors = self._top_errors(predictions_frame)
        baselines = self._baselines(pd.Series(truth))
        primary = str(self.metrics_config.get("primary", "macro_f1"))
        threshold = self.metrics_config.get("min_primary")
        verdict, threshold = self._verdict(metrics, primary, threshold)
        logger.info(
            "Evaluation finished | {} documents | {}={} | {}",
            len(frame),
            primary,
            _format(metrics.get(primary)),
            verdict,
        )
        return EvaluationResult(
            metrics=metrics,
            baselines=baselines,
            per_class=per_class_frame(
                pd.Series(truth), pd.Series(predictions), labels=list(labels)
            ),
            confusion=confusion_matrix_frame(
                pd.Series(truth), pd.Series(predictions), labels=list(labels)
            ),
            segments=segments,
            top_errors=top_errors,
            predictions=predictions_frame,
            verdict=verdict,
            primary_metric=primary,
            threshold=None if threshold is None else float(threshold),
            n_documents=len(frame),
            n_classes=len(labels),
        )

    # ------------------------------------------------------------------ interne -----------
    def _predict_with_latency(self, texts: Sequence[str]) -> tuple[np.ndarray, list[float]]:
        """Score the texts one by one, measuring the latency of each call.

        Args:
            texts: Texts to classify.

        Returns:
            The probability matrix and the per-document latency, in milliseconds.

        Note:
            Les documents sont scorés **un par un** pour mesurer une latence de service : vectoriser
            un lot entier donnerait un débit, pas un temps de réponse.
        """
        import time

        rows: list[np.ndarray] = []
        latencies: list[float] = []
        for text in texts:
            clock = time.perf_counter()
            probability = self.model.predict_proba([text])
            latencies.append((time.perf_counter() - clock) * 1000.0)
            rows.append(probability[0])
        if not rows:
            return np.zeros((0, len(self.model.labels)), dtype="float64"), []
        return np.vstack(rows), latencies

    def _segments(self, frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """Break the metrics down by every declared segment column present in the frame."""
        rows: list[dict[str, Any]] = []
        for column in self.segment_columns:
            if column not in frame.columns:
                continue
            values = frame[column].astype(str).to_numpy()
            for value in sorted(set(values)):
                mask = values == value
                scores = classification_metrics(
                    predictions.loc[mask, "expected_label"],
                    predictions.loc[mask, "prediction"],
                    confidences=predictions.loc[mask, "confidence"].to_numpy(),
                    labels=list(self.model.labels),
                )
                rows.append(
                    {
                        "segment": f"{column}={value}",
                        "n_documents": int(mask.sum()),
                        "accuracy": scores.get("accuracy", 0.0),
                        "macro_f1": scores.get("macro_f1", 0.0),
                        "expected_calibration_error": scores.get("expected_calibration_error", 0.0),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "segment",
                    "n_documents",
                    "accuracy",
                    "macro_f1",
                    "expected_calibration_error",
                ]
            )
        return pd.DataFrame(rows).sort_values("n_documents", ascending=False).reset_index(drop=True)

    def _top_errors(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Return the most confident mistakes, worst first."""
        mistakes = predictions[predictions["correct"] == 0].copy()
        mistakes = mistakes.sort_values("confidence", ascending=False).head(self.top_errors_count)
        mistakes["text_preview"] = mistakes["text"].astype(str).str.slice(0, 160)
        return mistakes.drop(columns=["text"]).reset_index(drop=True)

    def _baselines(self, truth: pd.Series) -> dict[str, dict[str, float]]:
        """Measure the trivial references on the evaluated labels."""
        seed = int(self.config.get("seed", 42))
        return {
            "classe_majoritaire": majority_baseline(truth),
            "tirage_stratifie": stratified_baseline(truth, seed=seed),
        }

    def _verdict(
        self, metrics: Mapping[str, float], primary: str, threshold: Any
    ) -> tuple[str, float | None]:
        """Read the contractual verdict of the run."""
        if threshold is None:
            return "indéterminé", None
        observed = metrics.get(primary)
        if observed is None:
            return "indéterminé", float(threshold)
        return ("conforme" if float(observed) >= float(threshold) else "non conforme"), float(
            threshold
        )


def _format(value: float | None) -> str:
    """Format a metric for the logs."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["SEGMENT_COLUMNS", "ClassificationEvaluator", "EvaluationResult"]
