"""Classification métriques, écrites à la main pour être lisibles.

Un projet pédagogique qui appelle ``sklearn.metrics.classification_report`` apprend à lire une
sortie, pas une définition. Les métriques de cette couche sont donc implémentées ici, en NumPy,
avec la définition en clair dans la docstring de chaque fonction :

* ``accuracy`` — la part de documents bien classés (trompeuse dès que les classes sont
  déséquilibrées : c'est pourquoi la métrique principale de la famille est ``macro_f1``) ;
* ``macro_f1`` — la F1 moyenne **par classe**, chaque classe pesant le même poids quelle que soit
  sa fréquence : la seule moyenne qui ne récompense pas un modèle qui ignore la classe rare ;
* ``balanced_accuracy`` — le rappel moyen par classe ;
* ``cohen_kappa`` — l'accord au-delà de ce qu'un tirage au hasard produirait ;
* ``expected_calibration_error`` — l'écart entre la confiance annoncée et l'exactitude observée.

Les références triviales (classe majoritaire, tirage stratifié) sont fournies par le même module :
un score de F1 macro ne veut rien dire sans le niveau à battre.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Metrics exposed by the project, in reporting order.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "cohen_kappa",
    "mean_confidence",
    "expected_calibration_error",
)

#: Human readable description of every metric (rendered in the README and the report).
METRIC_DESCRIPTIONS: dict[str, str] = {
    "accuracy": "Part des documents correctement classés, toutes classes confondues.",
    "macro_f1": "F1 moyenne par classe, chaque classe pesant le même poids (métrique principale).",
    "weighted_f1": "F1 moyenne pondérée par l'effectif de chaque classe (lecture « volume »).",
    "balanced_accuracy": "Rappel moyen par classe : insensible au déséquilibre des effectifs.",
    "macro_precision": "Précision moyenne par classe : ce qu'une alerte coûte en faux positifs.",
    "macro_recall": "Rappel moyen par classe : ce qu'une classe rare paie en documents manqués.",
    "cohen_kappa": "Accord au-delà du hasard (0 = tirage, 1 = accord parfait).",
    "mean_confidence": "Confiance moyenne des prédictions (probabilité de la classe prédite).",
    "expected_calibration_error": (
        "Écart moyen entre la confiance annoncée et l'exactitude observée."
    ),
    "latency_p50_ms": "Latence médiane d'une prédiction, en millisecondes.",
    "latency_p95_ms": "Latence au 95ᵉ centile d'une prédiction, en millisecondes.",
    "coverage": "Part des documents classés (la classification ne s'abstient pas : 1,0).",
}

#: Metrics by task. The evaluator iterates over the task's list, so a new metric is a one-line
#: change here.
METRICS_BY_TASK: dict[str, tuple[str, ...]] = {
    "multiclass": CLASSIFICATION_METRICS,
    "binary": CLASSIFICATION_METRICS,
}


def _labels_from(*frames: pd.Series) -> list[str]:
    """Collect every label seen in the given series, sorted.

    Args:
        *frames: Series of labels (strings).

    Returns:
        The sorted list of distinct labels.
    """
    seen: set[str] = set()
    for frame in frames:
        seen.update(str(value) for value in np.asarray(frame).ravel())
    return sorted(seen)


def _contingency(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> np.ndarray:
    """Build the confusion matrix of two label arrays.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        labels: Label order (rows = reference, columns = prediction).

    Returns:
        An integer ``(n_labels, n_labels)`` matrix.
    """
    positions = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype="int64")
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        matrix[positions[truth], positions[prediction]] += 1
    return matrix


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning ``0.0`` when the denominator is zero."""
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _complete_labels(labels: Sequence[str], *frames: pd.Series) -> list[str]:
    """Complete a label list with the labels observed in the frames.

    Un modèle ajusté sur quatre classes peut rencontrer six classes à l'évaluation : ignorer les
    deux classes qu'il ne connaît pas flatterait sa F1 macro (elles sortiraient du dénominateur).
    Elles sont donc ajoutées — avec un rappel nul, puisque le modèle ne peut pas les prédire — et la
    faiblesse devient lisible au lieu d'être invisible.

    Args:
        labels: Requested label order.
        *frames: Label series to inspect.

    Returns:
        The requested labels, followed by the observed labels that were missing.
    """
    ordered = [str(label) for label in labels]
    seen = set(ordered)
    for frame in frames:
        for value in pd.Series(frame).astype(str).unique():
            key = str(value)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def per_class_frame(
    y_true: pd.Series, y_pred: pd.Series, *, labels: list[str] | None = None
) -> pd.DataFrame:
    """Precision, recall, F1 and support for every class.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        labels: Label order (defaults to the union of both series, sorted).

    Returns:
        One row per class, ordered by decreasing support.
    """
    truth = np.asarray([str(value) for value in y_true])
    prediction = np.asarray([str(value) for value in y_pred])
    order = _complete_labels(labels, y_true, y_pred) if labels else _labels_from(y_true, y_pred)
    matrix = _contingency(truth, prediction, order)
    rows: list[dict[str, float | str]] = []
    for position, label in enumerate(order):
        true_positive = float(matrix[position, position])
        support = float(matrix[position, :].sum())
        predicted = float(matrix[:, position].sum())
        precision = _safe_div(true_positive, predicted)
        recall = _safe_div(true_positive, support)
        rows.append(
            {
                "class": label,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2.0 * precision * recall, precision + recall),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("support", ascending=False).reset_index(drop=True)


def confusion_matrix_frame(
    y_true: pd.Series, y_pred: pd.Series, *, labels: list[str] | None = None
) -> pd.DataFrame:
    """Confusion matrix, readable: rows are the reference labels, columns the predictions.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        labels: Label order (defaults to the union of both series, sorted).

    Returns:
        A frame indexed by reference label.
    """
    truth = np.asarray([str(value) for value in y_true])
    prediction = np.asarray([str(value) for value in y_pred])
    order = _complete_labels(labels, y_true, y_pred) if labels else _labels_from(y_true, y_pred)
    matrix = _contingency(truth, prediction, order)
    return pd.DataFrame(matrix, index=order, columns=order)


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, *, bins: int = 10
) -> float:
    """Measure the gap between announced confidence and observed accuracy.

    Args:
        confidences: Confidence of each prediction, in ``[0, 1]``.
        correctness: ``1.0`` when the prediction was right, ``0.0`` otherwise.
        bins: Number of confidence buckets.

    Returns:
        The support-weighted mean absolute gap, in ``[0, 1]``. A model that announces 0.9 and is
        right 60 % of the time on that bucket has a calibration error of 0.3.
    """
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(confidences))
    error = 0.0
    for lower, upper in itertools.pairwise(edges):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        weight = float(mask.sum()) / total
        error += weight * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(error)


def classification_metrics(
    y_true: pd.Series | list[str],
    y_pred: pd.Series | list[str],
    *,
    confidences: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Compute every classification metric of the project.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        confidences: Confidence of each prediction (enables the calibration metrics).
        labels: Label order (defaults to the union of both label sets, sorted).

    Returns:
        The metrics, keyed by name. Empty input returns an empty mapping rather than raising: an
        empty split is a legitimate configuration, and a metric computed on nothing would lie.
    """
    truth = np.asarray([str(value) for value in y_true])
    prediction = np.asarray([str(value) for value in y_pred])
    if len(truth) == 0:
        return {}
    order = labels or _labels_from(pd.Series(truth), pd.Series(prediction))
    per_class = per_class_frame(pd.Series(truth), pd.Series(prediction), labels=order)
    accuracy = float(np.mean(truth == prediction))
    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(per_class["f1"].mean()),
        "weighted_f1": float(
            np.average(per_class["f1"], weights=np.maximum(per_class["support"], 1e-9))
        ),
        "balanced_accuracy": float(per_class["recall"].mean()),
        "macro_precision": float(per_class["precision"].mean()),
        "macro_recall": float(per_class["recall"].mean()),
    }
    metrics["cohen_kappa"] = _cohen_kappa(truth, prediction, order)
    if confidences is not None and len(confidences) == len(truth):
        correctness = (truth == prediction).astype("float64")
        metrics["mean_confidence"] = float(np.mean(confidences))
        metrics["expected_calibration_error"] = expected_calibration_error(confidences, correctness)
    return metrics


def _cohen_kappa(truth: np.ndarray, prediction: np.ndarray, labels: list[str]) -> float:
    """Compute Cohen's kappa from two label arrays.

    Args:
        truth: Reference labels.
        prediction: Predicted labels.
        labels: Label order.

    Returns:
        The agreement beyond chance, in ``[-1, 1]``.
    """
    matrix = _contingency(truth, prediction, labels)
    total = float(matrix.sum())
    if total == 0:
        return 0.0
    observed = float(np.trace(matrix)) / total
    expected = float(
        np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / (total * total)
    )  # rows = truth, columns = prediction
    return _safe_div(observed - expected, 1.0 - expected)


def majority_baseline(labels: pd.Series) -> dict[str, float]:
    """Measure the trivial strategy that always answers the most frequent class.

    Args:
        labels: Training labels.

    Returns:
        Accuracy and macro-F1 of the constant prediction, as lower bounds of the task.
    """
    values = [str(value) for value in labels]
    if not values:
        return {}
    counts = pd.Series(values).value_counts()
    majority = str(counts.index[0])
    return classification_metrics(values, [majority] * len(values))


def stratified_baseline(
    labels: pd.Series, *, seed: int = 42, repetitions: int = 5
) -> dict[str, float]:
    """Measure the trivial strategy that draws a label with its training frequency.

    Args:
        labels: Reference labels of the evaluated split.
        seed: Seed of the draw (the baseline must be reproducible).
        repetitions: Number of draws averaged (one draw is noise).

    Returns:
        Accuracy and macro-F1 of the random draw, averaged over the repetitions.
    """
    values = [str(value) for value in labels]
    if not values:
        return {}
    frequencies = pd.Series(values).value_counts(normalize=True)
    generator = np.random.default_rng(int(seed))
    collected: dict[str, list[float]] = {}
    for _ in range(max(int(repetitions), 1)):
        draw = generator.choice(
            frequencies.index.to_numpy(), size=len(values), p=frequencies.to_numpy()
        )
        for key, value in classification_metrics(values, list(draw)).items():
            collected.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in collected.items()}


def metrics_for_task(task: str) -> tuple[str, ...]:
    """Return the metrics of a task.

    Args:
        task: Learning task (``multiclass``, ``binary``).

    Returns:
        The metric names, in reporting order.
    """
    return METRICS_BY_TASK.get(str(task), CLASSIFICATION_METRICS)


def describe_metrics(names: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Return the description of the requested metrics.

    Args:
        names: Metric names (defaults to every known metric).

    Returns:
        A mapping of metric name to its French description.
    """
    selected = list(names) if names else sorted(METRIC_DESCRIPTIONS)
    return {name: METRIC_DESCRIPTIONS.get(name, "") for name in selected}


def flatten_metrics(payload: dict[str, object], *, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics payload into finite numbers.

    Args:
        payload: Nested mapping (``{"val": {"accuracy": 0.9}}``).
        prefix: Prefix prepended to each key.

    Returns:
        A flat mapping holding only the finite numeric leaves, which is what a report can print
        and a JSON file can store without surprising its reader.
    """
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(dict(value), prefix=f"{name}_"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                flat[name] = number
    return flat


def percentile(values: list[float] | tuple[float, ...] | np.ndarray, quantile: float) -> float:
    """Return a percentile of a sample (linear interpolation, no SciPy dependency).

    Args:
        values: Sample.
        quantile: Quantile in ``[0, 1]``.

    Returns:
        The interpolated percentile, ``0.0`` for an empty sample.
    """
    sample = np.asarray([float(value) for value in values], dtype="float64")
    if sample.size == 0:
        return 0.0
    return float(np.quantile(sample, float(np.clip(quantile, 0.0, 1.0))))


__all__ = [
    "CLASSIFICATION_METRICS",
    "METRICS_BY_TASK",
    "METRIC_DESCRIPTIONS",
    "classification_metrics",
    "confusion_matrix_frame",
    "describe_metrics",
    "expected_calibration_error",
    "flatten_metrics",
    "majority_baseline",
    "metrics_for_task",
    "per_class_frame",
    "percentile",
    "stratified_baseline",
]
