"""Metric registry and loss resolution.

One place defines *what* is measured; the evaluator only asks for metric names. Adding a
metric means registering it here, nothing else (Open/Closed Principle).

Each metric declares:

* ``fn``: the implementation, called with keyword arguments only,
* ``requires``: the inputs it needs (``y_true``, ``y_proba``, ``X``, ``groups``),
* ``higher_is_better``: used to pick the best model and to orient the reports,
* ``tasks``: the learning tasks where the metric is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from src.utils.logging import get_logger
from src.utils.utils import safe_division

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Declaration of one metric."""

    name: str
    fn: Callable[..., float]
    requires: frozenset[str] = frozenset({"y_true", "y_pred"})
    higher_is_better: bool = True
    tasks: frozenset[str] = frozenset()
    description: str = ""


@dataclass(frozen=True, slots=True)
class MetricInputs:
    """Everything a metric may need.

    Attributes:
        y_true: Ground truth (labels, values or relevance).
        y_pred: Model predictions.
        y_proba: Class probabilities or scores.
        X: Feature matrix (silhouette, reconstruction error, ...).
        groups: Group identifiers (per-user ranking metrics).
        extra: Free-form payload (e.g. ``naive_pred`` for MASE).
    """

    y_true: Any = None
    y_pred: Any = None
    y_proba: Any = None
    X: Any = None
    groups: Any = None
    extra: Mapping[str, Any] | None = None


# ---------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------
def _accuracy(inputs: MetricInputs) -> float:
    """Accuracy (balanced over classes for imbalanced problems: see balanced_accuracy)."""
    return float(sk_metrics.accuracy_score(inputs.y_true, inputs.y_pred))


def _balanced_accuracy(inputs: MetricInputs) -> float:
    """Mean recall over classes."""
    return float(sk_metrics.balanced_accuracy_score(inputs.y_true, inputs.y_pred))


def _precision_macro(inputs: MetricInputs) -> float:
    """Macro-averaged precision."""
    return float(
        sk_metrics.precision_score(inputs.y_true, inputs.y_pred, average="macro", zero_division=0)
    )


def _recall_macro(inputs: MetricInputs) -> float:
    """Macro-averaged recall."""
    return float(
        sk_metrics.recall_score(inputs.y_true, inputs.y_pred, average="macro", zero_division=0)
    )


def _f1_macro(inputs: MetricInputs) -> float:
    """Macro-averaged F1."""
    return float(
        sk_metrics.f1_score(inputs.y_true, inputs.y_pred, average="macro", zero_division=0)
    )


def _f1_weighted(inputs: MetricInputs) -> float:
    """Support-weighted F1."""
    return float(
        sk_metrics.f1_score(inputs.y_true, inputs.y_pred, average="weighted", zero_division=0)
    )


def _f1_binary(inputs: MetricInputs) -> float:
    """F1 on the positive class."""
    return float(sk_metrics.f1_score(inputs.y_true, inputs.y_pred, zero_division=0))


def _precision_binary(inputs: MetricInputs) -> float:
    """Precision on the positive class."""
    return float(sk_metrics.precision_score(inputs.y_true, inputs.y_pred, zero_division=0))


def _recall_binary(inputs: MetricInputs) -> float:
    """Recall on the positive class."""
    return float(sk_metrics.recall_score(inputs.y_true, inputs.y_pred, zero_division=0))


def _roc_auc(inputs: MetricInputs) -> float:
    """ROC AUC, computed from probabilities (or scores)."""
    scores = _positive_class_scores(inputs)
    labels = np.asarray(inputs.y_true)
    if len(np.unique(labels)) < 2:
        logger.warning("roc_auc undefined: a single class is present in the evaluation split")
        return float("nan")
    return float(sk_metrics.roc_auc_score(labels, scores))


def _pr_auc(inputs: MetricInputs) -> float:
    """Average precision (area under the precision-recall curve)."""
    scores = _positive_class_scores(inputs)
    labels = np.asarray(inputs.y_true)
    if len(np.unique(labels)) < 2:
        logger.warning("pr_auc undefined: a single class is present in the evaluation split")
        return float("nan")
    return float(sk_metrics.average_precision_score(labels, scores))


def _log_loss(inputs: MetricInputs) -> float:
    """Cross-entropy loss (requires probabilities)."""
    probabilities = inputs.y_proba
    if probabilities is None:
        logger.warning("log_loss requires probabilities; returning NaN")
        return float("nan")
    labels = np.unique(np.asarray(inputs.y_true))
    return float(sk_metrics.log_loss(inputs.y_true, probabilities, labels=labels))


def _mcc(inputs: MetricInputs) -> float:
    """Matthews correlation coefficient (robust to class imbalance)."""
    return float(sk_metrics.matthews_corrcoef(inputs.y_true, inputs.y_pred))


def _positive_class_scores(inputs: MetricInputs) -> np.ndarray:
    """Return the score used for ranking metrics (probability of the positive class)."""
    if inputs.y_proba is not None:
        probabilities = np.asarray(inputs.y_proba)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.ravel()
    if inputs.y_pred is not None:
        return np.asarray(inputs.y_pred).ravel().astype("float64")
    msg = "Ranking metrics need either probabilities or predictions"
    raise ValueError(msg)


# ---------------------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------------------
def _rmse(inputs: MetricInputs) -> float:
    """Root mean squared error."""
    return float(np.sqrt(sk_metrics.mean_squared_error(inputs.y_true, inputs.y_pred)))


def _mae(inputs: MetricInputs) -> float:
    """Mean absolute error."""
    return float(sk_metrics.mean_absolute_error(inputs.y_true, inputs.y_pred))


def _mape(inputs: MetricInputs) -> float:
    """Mean absolute percentage error (ignores zero ground truth)."""
    truth = np.asarray(inputs.y_true, dtype="float64")
    predicted = np.asarray(inputs.y_pred, dtype="float64")
    mask = np.abs(truth) > 1e-8
    if not mask.any():
        logger.warning("mape undefined: every ground truth value is zero")
        return float("nan")
    return float(np.mean(np.abs((truth[mask] - predicted[mask]) / truth[mask])) * 100.0)


def _smape(inputs: MetricInputs) -> float:
    """Symmetric MAPE, bounded to [0, 200] %."""
    truth = np.asarray(inputs.y_true, dtype="float64")
    predicted = np.asarray(inputs.y_pred, dtype="float64")
    denominator = (np.abs(truth) + np.abs(predicted)) / 2.0
    mask = denominator > 1e-8
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(truth[mask] - predicted[mask]) / denominator[mask]) * 100.0)


def _r2(inputs: MetricInputs) -> float:
    """Coefficient of determination."""
    return float(sk_metrics.r2_score(inputs.y_true, inputs.y_pred))


def _max_error(inputs: MetricInputs) -> float:
    """Largest absolute error (worst case)."""
    return float(sk_metrics.max_error(inputs.y_true, inputs.y_pred))


def _mase(inputs: MetricInputs) -> float:
    """Mean absolute scaled error, using the naive forecast as the denominator.

    The naive forecast is provided through ``inputs.extra['naive_pred']``; when it is absent a
    seasonal-naive proxy (shift of the ground truth) is used.
    """
    truth = np.asarray(inputs.y_true, dtype="float64")
    predicted = np.asarray(inputs.y_pred, dtype="float64")
    naive = (inputs.extra or {}).get("naive_pred")
    if naive is None:
        season = int((inputs.extra or {}).get("seasonality", 1))
        naive = np.concatenate([truth[:season], truth[:-season]]) if len(truth) > season else truth
        predicted = predicted[-len(naive) :]
        truth = truth[-len(naive) :]
    naive = np.asarray(naive, dtype="float64")
    scale = np.mean(np.abs(np.diff(naive))) if len(naive) > 1 else 0.0
    if scale <= 1e-8:
        logger.warning("mase undefined: naive forecast has no variation")
        return float("nan")
    return float(np.mean(np.abs(truth - predicted)) / scale)


# ---------------------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------------------
def _silhouette(inputs: MetricInputs) -> float:
    """Mean silhouette coefficient (requires the feature matrix)."""
    labels = np.asarray(inputs.y_pred)
    if inputs.X is None or len(np.unique(labels)) < 2:
        return float("nan")
    return float(sk_metrics.silhouette_score(inputs.X, labels))


def _calinski_harabasz(inputs: MetricInputs) -> float:
    """Calinski-Harabasz index (between/within cluster variance ratio)."""
    labels = np.asarray(inputs.y_pred)
    if inputs.X is None or len(np.unique(labels)) < 2:
        return float("nan")
    return float(sk_metrics.calinski_harabasz_score(inputs.X, labels))


def _davies_bouldin(inputs: MetricInputs) -> float:
    """Davies-Bouldin index (lower is better)."""
    labels = np.asarray(inputs.y_pred)
    if inputs.X is None or len(np.unique(labels)) < 2:
        return float("nan")
    return float(sk_metrics.davies_bouldin_score(inputs.X, labels))


def _cluster_count(inputs: MetricInputs) -> float:
    """Number of clusters found (a diagnostic, not a quality metric)."""
    return float(len(np.unique(np.asarray(inputs.y_pred))))


# ---------------------------------------------------------------------------------------
# Ranking / recommendation
# ---------------------------------------------------------------------------------------
def _grouped_ranking(
    inputs: MetricInputs, top_k: int, scorer: Callable[[np.ndarray, np.ndarray, int], float]
) -> float:
    """Average a per-group ranking score.

    Args:
        inputs: Metric inputs (``groups`` must be provided).
        top_k: Number of recommended items.
        scorer: Per-group scoring function.

    Returns:
        The mean score over groups.
    """
    if inputs.groups is None:
        logger.warning("ranking metric without groups: falling back to a global computation")
        return scorer(np.asarray(inputs.y_true), np.asarray(inputs.y_pred), top_k)

    frame = pd.DataFrame(
        {
            "group": np.asarray(inputs.groups),
            "truth": np.asarray(inputs.y_true),
            "score": np.asarray(inputs.y_pred),
        }
    )
    scores: list[float] = []
    for _, group in frame.groupby("group", observed=True):
        if group["truth"].sum() <= 0:
            continue
        scores.append(scorer(group["truth"].to_numpy(), group["score"].to_numpy(), top_k))
    return float(np.mean(scores)) if scores else float("nan")


def _precision_at_k(truth: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """Precision among the ``top_k`` highest scored items."""
    order = np.argsort(-np.asarray(scores, dtype="float64"))[:top_k]
    relevant = np.asarray(truth, dtype="float64")[order]
    return float(safe_division(relevant.sum(), len(order)))


def _recall_at_k(truth: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """Share of relevant items captured in the ``top_k`` recommendations."""
    order = np.argsort(-np.asarray(scores, dtype="float64"))[:top_k]
    relevant = np.asarray(truth, dtype="float64")[order]
    total = float(np.asarray(truth, dtype="float64").sum())
    return float(safe_division(relevant.sum(), total))


def _hit_rate_at_k(truth: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """Whether at least one relevant item is recommended."""
    return 1.0 if _recall_at_k(truth, scores, top_k) > 0 else 0.0


def _ndcg_at_k(truth: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """Normalised discounted cumulative gain at ``top_k``."""
    truth = np.asarray(truth, dtype="float64")
    order = np.argsort(-np.asarray(scores, dtype="float64"))
    gains = truth[order][:top_k]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(truth)[::-1][:top_k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float((ideal * ideal_discounts).sum())
    return float(safe_division(dcg, idcg))


def _map_at_k(truth: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """Mean average precision at ``top_k``."""
    truth = np.asarray(truth, dtype="float64")
    order = np.argsort(-np.asarray(scores, dtype="float64"))[:top_k]
    relevant = truth[order]
    if relevant.sum() <= 0:
        return 0.0
    precisions = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
    return float((precisions * relevant).sum() / min(relevant.sum(), top_k))


def _make_ranking_metric(
    scorer: Callable[[np.ndarray, np.ndarray, int], float], top_k: int = 10
) -> Callable[[MetricInputs], float]:
    """Create a group-aware ranking metric.

    Args:
        scorer: Per-group scoring function.
        top_k: Number of recommendations.

    Returns:
        A function accepting :class:`MetricInputs`.
    """

    def compute(inputs: MetricInputs) -> float:
        """Compute the metric for the given inputs."""
        effective_top_k = int((inputs.extra or {}).get("top_k", top_k))
        return _grouped_ranking(inputs, effective_top_k, scorer)

    return compute


# ---------------------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------------------
def _recall_at_top_k_anomaly(inputs: MetricInputs) -> float:
    """Recall obtained when flagging the ``k`` most anomalous observations.

    ``k`` defaults to the number of true anomalies, which is the standard operating point for
    an unsupervised detector deployed with a fixed investigation budget.
    """
    truth = np.asarray(inputs.y_true).astype(int)
    scores = np.asarray(_positive_class_scores(inputs), dtype="float64")
    budget = int((inputs.extra or {}).get("budget", max(int(truth.sum()), 1)))
    budget = min(budget, len(scores))
    threshold_index = np.argsort(-scores)[:budget]
    flagged = np.zeros_like(truth)
    flagged[threshold_index] = 1
    return float(safe_division((flagged & truth).sum(), max(int(truth.sum()), 1)))


def _precision_at_budget(inputs: MetricInputs) -> float:
    """Precision among the flagged observations (same budget as above)."""
    truth = np.asarray(inputs.y_true).astype(int)
    scores = np.asarray(_positive_class_scores(inputs), dtype="float64")
    budget = int((inputs.extra or {}).get("budget", max(int(truth.sum()), 1)))
    budget = min(budget, len(scores))
    threshold_index = np.argsort(-scores)[:budget]
    flagged = np.zeros_like(truth)
    flagged[threshold_index] = 1
    return float(safe_division((flagged & truth).sum(), budget))


# ---------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------
_CLASSIFICATION_TASKS = frozenset({"binary", "multiclass"})
_ALL_TASKS = frozenset(
    {"binary", "multiclass", "regression", "clustering", "forecasting", "ranking", "anomaly"}
)

METRICS: dict[str, MetricDefinition] = {
    # classification
    "accuracy": MetricDefinition(
        "accuracy",
        _accuracy,
        tasks=_CLASSIFICATION_TASKS,
        description="Part de prédictions correctes",
    ),
    "balanced_accuracy": MetricDefinition(
        "balanced_accuracy",
        _balanced_accuracy,
        tasks=_CLASSIFICATION_TASKS,
        description="Moyenne des rappels par classe",
    ),
    "precision": MetricDefinition(
        "precision",
        _precision_binary,
        requires=frozenset({"y_true", "y_pred"}),
        tasks=frozenset({"binary", "anomaly"}),
        description="Précision sur la classe positive",
    ),
    "recall": MetricDefinition(
        "recall",
        _recall_binary,
        tasks=frozenset({"binary", "anomaly"}),
        description="Rappel sur la classe positive",
    ),
    "f1": MetricDefinition(
        "f1",
        _f1_binary,
        tasks=frozenset({"binary", "anomaly"}),
        description="F1 sur la classe positive",
    ),
    "precision_macro": MetricDefinition(
        "precision_macro",
        _precision_macro,
        tasks=_CLASSIFICATION_TASKS,
        description="Précision macro-moyennée",
    ),
    "recall_macro": MetricDefinition(
        "recall_macro",
        _recall_macro,
        tasks=_CLASSIFICATION_TASKS,
        description="Rappel macro-moyenné",
    ),
    "f1_macro": MetricDefinition(
        "f1_macro", _f1_macro, tasks=_CLASSIFICATION_TASKS, description="F1 macro-moyenné"
    ),
    "f1_weighted": MetricDefinition(
        "f1_weighted",
        _f1_weighted,
        tasks=_CLASSIFICATION_TASKS,
        description="F1 pondéré par le support",
    ),
    "roc_auc": MetricDefinition(
        "roc_auc",
        _roc_auc,
        requires=frozenset({"y_true", "y_proba"}),
        tasks=frozenset({"binary", "anomaly"}),
        description="Aire sous la courbe ROC",
    ),
    "pr_auc": MetricDefinition(
        "pr_auc",
        _pr_auc,
        requires=frozenset({"y_true", "y_proba"}),
        tasks=frozenset({"binary", "anomaly"}),
        description="Aire sous la courbe précision-rappel",
    ),
    "log_loss": MetricDefinition(
        "log_loss",
        _log_loss,
        higher_is_better=False,
        requires=frozenset({"y_true", "y_proba"}),
        tasks=_CLASSIFICATION_TASKS,
        description="Entropie croisée",
    ),
    "mcc": MetricDefinition(
        "mcc", _mcc, tasks=_CLASSIFICATION_TASKS, description="Corrélation de Matthews"
    ),
    # regression / forecasting
    "rmse": MetricDefinition(
        "rmse",
        _rmse,
        higher_is_better=False,
        tasks=frozenset({"regression", "forecasting", "anomaly"}),
        description="Racine de l'erreur quadratique moyenne",
    ),
    "mae": MetricDefinition(
        "mae",
        _mae,
        higher_is_better=False,
        tasks=frozenset({"regression", "forecasting"}),
        description="Erreur absolue moyenne",
    ),
    "mape": MetricDefinition(
        "mape",
        _mape,
        higher_is_better=False,
        tasks=frozenset({"regression", "forecasting"}),
        description="Erreur absolue en pourcentage",
    ),
    "smape": MetricDefinition(
        "smape",
        _smape,
        higher_is_better=False,
        tasks=frozenset({"regression", "forecasting"}),
        description="MAPE symétrique",
    ),
    "r2": MetricDefinition(
        "r2",
        _r2,
        tasks=frozenset({"regression", "forecasting"}),
        description="Coefficient de détermination",
    ),
    "max_error": MetricDefinition(
        "max_error",
        _max_error,
        higher_is_better=False,
        tasks=frozenset({"regression", "forecasting"}),
        description="Pire erreur absolue",
    ),
    "mase": MetricDefinition(
        "mase",
        _mase,
        higher_is_better=False,
        tasks=frozenset({"forecasting"}),
        description="Erreur absolue échelonnée par le naive",
    ),
    # clustering
    "silhouette": MetricDefinition(
        "silhouette",
        _silhouette,
        requires=frozenset({"y_pred", "X"}),
        tasks=frozenset({"clustering"}),
        description="Coefficient de silhouette moyen",
    ),
    "calinski_harabasz": MetricDefinition(
        "calinski_harabasz",
        _calinski_harabasz,
        requires=frozenset({"y_pred", "X"}),
        tasks=frozenset({"clustering"}),
        description="Indice de Calinski-Harabasz",
    ),
    "davies_bouldin": MetricDefinition(
        "davies_bouldin",
        _davies_bouldin,
        higher_is_better=False,
        requires=frozenset({"y_pred", "X"}),
        tasks=frozenset({"clustering"}),
        description="Indice de Davies-Bouldin",
    ),
    "n_clusters": MetricDefinition(
        "n_clusters",
        _cluster_count,
        requires=frozenset({"y_pred"}),
        tasks=frozenset({"clustering", "anomaly"}),
        description="Nombre de groupes détectés",
    ),
    # ranking
    "precision_at_k": MetricDefinition(
        "precision_at_k",
        _make_ranking_metric(_precision_at_k),
        requires=frozenset({"y_true", "y_pred", "groups"}),
        tasks=frozenset({"ranking"}),
        description="Précision@K",
    ),
    "recall_at_k": MetricDefinition(
        "recall_at_k",
        _make_ranking_metric(_recall_at_k),
        requires=frozenset({"y_true", "y_pred", "groups"}),
        tasks=frozenset({"ranking"}),
        description="Rappel@K",
    ),
    "ndcg_at_k": MetricDefinition(
        "ndcg_at_k",
        _make_ranking_metric(_ndcg_at_k),
        requires=frozenset({"y_true", "y_pred", "groups"}),
        tasks=frozenset({"ranking"}),
        description="NDCG@K",
    ),
    "map_at_k": MetricDefinition(
        "map_at_k",
        _make_ranking_metric(_map_at_k),
        requires=frozenset({"y_true", "y_pred", "groups"}),
        tasks=frozenset({"ranking"}),
        description="MAP@K",
    ),
    "hit_rate_at_k": MetricDefinition(
        "hit_rate_at_k",
        _make_ranking_metric(_hit_rate_at_k),
        requires=frozenset({"y_true", "y_pred", "groups"}),
        tasks=frozenset({"ranking"}),
        description="Hit rate@K",
    ),
    # anomaly
    "recall_at_budget": MetricDefinition(
        "recall_at_budget",
        _recall_at_top_k_anomaly,
        requires=frozenset({"y_true", "y_proba"}),
        tasks=frozenset({"anomaly", "binary"}),
        description="Rappel quand on ne peut investiguer que K alertes",
    ),
    "precision_at_budget": MetricDefinition(
        "precision_at_budget",
        _precision_at_budget,
        requires=frozenset({"y_true", "y_proba"}),
        tasks=frozenset({"anomaly", "binary"}),
        description="Précision sur le budget d'investigation",
    ),
}

#: Metric names grouped by task, used for validation and defaults.
METRICS_BY_TASK: dict[str, list[str]] = {
    "binary": [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "mcc",
    ],
    "multiclass": [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "log_loss",
    ],
    "regression": ["rmse", "mae", "r2", "mape", "smape", "max_error"],
    "forecasting": ["mae", "rmse", "mape", "smape", "mase", "r2"],
    "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin", "n_clusters"],
    "ranking": ["ndcg_at_k", "precision_at_k", "recall_at_k", "map_at_k", "hit_rate_at_k"],
    "anomaly": [
        "pr_auc",
        "roc_auc",
        "recall_at_budget",
        "precision_at_budget",
        "precision",
        "recall",
        "f1",
    ],
}


def available_metrics(task: str | None = None) -> list[str]:
    """Return the metric names available for a task.

    Args:
        task: Learning task (``None`` returns every metric).

    Returns:
        The metric names.
    """
    if task is None:
        return sorted(METRICS)
    return [
        name
        for name, definition in METRICS.items()
        if not definition.tasks or task in definition.tasks
    ]


def validate_metric_names(names: Iterable[str], task: str | None = None) -> list[str]:
    """Check metric names and warn on metrics that are not meaningful for the task.

    Args:
        names: Requested metric names.
        task: Learning task.

    Returns:
        The deduplicated metric names.

    Raises:
        KeyError: When a metric name is unknown.
    """
    resolved: list[str] = []
    for name in names:
        if name not in METRICS:
            msg = f"Unknown metric '{name}'. Available: {sorted(METRICS)}"
            raise KeyError(msg)
        if task and METRICS[name].tasks and task not in METRICS[name].tasks:
            logger.warning("Metric '{}' is unusual for task '{}'", name, task)
        if name not in resolved:
            resolved.append(name)
    return resolved


class MetricCalculator:
    """Compute a set of metrics for a task, tolerating missing inputs.

    Example:
        >>> calculator = MetricCalculator(task="binary", metrics=["accuracy", "roc_auc"])
        >>> inputs = MetricInputs(
        ...     y_true=[0, 1, 1, 0],
        ...     y_pred=[0, 1, 0, 0],
        ...     y_proba=[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2]],
        ... )
        >>> values = calculator.evaluate(inputs)
        >>> sorted(values)
        ['accuracy', 'roc_auc']
    """

    def __init__(
        self, task: str, metrics: Sequence[str], *, extra: Mapping[str, Any] | None = None
    ) -> None:
        """Validate and store the metric selection.

        Args:
            task: Learning task identifier.
            metrics: Metric names to compute.
            extra: Values forwarded to every metric through ``MetricInputs.extra``.
        """
        self.task = task
        self.metric_names = validate_metric_names(metrics, task)
        self.extra = dict(extra or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MetricCalculator:
        """Build the calculator from the ``metrics`` configuration node.

        Args:
            config: Root configuration mapping.

        Returns:
            The configured calculator.
        """
        node = dict(config.get("metrics", {}) or {})
        task = str(node.get("task", "binary"))
        names = [
            str(node.get("primary", "accuracy")),
            *[str(name) for name in node.get("secondary", []) or []],
        ]
        return cls(task=task, metrics=names, extra=dict(node.get("extra", {}) or {}))

    def evaluate(self, inputs: MetricInputs) -> dict[str, float]:
        """Compute every metric, skipping the ones whose inputs are unavailable.

        Args:
            inputs: Predictions, ground truth and optional payloads.

        Returns:
            Mapping of metric name to value (NaN when undefined).
        """
        payload = MetricInputs(
            y_true=inputs.y_true,
            y_pred=inputs.y_pred,
            y_proba=inputs.y_proba,
            X=inputs.X,
            groups=inputs.groups,
            extra={**self.extra, **(inputs.extra or {})},
        )
        values: dict[str, float] = {}
        for name in self.metric_names:
            definition = METRICS[name]
            missing = [
                requirement
                for requirement in definition.requires
                if _missing(getattr(payload, requirement, None))
            ]
            if missing:
                logger.warning("Metric '{}' skipped: missing input(s) {}", name, missing)
                values[name] = float("nan")
                continue
            try:
                values[name] = float(definition.fn(payload))
            except Exception as exc:  # noqa: BLE001 - a metric must never crash the pipeline
                logger.error("Metric '{}' failed: {}", name, exc)
                values[name] = float("nan")
        return values

    def direction_of(self, name: str) -> str:
        """Return ``maximize`` or ``minimize`` for a metric."""
        return "maximize" if METRICS[name].higher_is_better else "minimize"

    def describe(self) -> list[dict[str, Any]]:
        """Return the metadata of the selected metrics (for reports)."""
        return [
            {
                "name": name,
                "direction": self.direction_of(name),
                "description": METRICS[name].description,
                "requires": sorted(METRICS[name].requires),
            }
            for name in self.metric_names
        ]


def _missing(value: Any) -> bool:
    """Whether a metric input is unavailable."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return len(value) == 0
    return False


# ---------------------------------------------------------------------------------------
# Loss resolution (deep learning stacks)
# ---------------------------------------------------------------------------------------
LOSS_ALIASES: dict[str, dict[str, str]] = {
    "binary": {
        "torch": "bce_with_logits",
        "tensorflow": "binary_crossentropy",
        "keras": "binary_crossentropy",
        "sklearn": "log_loss",
    },
    "multiclass": {
        "torch": "cross_entropy",
        "tensorflow": "sparse_categorical_crossentropy",
        "keras": "sparse_categorical_crossentropy",
        "sklearn": "log_loss",
    },
    "regression": {"torch": "mse", "tensorflow": "mse", "keras": "mse", "sklearn": "squared_error"},
    "forecasting": {
        "torch": "mse",
        "tensorflow": "mse",
        "keras": "mse",
        "sklearn": "squared_error",
    },
    "anomaly": {"torch": "mse", "tensorflow": "mse", "keras": "mse", "sklearn": "squared_error"},
}


def resolve_loss_name(task: str, framework: str) -> str:
    """Map a task to the canonical loss name of a framework.

    Args:
        task: Learning task.
        framework: ``torch``, ``tensorflow``, ``keras`` or ``sklearn``.

    Returns:
        The loss name.

    Raises:
        ValueError: When the task or framework is unknown.
    """
    if task not in LOSS_ALIASES:
        msg = f"No loss defined for task '{task}'. Allowed: {sorted(LOSS_ALIASES)}"
        raise ValueError(msg)
    mapping = LOSS_ALIASES[task]
    if framework not in mapping:
        msg = f"No loss defined for framework '{framework}'. Allowed: {sorted(mapping)}"
        raise ValueError(msg)
    return mapping[framework]


def build_torch_loss(name: str, **kwargs: Any) -> Any:
    """Instantiate a PyTorch loss module.

    Args:
        name: Loss identifier, resolved per task through :data:`LOSS_ALIASES` (``bce``,
            ``cross_entropy``, ``mse``, ``mae``, ``huber``, ``focal``, ...).
        **kwargs: Parameters forwarded to the loss constructor.

    Returns:
        The loss module.

    Raises:
        ValueError: When the loss name is unknown.
    """
    import torch.nn as nn

    registry: dict[str, Callable[..., Any]] = {
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "nll": nn.NLLLoss,
    }
    if name == "focal":
        return FocalLoss(**kwargs)
    if name not in registry:
        msg = f"Unknown torch loss '{name}'. Allowed: {sorted([*registry, 'focal'])}"
        raise ValueError(msg)
    return registry[name](**kwargs)


def build_keras_loss(name: str, **kwargs: Any) -> Any:
    """Instantiate a Keras/TensorFlow loss.

    Args:
        name: Keras loss name (``binary_crossentropy``, ``mse``, ``huber``, ...).
        **kwargs: Parameters forwarded to the loss constructor.

    Returns:
        The loss object.
    """
    import tensorflow as tf

    registry: dict[str, Any] = {
        "binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
        "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
        "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
        "mse": tf.keras.losses.MeanSquaredError,
        "mae": tf.keras.losses.MeanAbsoluteError,
        "huber": tf.keras.losses.Huber,
    }
    if name not in registry:
        msg = f"Unknown keras loss '{name}'. Allowed: {sorted(registry)}"
        raise ValueError(msg)
    return registry[name](**kwargs)


class FocalLoss:
    """Focal loss for PyTorch (used on imbalanced classification problems).

    Down-weights easy examples so that the optimisation focuses on hard, rare cases.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Store the focal loss parameters.

        Args:
            alpha: Weight of the positive class.
            gamma: Focusing exponent.
            reduction: ``mean`` or ``sum``.
        """
        import torch.nn as nn

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, logits: Any, targets: Any) -> Any:
        """Compute the focal loss.

        Args:
            logits: Raw model outputs.
            targets: Binary targets (same shape as ``logits``).

        Returns:
            The scalar (or element-wise) loss tensor.
        """
        import torch

        probability = torch.sigmoid(logits)
        bce = self._bce(logits, targets.float())
        weight = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
        focal_weight = weight * torch.pow(torch.abs(targets.float() - probability), self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
