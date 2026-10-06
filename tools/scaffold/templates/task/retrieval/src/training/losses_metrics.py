"""Metrics of a retrieval / generation system.

Two families of numbers are computed here, and the project never mixes them up:

**Retrieval metrics** measure the *ranking* the model produced, given the set of relevant
passages. They are defined per question and averaged over questions — never computed on a flat
concatenation, which would let a single multi-document question dominate the mean:

* ``recall@k`` — share of the relevant documents found in the top ``k`` passages;
* ``precision@k`` — share of the top ``k`` passages that are relevant;
* ``hit_rate@k`` — share of questions with at least one relevant passage in the top ``k``;
* ``mrr`` — mean reciprocal rank of the first relevant passage;
* ``ndcg@k`` — rank-weighted recall: rewards finding the relevant passage *first*;
* ``map@k`` — mean average precision over the relevant passages found in the top ``k``.

**Answer metrics** measure the *grounding* of the generated answer, independently of its
wording: does it cite passages that are actually relevant, does it abstain when the corpus holds
nothing, does it overlap the reference answer. Fluency is not measurable offline and is therefore
not claimed.

Both families follow the same convention: a metric that cannot be computed (no relevant passage
in a ranking, no answer for an unanswerable question) is **excluded from the mean** and the
number of contributing questions is reported separately. Returning 0.0 for "not applicable"
would silently punish abstention and reward hallucination.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Retrieval metrics exposed by the project, in reporting order.
RETRIEVAL_METRICS: tuple[str, ...] = (
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "map_at_k",
)

#: Answer / grounding metrics exposed by the project.
ANSWER_METRICS: tuple[str, ...] = (
    "answer_f1",
    "answer_exact_match",
    "citation_precision",
    "citation_recall",
    "abstention_accuracy",
    "abstention_recall",
    "answer_coverage",
    "abstention_balanced_accuracy",
)

#: Metrics by task. The evaluator iterates over the task's list, so adding a metric to a task is
#: a one-line change here.
METRICS_BY_TASK: dict[str, tuple[str, ...]] = {
    "retrieval": (*RETRIEVAL_METRICS, *ANSWER_METRICS),
    "generation": ("answer_f1", "answer_exact_match", "citation_precision", "compression_ratio"),
    "multiclass": ("macro_f1", "accuracy", "balanced_accuracy"),
}

#: Human readable description of every metric (rendered in the README and the report).
METRIC_DESCRIPTIONS: dict[str, str] = {
    "recall_at_k": "Part des documents pertinents présents dans le top-k (par question).",
    "precision_at_k": "Part des passages du top-k qui sont pertinents.",
    "hit_rate_at_k": "Part des questions dont au moins un passage pertinent est dans le top-k.",
    "mrr": "Moyenne de l'inverse du rang du premier passage pertinent.",
    "ndcg_at_k": "Gain cumulé pondéré par le rang (1/log2(rang+1)), normalisé par l'idéal.",
    "map_at_k": "Précision moyenne sur les passages pertinents trouvés dans le top-k.",
    "answer_f1": "F1 token entre la réponse produite et la réponse de référence.",
    "answer_exact_match": "1 si la réponse normalisée est identique à la référence, 0 sinon.",
    "citation_precision": "Part des passages cités qui sont réellement pertinents.",
    "citation_recall": "Part des passages pertinents effectivement cités.",
    "abstention_accuracy": "Part des questions correctement traitées en abstention ou en réponse.",
    "abstention_recall": "Part des questions hors corpus correctement refusées (sensibilité).",
    "answer_coverage": "Part des questions présentes au corpus effectivement traitées.",
    "abstention_balanced_accuracy": "Moyenne du taux de refus correct et de la couverture : "
    "elle ne récompense ni le refus systématique ni la réponse systématique.",
    "compression_ratio": "Longueur de la réponse rapportée à celle des passages fournis.",
}


def _normalise_documents(values: Sequence[str] | set[str]) -> list[str]:
    """Return the values as a de-duplicated list, preserving order."""
    return list(dict.fromkeys(str(value) for value in values))


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents found in the top ``k``.

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``, or NaN when there is no relevant document (metric not applicable).
    """
    if not relevant:
        return float("nan")
    found = len(set(ranking[:k]) & relevant)
    return found / len(relevant)


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the top ``k`` passages that are relevant.

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]`` (``0.0`` when the ranking is empty, which *is* a measurement).
    """
    top = list(ranking[:k])
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def hit_rate_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Whether at least one relevant document is in the top ``k``.

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.
        k: Cut-off.

    Returns:
        ``1.0`` or ``0.0``, NaN when there is no relevant document.
    """
    if not relevant:
        return float("nan")
    return 1.0 if set(ranking[:k]) & relevant else 0.0


def mean_reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    """Inverse of the rank of the first relevant document.

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.

    Returns:
        ``1/rank`` of the first relevant document, ``0.0`` when none was found, NaN when the
        question has no relevant document.
    """
    if not relevant:
        return float("nan")
    for position, identifier in enumerate(ranking, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at ``k`` with binary relevance.

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``, NaN when there is no relevant document.
    """
    if not relevant:
        return float("nan")
    gains = [
        1.0 / math.log2(position + 1)
        for position, identifier in enumerate(ranking[:k], start=1)
        if identifier in relevant
    ]
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return float(sum(gains) / ideal) if ideal > 0 else float("nan")


def average_precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at ``k`` (binary relevance).

    Args:
        ranking: Retrieved documents, best first.
        relevant: Set of relevant documents.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``, NaN when there is no relevant document.
    """
    if not relevant:
        return float("nan")
    hits = 0
    total = 0.0
    for position, identifier in enumerate(ranking[:k], start=1):
        if identifier in relevant:
            hits += 1
            total += hits / position
    return float(total / min(len(relevant), k))


def retrieval_metrics(
    rankings: Sequence[Sequence[str]],
    relevances: Sequence[set[str]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate ranking metrics over a batch of questions.

    Args:
        rankings: One ranking (best first) per question.
        relevances: One set of relevant documents per question.
        ks: Cut-offs to report.

    Returns:
        Mapping of metric name (``recall_at_5``, ``mrr``, ...) to the mean over the questions
        where the metric is defined, plus ``n_questions`` and ``n_scored_<metric>`` counters.

    Raises:
        ValueError: When the two sequences do not have the same length.
    """
    if len(rankings) != len(relevances):
        msg = (
            f"rankings and relevances must have the same length, got "
            f"{len(rankings)} and {len(relevances)}"
        )
        raise ValueError(msg)

    aggregated: dict[str, float] = {}
    for k in ks:
        for name, function in (
            ("recall", recall_at_k),
            ("precision", precision_at_k),
            ("hit_rate", hit_rate_at_k),
            ("ndcg", ndcg_at_k),
            ("map", average_precision_at_k),
        ):
            values = [
                function(ranking, relevant, k)
                for ranking, relevant in zip(rankings, relevances, strict=True)
            ]
            finite = [value for value in values if not math.isnan(value)]
            aggregated[f"{name}_at_{k}"] = float(np.mean(finite)) if finite else float("nan")
            aggregated[f"n_scored_{name}_at_{k}"] = float(len(finite))
    mrr_values = [
        mean_reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevances, strict=True)
    ]
    finite_mrr = [value for value in mrr_values if not math.isnan(value)]
    aggregated["mrr"] = float(np.mean(finite_mrr)) if finite_mrr else float("nan")
    aggregated["n_scored_mrr"] = float(len(finite_mrr))
    aggregated["n_questions"] = float(len(rankings))
    return aggregated


def _content_tokens(text: str) -> list[str]:
    """Tokenise a text for the answer-overlap metrics (letters and digits only)."""
    cleaned = "".join(
        char if char.isalnum() or char.isspace() else " " for char in str(text).lower()
    )
    return [token for token in cleaned.split() if token]


def answer_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a produced answer and the reference answer.

    Args:
        prediction: Produced answer.
        reference: Reference answer.

    Returns:
        A value in ``[0, 1]`` (``0.0`` when the prediction or the reference is empty).
    """
    predicted = _content_tokens(prediction)
    expected = _content_tokens(reference)
    if not predicted or not expected:
        return 0.0
    common = 0
    remaining = list(expected)
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return float(2 * precision * recall / (precision + recall))


def answer_exact_match(prediction: str, reference: str) -> float:
    """Whether the prediction equals the reference after normalisation.

    Args:
        prediction: Produced answer.
        reference: Reference answer.

    Returns:
        ``1.0`` or ``0.0``.
    """
    return float(" ".join(_content_tokens(prediction)) == " ".join(_content_tokens(reference)))


def citation_precision(cited: Sequence[str], relevant: set[str]) -> float:
    """Share of the cited passages that are actually relevant.

    Args:
        cited: Cited passage identifiers.
        relevant: Set of relevant passage identifiers.

    Returns:
        A value in ``[0, 1]``; NaN when nothing was cited (the metric is not defined rather than
        perfect, which is what keeps a silent answer from scoring 1.0).
    """
    unique = _normalise_documents(cited)
    if not unique:
        return float("nan")
    return len(set(unique) & relevant) / len(unique)


def citation_recall(cited: Sequence[str], relevant: set[str]) -> float:
    """Share of the relevant passages that were cited.

    Args:
        cited: Cited passage identifiers.
        relevant: Set of relevant passage identifiers.

    Returns:
        A value in ``[0, 1]``, NaN when there is no relevant passage.
    """
    if not relevant:
        return float("nan")
    return len(set(_normalise_documents(cited)) & relevant) / len(relevant)


def abstention_accuracy(abstained: bool, should_abstain: bool) -> float:
    """Whether the system took the right answer/abstain decision.

    Args:
        abstained: Whether the system abstained.
        should_abstain: Whether the question has no answer in the corpus.

    Returns:
        ``1.0`` when the decision is correct, ``0.0`` otherwise.
    """
    return float(bool(abstained) == bool(should_abstain))


def percentile(values: Sequence[float], quantile: float) -> float:
    """Percentile of a sample, tolerant to empty inputs.

    Args:
        values: Observations.
        quantile: Quantile in ``[0, 1]``.

    Returns:
        The interpolated percentile, or NaN when there is no observation.
    """
    if not len(values):
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype="float64"), 100.0 * quantile))


def metrics_for_task(task: str) -> tuple[str, ...]:
    """Return the metrics declared for a task.

    Args:
        task: Learning task.

    Returns:
        The metric names, or an empty tuple for an unknown task (the evaluator then reports the
        generic diagnostics only).
    """
    return METRICS_BY_TASK.get(str(task), ())


def describe_metrics(names: Sequence[str] | None = None) -> dict[str, str]:
    """Return the human readable description of the requested metrics.

    Args:
        names: Metric names (defaults to every documented metric).

    Returns:
        Mapping of metric name to its French description.
    """
    selected = list(names) if names is not None else list(METRIC_DESCRIPTIONS)
    return {name: METRIC_DESCRIPTIONS.get(name, "Métrique non documentée.") for name in selected}


def flatten_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics mapping into ``a.b.c`` keys, keeping finite numbers only.

    Args:
        payload: Nested mapping of metrics.
        prefix: Prefix applied to the keys (used for the ``val_`` / ``test_`` namespaces).

    Returns:
        The flattened, finite metrics.
    """
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=f"{name}."))
            continue
        if isinstance(value, bool):
            flattened[name] = float(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            flattened[name] = number
    return flattened


__all__ = [
    "ANSWER_METRICS",
    "METRICS_BY_TASK",
    "METRIC_DESCRIPTIONS",
    "RETRIEVAL_METRICS",
    "abstention_accuracy",
    "answer_exact_match",
    "answer_f1",
    "average_precision_at_k",
    "citation_precision",
    "citation_recall",
    "describe_metrics",
    "flatten_metrics",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "metrics_for_task",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "retrieval_metrics",
]
