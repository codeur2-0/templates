"""Métriques d'extraction d'entités : ce que « trouver une entité » veut dire, et à quel prix.

Un span est **correct** s'il a le bon type *et* les bonnes bornes. Toutes les métriques de ce
module
découlent de cette définition, et c'est une décision de projet, pas un détail : une mention
décalée
d'un caractère est fausse pour le back-office qui remplit un dossier, donc elle est fausse ici. Ce
que cette exigence coûte est mesuré à part, par :func:`partial_scores`, qui compte comme correcte
une
mention du bon type **recouvrant** la bonne entité — l'écart entre les deux F1 est publié.

Trois familles de métriques, jamais mélangées :

* les scores **micro** (toutes les mentions comptent pareil) : :func:`entity_scores` ;
* les scores **par type** (:func:`per_label_frame`) et leur moyenne : la F1 macro, qui protège un
  type minoritaire — ici le transporteur, dont les surfaces abrégées sont ambiguës ;
* les lectures d'erreur : :func:`error_frame` dit *où* le système s'est trompé (mention manquée,
  mention inventée, bornes décalées), et :func:`confidence_table` dit si les mentions corroborées
  par
  les règles sont réellement plus précises que les autres — une table de précision, pas une
  probabilité inventée.

Le plancher trivial est publié avec les scores : un système qui n'annote rien obtient une F1 de
0,0
(par convention, l'absence totale de prédiction n'est pas une précision parfaite).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: A span is the triple ``(start, end, label)``: the key of an exact-match evaluation.
Span = tuple[int, int, str]


def span_sets(frame: pd.DataFrame) -> list[set[Span]]:
    """Turn an annotation table into one set of spans per message.

    Args:
        frame: Annotation table (``msg_id``, ``start``, ``end``, ``label``).

    Returns:
        One set per message, **in the order of first appearance** of the messages: les deux tables
        de vérité et de prédiction sont alignées par cette fonction, jamais par un tri implicite.
    """
    grouped: dict[str, set[Span]] = {}
    for row in frame.itertuples(index=False):
        grouped.setdefault(str(row.msg_id), set()).add(
            (int(row.start), int(row.end), str(row.label))
        )
    return [grouped[key] for key in grouped]


def counts_by_label(
    gold: Sequence[set[Span]],
    predicted: Sequence[set[Span]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Count true positives, false positives and false negatives, per type and overall.

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message (aligned with ``gold``).
        labels: Declared entity types; inferred from both sides when ``None``.

    Returns:
        Mapping of entity type (plus ``micro``) to ``{"tp": ..., "fp": ..., "fn": ...}``.

    Raises:
        ValueError: When the two sequences have different lengths.
    """
    if len(gold) != len(predicted):
        msg = f"gold has {len(gold)} messages and predicted has {len(predicted)}"
        raise ValueError(msg)
    declared = list(labels) if labels is not None else _labels_from(gold, predicted)
    table: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in [*declared, "micro"]
    }
    for truth, guess in zip(gold, predicted, strict=True):
        matched = truth & guess
        table["micro"]["tp"] += len(matched)
        table["micro"]["fp"] += len(guess - truth)
        table["micro"]["fn"] += len(truth - guess)
        for label in declared:
            truth_label = {span for span in truth if span[2] == label}
            guess_label = {span for span in guess if span[2] == label}
            table[label]["tp"] += len(truth_label & guess_label)
            table[label]["fp"] += len(guess_label - truth_label)
            table[label]["fn"] += len(truth_label - guess_label)
    return table


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Turn raw counts into precision, recall and F1.

    Args:
        tp: Number of matched spans.
        fp: Number of spurious spans.
        fn: Number of missed spans.

    Returns:
        The three scores, plus the ``support`` (number of reference spans). Un dénominateur nul
        donne 0,0 et non 1,0 : un système qui n'annonce rien n'a aucune précision.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": int(tp + fn),
    }


def entity_scores(
    gold: Sequence[set[Span]],
    predicted: Sequence[set[Span]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Score a prediction at the entity level (exact matches, micro and macro).

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message (aligned with ``gold``).
        labels: Declared entity types.

    Returns:
        A flat mapping with ``entity_precision``, ``entity_recall``, ``entity_f1`` (micro),
        ``macro_f1``, ``type_accuracy``, ``boundary_accuracy``, ``predicted_entities``,
        ``true_entities`` and ``matched_entities``.
    """
    table = counts_by_label(gold, predicted, labels=labels)
    micro = precision_recall_f1(**table["micro"])
    per_label = {
        label: precision_recall_f1(**counts) for label, counts in table.items() if label != "micro"
    }
    macro = float(np.mean([scores["f1"] for scores in per_label.values()])) if per_label else 0.0
    type_accuracy, boundary_accuracy = _alignment_scores(gold, predicted)
    return {
        "entity_precision": micro["precision"],
        "entity_recall": micro["recall"],
        "entity_f1": micro["f1"],
        "macro_f1": round(macro, 4),
        "type_accuracy": round(type_accuracy, 4),
        "boundary_accuracy": round(boundary_accuracy, 4),
        "predicted_entities": float(sum(len(spans) for spans in predicted)),
        "true_entities": float(sum(len(spans) for spans in gold)),
        "matched_entities": float(table["micro"]["tp"]),
    }


def per_label_frame(
    gold: Sequence[set[Span]],
    predicted: Sequence[set[Span]],
    *,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return precision, recall, F1 and support per entity type.

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message.
        labels: Declared entity types.

    Returns:
        One row per entity type, plus a ``micro`` row, sorted by decreasing support.
    """
    table = counts_by_label(gold, predicted, labels=labels)
    rows: list[dict[str, Any]] = []
    for label, counts in table.items():
        scores = precision_recall_f1(**counts)
        counts_row = {f"n_{key}": value for key, value in counts.items()}
        rows.append({"label": label, **scores, **counts_row})
    frame = pd.DataFrame(rows)
    order = {"micro": -1}
    frame["_order"] = frame["label"].map(lambda label: order.get(label, 0))
    frame = frame.sort_values(["_order", "support"], ascending=[True, False], kind="stable")
    return frame.drop(columns=["_order"]).reset_index(drop=True)


def partial_scores(
    gold: Sequence[set[Span]],
    predicted: Sequence[set[Span]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Score with **tolerance on the boundaries**: an overlapping span of the right type matches.

    L'appariement est fait au plus grand recouvrement, un pour un : une prédiction ne peut pas
    valider deux entités, et une entité ne peut pas être validée deux fois. L'écart entre cette F1
    et
    la F1 exacte est ce que coûte l'exigence de bornes exactes — et il est publié.

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message.
        labels: Declared entity types (used to report the per-type detail).

    Returns:
        A flat mapping with ``partial_precision``, ``partial_recall``, ``partial_f1`` and
        ``boundary_errors`` (mentioned but not exactly matched).
    """
    tp = fp = fn = 0
    boundary_errors = 0
    for truth, guess in zip(gold, predicted, strict=True):
        exact = truth & guess
        remaining_truth = [span for span in truth if span not in exact]
        pairs: list[tuple[int, int, int]] = []
        for index, predicted_span in enumerate(sorted(guess)):
            if predicted_span in exact:
                continue
            for position, gold_span in enumerate(remaining_truth):
                if gold_span[2] != predicted_span[2]:
                    continue
                overlap = min(predicted_span[1], gold_span[1]) - max(
                    predicted_span[0], gold_span[0]
                )
                if overlap > 0:
                    pairs.append((overlap, index, position))
        pairs.sort(reverse=True, key=lambda item: item[0])
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for _, index, position in pairs:
            if index in used_predictions or position in used_truth:
                continue
            used_predictions.add(index)
            used_truth.add(position)
            boundary_errors += 1
        tp += len(exact) + len(used_predictions)
        fp += len(guess) - len(exact) - len(used_predictions)
        fn += len(truth) - len(exact) - len(used_truth)
    scores = precision_recall_f1(tp, fp, fn)
    return {
        "partial_precision": scores["precision"],
        "partial_recall": scores["recall"],
        "partial_f1": scores["f1"],
        "boundary_errors": float(boundary_errors),
    }


def error_frame(
    documents: pd.DataFrame,
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    """Describe the mistakes: missed entities, spurious ones and shifted boundaries.

    Args:
        documents: Message table (``msg_id``, ``text``).
        gold: Reference annotation table.
        predicted: Predicted mention table (``msg_id``, ``start``, ``end``, ``label``).
        limit: Maximum number of rows returned.

    Returns:
        One row per mistake, with its kind, its type, its surface and the sentence around it.
    """
    texts = dict(zip(documents["msg_id"], documents["text"], strict=True))
    gold_by_message: dict[str, dict[Span, str]] = {}
    for row in gold.itertuples(index=False):
        gold_by_message.setdefault(str(row.msg_id), {})[
            (int(row.start), int(row.end), str(row.label))
        ] = str(row.surface)
    predicted_by_message: dict[str, dict[Span, float]] = {}
    for row in predicted.itertuples(index=False):
        confidence = float(getattr(row, "confidence", 0.0))
        predicted_by_message.setdefault(str(row.msg_id), {})[
            (int(row.start), int(row.end), str(row.label))
        ] = confidence

    rows: list[dict[str, Any]] = []
    for message_id, truth in gold_by_message.items():
        guess = predicted_by_message.get(message_id, {})
        text = str(texts.get(message_id, ""))
        for span in sorted(truth):
            if span in guess:
                continue
            overlapping = [
                candidate
                for candidate in guess
                if candidate[2] == span[2]
                and min(candidate[1], span[1]) - max(candidate[0], span[0]) > 0
            ]
            kind = "bornes" if overlapping else "manquee"
            nearest = overlapping[0] if overlapping else None
            rows.append(
                {
                    "msg_id": message_id,
                    "kind": kind,
                    "label": span[2],
                    "start": span[0],
                    "end": span[1],
                    "surface": str(truth[span]),
                    "predicted_surface": _surface(text, nearest),
                    "predicted_length": int(nearest[1] - nearest[0]) if nearest else None,
                    "context": _context(text, span[0], span[1]),
                }
            )
        for span in sorted(guess):
            if span in truth:
                continue
            if any(
                candidate[2] == span[2]
                and min(candidate[1], span[1]) - max(candidate[0], span[0]) > 0
                for candidate in truth
            ):
                continue
            rows.append(
                {
                    "msg_id": message_id,
                    "kind": "inventee",
                    "label": span[2],
                    "start": span[0],
                    "end": span[1],
                    "surface": _surface(text, span),
                    "predicted_surface": _surface(text, span),
                    "predicted_length": int(span[1] - span[0]),
                    "context": _context(text, span[0], span[1]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {"inventee": 0, "bornes": 1, "manquee": 2}
    frame["_order"] = frame["kind"].map(order)
    frame = frame.sort_values(["_order", "label", "msg_id"], kind="stable")
    return frame.drop(columns=["_order"]).head(int(limit)).reset_index(drop=True)


def confidence_table(
    predicted: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    bins: int = 5,
) -> pd.DataFrame:
    """Measure the precision of every confidence level.

    La confiance publiée par la stack est un **niveau de corroboration** (part du span confirmée
    par
    les règles déclarées), pas une probabilité du modèle. Cette table la rend lisible : par
    tranche,
    combien de mentions sont correctes. Si les tranches hautes ne sont pas plus précises que les
    basses, la table le dit.

    Args:
        predicted: Predicted mention table (``msg_id``, ``start``, ``end``, ``label``,
        ``confidence``).
        gold: Reference annotation table.
        bins: Number of equal-width buckets between 0 and 1.

    Returns:
        One row per bucket with the number of mentions, the matched ones and the observed
        precision.
    """
    if predicted.empty:
        return pd.DataFrame(columns=["bucket", "n_mentions", "n_correct", "precision"])
    truth = {
        (str(row.msg_id), int(row.start), int(row.end), str(row.label))
        for row in gold.itertuples(index=False)
    }
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[dict[str, Any]] = []
    confidence = predicted["confidence"].astype(float).to_numpy()
    keys = [
        (str(row.msg_id), int(row.start), int(row.end), str(row.label))
        for row in predicted.itertuples(index=False)
    ]
    correct = np.asarray([key in truth for key in keys])
    for index in range(int(bins)):
        low, high = float(edges[index]), float(edges[index + 1])
        upper = confidence <= high if index == bins - 1 else confidence < high
        selected = (confidence >= low) & upper
        count = int(selected.sum())
        good = int(correct[selected].sum())
        rows.append(
            {
                "bucket": f"[{low:.1f}; {high:.1f}]",
                "n_mentions": count,
                "n_correct": good,
                "precision": round(good / count, 4) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def confidence_gap(predicted: pd.DataFrame, gold: pd.DataFrame) -> float:
    """Return the absolute gap between the mean confidence of the correct mentions and 1.0.

    Args:
        predicted: Predicted mention table (``confidence`` column).
        gold: Reference annotation table.

    Returns:
        ``|mean(confidence | correct) - 1.0|``: une confiance qui vaut la précision observée donne
        un
        écart nul, une confiance systématiquement plus haute que la précision donne un écart
        positif.
        Zéro quand aucune mention n'est correcte, faute de quoi lire un écart serait trompeur.
    """
    if predicted.empty or "confidence" not in predicted.columns:
        return 0.0
    truth = {
        (str(row.msg_id), int(row.start), int(row.end), str(row.label))
        for row in gold.itertuples(index=False)
    }
    keys = [
        (str(row.msg_id), int(row.start), int(row.end), str(row.label))
        for row in predicted.itertuples(index=False)
    ]
    correct = np.asarray([key in truth for key in keys])
    if not correct.any():
        return 0.0
    values = predicted["confidence"].astype(float).to_numpy()[correct]
    return round(float(abs(values.mean() - 1.0)), 4)


def trivial_floor() -> dict[str, float]:
    """Return the floor of a system that predicts nothing.

    Returns:
        A zero F1 for every entity-level metric: c'est le niveau que le rapport publie à côté des
        scores, pour qu'aucun résultat ne soit lu sans son point de comparaison.
    """
    return {
        "entity_f1": 0.0,
        "entity_precision": 0.0,
        "entity_recall": 0.0,
        "macro_f1": 0.0,
        "partial_f1": 0.0,
    }


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarise a list of per-document latencies, in milliseconds.

    Args:
        values: Latency of every scored document.

    Returns:
        Mean, median, p95 and maximum latency.
    """
    if not values:
        return {
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    array = np.asarray([float(value) for value in values], dtype=float)
    return {
        "latency_mean_ms": round(float(array.mean()), 4),
        "latency_p50_ms": round(float(np.quantile(array, 0.5)), 4),
        "latency_p95_ms": round(float(np.quantile(array, 0.95)), 4),
        "latency_max_ms": round(float(array.max()), 4),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return one quantile of a list of values.

    Args:
        values: Values to summarise.
        quantile: Quantile to read (0 to 1).

    Returns:
        The quantile, or 0.0 for an empty list.
    """
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(list(values), dtype=float), float(quantile)))


def describe_metrics(names: Sequence[str] | None = None) -> dict[str, str]:
    """Return the documentation of the entity metrics (used by the reports).

    Args:
        names: Metric names to document (defaults to the whole registry).

    Returns:
        Mapping of metric name to its definition.
    """
    registry = {
        "entity_f1": (
            "F1 micro au niveau entité : type et bornes exacts, toutes mentions pesant pareil."
        ),
        "entity_precision": "Part des mentions prédites qui sont exactes (précision micro).",
        "entity_recall": "Part des mentions du corpus qui ont été retrouvées (rappel micro).",
        "macro_f1": "Moyenne non pondérée des F1 par type : protège un type minoritaire.",
        "partial_f1": ("F1 avec tolérance sur les bornes : un recouvrement du bon type compte."),
        "type_accuracy": "Part des mentions alignées dont le type est correct.",
        "boundary_accuracy": "Part des mentions bien typées dont les bornes sont exactes.",
        "mean_confidence": (
            "Confiance moyenne publiée avec les mentions (niveau de corroboration)."
        ),
        "confidence_gap": (
            "Écart entre la confiance des mentions correctes et 1,0 : une table de précision, "
            "pas une probabilité."
        ),
        "predicted_entities": (
            "Nombre de mentions prédites (un système muet n'a aucune précision)."
        ),
        "true_entities": "Nombre de mentions de référence.",
        "latency_p50_ms": "Latence médiane d'extraction d'un message, artefact chaud.",
        "latency_p95_ms": "Latence p95 d'extraction d'un message.",
        "boundary_errors": "Nombre de mentions du bon type mais aux bornes décalées.",
    }
    if names is None:
        return registry
    return {name: registry.get(name, "Métrique déclarée par le manifeste.") for name in names}


def flatten_metrics(payload: Mapping[str, object], *, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics payload into ``prefix + name`` float entries.

    Args:
        payload: Nested mapping of metrics.
        prefix: Prefix applied to every key.

    Returns:
        The flat mapping, keeping only finite numeric values.
    """
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=f"{name}_"))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) and np.isfinite(float(value)):
            flat[name] = float(value)
    return flat


def _alignment_scores(
    gold: Sequence[set[Span]], predicted: Sequence[set[Span]]
) -> tuple[float, float]:
    """Measure typing and boundary quality on the aligned spans.

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message.

    Returns:
        ``(type_accuracy, boundary_accuracy)``: parmi les paires qui se recouvrent, la part dont
        le
        type est correct, puis la part de celles-là dont les bornes sont exactes.
    """
    aligned = 0
    typed_right = 0
    boundaries_right = 0
    for truth, guess in zip(gold, predicted, strict=True):
        for predicted_span in guess:
            overlapping = [
                gold_span
                for gold_span in truth
                if min(predicted_span[1], gold_span[1]) - max(predicted_span[0], gold_span[0]) > 0
            ]
            if not overlapping:
                continue
            aligned += 1
            same_type = [span for span in overlapping if span[2] == predicted_span[2]]
            if same_type:
                typed_right += 1
                if predicted_span in same_type:
                    boundaries_right += 1
    type_accuracy = typed_right / aligned if aligned else 0.0
    boundary_accuracy = boundaries_right / typed_right if typed_right else 0.0
    return type_accuracy, boundary_accuracy


def _labels_from(*frames: Sequence[set[Span]]) -> list[str]:
    """Return the sorted entity types observed on both sides."""
    observed = {span[2] for spans in frames for doc in spans for span in doc}
    return sorted(observed)


def _surface(text: str, span: Span | None) -> str | None:
    """Return the text of a span, or ``None`` when there is no span."""
    if span is None:
        return None
    return text[span[0] : span[1]]


def _context(text: str, start: int, end: int, *, window: int = 24) -> str:
    """Return the sentence around a span, elided when it is long."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


__all__ = [
    "Span",
    "confidence_gap",
    "confidence_table",
    "counts_by_label",
    "describe_metrics",
    "entity_scores",
    "error_frame",
    "flatten_metrics",
    "latency_stats",
    "partial_scores",
    "per_label_frame",
    "percentile",
    "precision_recall_f1",
    "span_sets",
    "trivial_floor",
]
