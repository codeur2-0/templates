"""Métriques du résumé : fidélité aux faits, longueurs, segments et verdict.

Le ROUGE dit à quel point un résumé *ressemble* à sa référence ; il ne dit pas
si ce qu'il raconte est **vrai**. Un décodeur qui recopie « 45 minutes » au
hasard dans un document où la durée est de 30 minutes obtient un bon ROUGE-2 et
raconte une contre-vérité. Ce module fournit la seconde moitié de la mesure :

* **couverture** — la part des faits saillants du document (durées, références de pièce, symptômes,
  actions, statuts) que le résumé rapporte, globalement et **par type de fait** : un résumé qui
  couvre les symptômes et oublie les durées n'a pas la même valeur pour un lecteur pressé ;
* **faits non supportés** — les valeurs présentes dans le résumé et **absentes du document** : c'est
  la seule mesure d'hallucination possible sans modèle de langue juge, et elle est honnête : elle
  détecte une valeur inventée, jamais une phrase inventée sans valeur ;
* **longueurs** — compression moyenne, part de résumés qui atteignent leur budget, longueur en
  phrases. Un résumé qui bute sur sa borne ne résume pas, il s'arrête ;
* **segments** — les mêmes métriques ventilées par type d'intervention et par urgence déclarée, pour
  qu'une moyenne ne puisse pas cacher un effondrement sur une catégorie ;
* **planchers** — les références mesurées sur les mêmes lignes : une baseline qui copie les
  premières phrases du document (``plancher_extractif``) et un résumé vide (score de ROUGE nul). Les
  deux sont publiés avec le corpus, donc le score du modèle se lit par différence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from src.data.schemas import FACT_TYPES, fact_columns

#: Values that carry a fact and can be checked against the document: reference codes (``JNT-204``),
#: durations (``45 minutes``), plain numbers (``12``) and decimals with a comma or a dot.
VALUE_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,4}-\d{1,4}|[a-zà-öø-ÿ]+-\d{1,4}|\d{1,3}(?:[.,]\d{1,2})?)\b",
    flags=re.IGNORECASE,
)

#: Words that make a bare number a *value of a fact* rather than an enumeration.
NUMBER_UNITS: tuple[str, ...] = (
    "minute",
    "minutes",
    "heure",
    "heures",
    "jour",
    "jours",
    "mm",
    "bar",
)


def normalise_value(value: object) -> str:
    """Normalise a value so that two spellings of the same fact compare equal.

    Args:
        value: Raw value (``45 minutes``, ``45 min.``, ``JNT-204``…).

    Returns:
        The lower-cased, whitespace-collapsed value, commas turned into dots.
    """
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text.replace(",", ".")


def salient_values(facts: pd.DataFrame) -> dict[str, set[str]]:
    """Group the salient values of a document by fact type.

    Args:
        facts: Fact table restricted to one document.

    Returns:
        Mapping ``fact_type -> normalised values`` for the salient facts only.
    """
    grouped: dict[str, set[str]] = {name: set() for name in FACT_TYPES}
    salient = facts.loc[facts["salient"] == 1]
    for row in salient.itertuples(index=False):
        grouped[str(row.fact_type)].add(normalise_value(row.value))
        grouped[str(row.fact_type)].add(normalise_value(row.surface))
    return {name: values for name, values in grouped.items() if values}


def unsupported_values(prediction: str, document: str) -> list[str]:
    """List the values of a summary that do not appear in its source document.

    The check is deliberately narrow: a value is *unsupported* when it is not a substring of the
    document, in a normalised form. It therefore catches an invented duration, an invented part
    reference or a swapped figure — and it says nothing about a sentence that reorders true facts.

    Args:
        prediction: Generated summary.
        document: Source document.

    Returns:
        The distinct unsupported values, in order of appearance (deterministic).
    """
    haystack = normalise_value(document)
    seen: list[str] = []
    for match in VALUE_PATTERN.finditer(prediction):
        value = normalise_value(match.group(0))
        if len(value) < 2 or value in haystack or value in seen:
            continue
        # Un nombre nu (``3``, ``12``) est ambigu : il peut venir d'une énumération réécrite. On ne
        # le compte comme inventé que lorsqu'il est suivi d'une unité de mesure, ce qui est le seul
        # cas où il porte un fait vérifiable.
        tail = prediction[match.end() : match.end() + 12].casefold()
        if value.isdigit() and not any(unit in tail for unit in NUMBER_UNITS):
            continue
        seen.append(value)
    return seen


def coverage_scores(
    prediction: str, facts: pd.DataFrame, *, document: str | None = None
) -> dict[str, float]:
    """Score one generated summary against the facts of its document.

    Args:
        prediction: Generated summary.
        facts: Fact table restricted to the document.
        document: Source text, used to detect unsupported values (measured only when provided).

    Returns:
        ``fact_coverage``, ``n_salient``, ``n_covered``, ``n_unsupported``, the
        per-type coverage columns (``covered_<type>``) and the fact precision
        (covered facts over the facts the summary states, right or wrong).
    """
    normalised = normalise_value(prediction)
    grouped = salient_values(facts)
    metrics: dict[str, float] = {}
    n_salient = 0
    n_covered = 0
    for name in FACT_TYPES:
        values = grouped.get(name, set())
        covered = sum(1 for value in values if value in normalised)
        metrics[f"covered_{name}"] = round(covered / len(values), 4) if values else 0.0
        metrics[f"n_salient_{name}"] = float(len(values))
        n_salient += len(values)
        n_covered += covered
    unsupported = unsupported_values(prediction, document) if document is not None else []
    n_unsupported = len(unsupported)
    metrics.update(
        {
            "fact_coverage": round(n_covered / n_salient, 4) if n_salient else 0.0,
            "n_salient": float(n_salient),
            "n_covered": float(n_covered),
            "n_unsupported": float(n_unsupported),
            "fact_precision": round(n_covered / max(n_covered + n_unsupported, 1), 4),
            "unsupported_rate": round(n_unsupported / max(n_covered + n_unsupported, 1), 4),
        }
    )
    return metrics


def coverage_frame(
    predictions: pd.DataFrame, facts: pd.DataFrame, documents: pd.DataFrame
) -> pd.DataFrame:
    """Compute the fact metrics of every (document, strategy) pair of a prediction table.

    Args:
        predictions: Prediction table (``doc_id``, ``strategy``, ``prediction``).
        facts: Fact table of the evaluated split.
        documents: Document table of the evaluated split (source text).

    Returns:
        One row per prediction: coverage, per-type coverage, unsupported values and lengths.
    """
    texts = dict(zip(documents["doc_id"], documents["text"], strict=False))
    rows: list[dict[str, object]] = []
    for (doc_id, strategy), group in predictions.groupby(["doc_id", "strategy"], sort=True):
        prediction = str(group["prediction"].iloc[0])
        document = str(texts.get(str(doc_id), ""))
        metrics = coverage_scores(
            prediction, facts.loc[facts["doc_id"] == doc_id], document=document
        )
        rows.append(
            {
                "doc_id": str(doc_id),
                "strategy": str(strategy),
                "n_words": len(prediction.split()),
                "unsupported_values": " | ".join(unsupported_values(prediction, document)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def aggregate_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Average the per-document fact metrics of an evaluation.

    Args:
        frame: Table produced by :func:`coverage_frame`.

    Returns:
        The mean coverage, the mean per-type coverage, the mean length and the share of summaries
        with at least one unsupported value.
    """
    if frame.empty:
        return {"fact_coverage": 0.0, "fact_precision": 0.0, "unsupported_share": 0.0}
    metrics: dict[str, float] = {
        "fact_coverage": round(float(frame["fact_coverage"].mean()), 4),
        "fact_precision": round(float(frame["fact_precision"].mean()), 4),
        "unsupported_share": round(float((frame["n_unsupported"] > 0).mean()), 4),
        "unsupported_facts_mean": round(float(frame["n_unsupported"].mean()), 3),
        "n_words_mean": round(float(frame["n_words"].mean()), 2),
    }
    for column in fact_columns():
        if column in frame.columns:
            metrics[column] = round(float(frame[column].mean()), 4)
    return metrics


def length_stats(predictions: pd.DataFrame) -> dict[str, float]:
    """Describe the lengths of a set of generated summaries.

    Args:
        predictions: Prediction table (``compression``, ``n_sentences``, ``hit_max_length``).

    Returns:
        Mean and maximum compression, mean number of sentences and the share of summaries that hit
        their length budget.
    """
    if predictions.empty:
        return {
            "compression_mean": 0.0,
            "compression_max": 0.0,
            "sentences_mean": 0.0,
            "hit_max_length_share": 0.0,
        }
    return {
        "compression_mean": round(float(predictions["compression"].mean()), 4),
        "compression_max": round(float(predictions["compression"].max()), 4),
        "sentences_mean": round(float(predictions["n_sentences"].mean()), 2),
        "hit_max_length_share": round(float(predictions["hit_max_length"].mean()), 4),
    }


def segment_frame(
    predictions: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    columns: Sequence[str] = ("intervention_type", "urgency"),
    metric_columns: Sequence[str] = ("rouge1_f", "rouge2_f", "rouge_l_f", "fact_coverage"),
) -> pd.DataFrame:
    """Break the metrics down by segment value.

    Args:
        predictions: Prediction table with one row per (document, strategy).
        documents: Document table carrying the segment columns.
        columns: Segment dimensions to ventilate.
        metric_columns: Metrics averaged inside each segment.

    Returns:
        One row per ``(segment, value)`` with the mean of each metric and the row count.
    """
    merged = predictions.merge(documents, on="doc_id", how="left", suffixes=("", "_doc"))
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in merged.columns:
            continue
        for value, group in merged.groupby(column, sort=True):
            row: dict[str, object] = {
                "segment": str(column),
                "value": str(value),
                "n_documents": len(group),
            }
            for metric in metric_columns:
                if metric in group.columns:
                    row[metric] = round(float(group[metric].mean()), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def per_strategy_frame(
    predictions: pd.DataFrame, *, metric_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Aggregate the metrics of each strategy, so that two models are read on the same lines.

    Args:
        predictions: Prediction table with a ``strategy`` column.
        metric_columns: Metrics to average (defaults to every numeric metric available).

    Returns:
        One row per strategy, sorted by decreasing ROUGE-1 F1 (ties broken by strategy name).
    """
    if predictions.empty:
        return pd.DataFrame()
    candidates = [
        "rouge1_f",
        "rouge2_f",
        "rouge_l_f",
        "rouge1_p",
        "rouge1_r",
        "rouge_l_r",
        "fact_coverage",
        "fact_precision",
        "unsupported_facts",
        "compression",
        "hit_max_length",
        "latency_ms",
    ]
    metrics = list(metric_columns or candidates)
    available = [name for name in metrics if name in predictions.columns]
    grouped = predictions.groupby("strategy", sort=True)
    rows: list[dict[str, object]] = []
    for strategy, group in grouped:
        row: dict[str, object] = {"strategy": str(strategy), "n_documents": len(group)}
        for name in available:
            row[name] = round(float(group[name].mean()), 4)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "rouge1_f" in frame.columns:
        frame = frame.sort_values(["rouge1_f", "strategy"], ascending=[False, True]).reset_index(
            drop=True
        )
    return frame


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarise generation latencies (published, never used as a criterion).

    Args:
        values: Latencies in milliseconds.

    Returns:
        Mean, median and 95th percentile, in milliseconds.
    """
    if not values:
        return {"latency_mean_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    array = np.asarray([float(value) for value in values], dtype="float64")
    return {
        "latency_mean_ms": round(float(array.mean()), 3),
        "latency_p50_ms": round(float(np.percentile(array, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(array, 95)), 3),
    }


def trivial_floor() -> dict[str, float]:
    """Metrics of a system that returns an empty summary.

    Returns:
        ROUGE zeros (a score of 0.0 is *defined*, not ``nan``, for an empty prediction).
    """
    return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rouge_l_f": 0.0, "fact_coverage": 0.0}


def lead_baseline_metrics(
    references: Iterable[str], lead_summaries: Iterable[str]
) -> dict[str, float]:
    """ROUGE of a lead-``n`` extraction against the references (the published extractive floor).

    Args:
        references: Reference summaries.
        lead_summaries: Summaries produced by :class:`~src.models.lead.LeadSummarizer`.

    Returns:
        The mean ROUGE-1/2/L of that extraction.
    """
    from src.evaluation.rouge import corpus_rouge

    return corpus_rouge(list(references), list(lead_summaries))


def verdict_from_metrics(
    metrics: Mapping[str, float],
    *,
    primary: str = "rouge1_f",
    minimum: float | None = None,
    direction: str = "maximize",
) -> tuple[str, dict[str, object]]:
    """Read the contractual verdict of a run.

    Args:
        metrics: Flat metrics of the run.
        primary: Name of the contractual metric.
        minimum: Contractual threshold (``None`` when the family declares none).
        direction: ``maximize`` or ``minimize``.

    Returns:
        ``(verdict, detail)`` where verdict is ``conforme``, ``non conforme`` or
        ``indéterminé``, and detail carries the observed value, the threshold and the margin.
    """
    detail: dict[str, object] = {"metric": primary, "direction": direction}
    observed = metrics.get(primary)
    if observed is None:
        return "indéterminé", {**detail, "reason": f"métrique '{primary}' absente"}
    detail["observed"] = round(float(observed), 4)
    if minimum is None:
        return "indéterminé", {**detail, "reason": "aucun seuil déclaré par la famille"}
    detail["threshold"] = float(minimum)
    margin = (
        float(observed) - float(minimum)
        if direction == "maximize"
        else float(minimum) - float(observed)
    )
    detail["margin"] = round(margin, 4)
    return ("conforme" if margin >= 0.0 else "non conforme"), detail


def describe_metrics(names: Sequence[str] | None = None) -> dict[str, str]:
    """Describe the metrics of the family, so the report reads like a document.

    Args:
        names: Restrict the description to those metric names.

    Returns:
        Mapping ``metric -> one-sentence French definition``.
    """
    catalogue = {
        "rouge1_f": (
            "ROUGE-1 F1 : recouvrement des unigrams avec la meilleure référence du document."
        ),
        "rouge2_f": (
            "ROUGE-2 F1 : recouvrement des bigrams — la variante qui récompense l'ordre des mots."
        ),
        "rouge_l_f": (
            "ROUGE-L F1 : plus longue sous-séquence commune, proche de la structure de phrase."
        ),
        "fact_coverage": "Part des faits saillants du document que le résumé rapporte.",
        "fact_precision": (
            "Faits rapportés justes, divisés par les faits rapportés (justes ou inventés)."
        ),
        "unsupported_facts": "Valeurs du résumé absentes du document : hallucination détectable.",
        "compression": "Longueur du résumé rapportée à celle du document.",
        "hit_max_length": (
            "Part des résumés qui atteignent leur budget de longueur (le décodeur s'arrête)."
        ),
        "latency_p50_ms": "Latence médiane de génération, en millisecondes (jamais un critère).",
    }
    if names is None:
        return catalogue
    return {name: catalogue.get(name, "Métrique publiée par le projet.") for name in names}


def flatten_metrics(payload: Mapping[str, object], *, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics mapping into ``dot.path -> float`` (JSON-friendly).

    Args:
        payload: Nested mapping.
        prefix: Prefix applied to every key.

    Returns:
        The flattened mapping; non-numeric leaves are skipped.
    """
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[name] = float(value)
    return flattened


__all__ = [
    "NUMBER_UNITS",
    "VALUE_PATTERN",
    "aggregate_coverage",
    "coverage_frame",
    "coverage_scores",
    "describe_metrics",
    "flatten_metrics",
    "latency_stats",
    "lead_baseline_metrics",
    "length_stats",
    "normalise_value",
    "per_strategy_frame",
    "salient_values",
    "segment_frame",
    "trivial_floor",
    "unsupported_values",
    "verdict_from_metrics",
]
