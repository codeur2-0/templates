"""Évaluation : la couche qui publie les chiffres, et la fidélité qui les nuance.

Un projet de résumé se juge sur deux axes, et un seul des deux se voit dans un score : la
**ressemblance** (ROUGE) et la **vérité** (couverture des faits saillants, valeurs absentes du
document). Les tests ci-dessous verrouillent les deux, et surtout les propriétés qui se dégradent
sans bruit :

* le verdict est lu sur la métrique contractuelle mesurée, jamais recopié ;
* les références publiées sont mesurées sur les **mêmes** documents que la stratégie servie, une
  seule fois chacune — deux lignes identiques avec un effectif doublé feraient douter du tableau ;
* une valeur absente du document est comptée, alors même que le ROUGE peut s'améliorer : c'est
  précisément le cas où un rapport complaisant ne dit rien ;
* la table évaluée se lit avec ses métriques de longueur (``n_tokens``), pas avec un champ qui
  n'existe qu'en inférence.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.evaluator import (
    SummaryEvaluation,
    SummaryEvaluator,
    coverage_scores_for,
    unsupported_values,
)
from src.models.contract import BaseTextGenerator
from src.models.lead import LeadSummarizer
from src.schemas.config import AppConfig
from src.training.metrics import verdict_from_metrics
from src.utils.config_access import node
from src.utils.paths import ProjectPaths


def _evaluator(
    model: BaseTextGenerator,
    app_config: AppConfig,
    paths: ProjectPaths,
    *,
    top_errors: int = 25,
    baselines: dict[str, BaseTextGenerator] | None = None,
) -> SummaryEvaluator:
    """Return an evaluator wired like ``EvaluationPipeline`` wires one."""
    return SummaryEvaluator(
        model,
        config=app_config.model_dump(),
        metrics_config=app_config.metrics.model_dump(),
        paths=paths,
        text_column=str(node(app_config, "model").get("text_column", "text")),
        id_column=str(app_config.data.id_column or "doc_id"),
        top_errors=top_errors,
        baselines=baselines,
    )


def test_the_extractive_floor_publishes_its_rouge_and_its_fidelity(
    evaluation: SummaryEvaluation,
    test_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Une stratégie mesurée publie ses deux axes sur l'effectif exact du test."""
    documents, _, _ = test_split
    metrics = evaluation.metrics

    assert metrics["n_documents"] == float(len(documents))
    assert 0.0 <= metrics["rouge1_f"] <= 1.0
    assert 0.0 <= metrics["fact_coverage"] <= 1.0
    assert 0.0 <= metrics["fact_precision"] <= 1.0
    assert 0.0 <= metrics["unsupported_share"] <= 1.0
    assert metrics["compression_mean"] >= 0.0
    assert {"latency_p50_ms", "latency_p95_ms", "hit_max_length_share"} <= set(metrics)


def test_the_verdict_follows_the_measured_metric_and_the_family_threshold(
    evaluation: SummaryEvaluation, app_config: AppConfig
) -> None:
    """Le verdict publié est celui que rend la mesure confrontée au seuil de la famille.

    Recalculer le verdict ici (plutôt que de le comparer à une chaîne écrite en dur) est le point du
    test : il échoue dès que le rapport s'écarte de la métrique qu'il prétend publier.
    """
    minimum = app_config.metrics.min_primary
    assert minimum is not None, "la famille doit déclarer un seuil contractuel"
    expected, detail = verdict_from_metrics(
        evaluation.metrics,
        primary=str(app_config.metrics.primary),
        minimum=float(minimum),
        direction=str(app_config.metrics.direction),
    )

    assert evaluation.verdict == expected
    assert evaluation.primary_metric == str(app_config.metrics.primary)
    assert evaluation.threshold == float(minimum)
    assert evaluation.verdict_detail["observed"] == detail["observed"]
    assert evaluation.verdict_detail["margin"] == detail["margin"]


def test_the_published_tables_keep_the_length_contract(
    evaluation: SummaryEvaluation,
) -> None:
    """La table évaluée publie ``n_tokens`` (longueur servie) et non ``n_words`` (inférence)."""
    predictions = evaluation.predictions

    assert not predictions.empty
    assert {"doc_id", "strategy", "reference_summary", "prediction"} <= set(predictions.columns)
    assert {"n_tokens", "compression", "hit_max_length", "latency_ms"} <= set(predictions.columns)
    assert "n_words" not in predictions.columns
    assert predictions["n_tokens"].ge(1).all()
    assert predictions["doc_id"].is_unique
    # La trame de fidélité porte une ligne par document évalué, plus les colonnes par type de fait.
    assert len(evaluation.fidelity) == len(predictions)
    assert {"fact_coverage", "fact_precision", "n_unsupported"} <= set(evaluation.fidelity.columns)


def test_every_published_reference_is_measured_once_on_the_same_lines(
    lead_model: LeadSummarizer,
    textrank_model: BaseTextGenerator,
    app_config: AppConfig,
    tiny_paths: ProjectPaths,
    test_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Chaque référence apparaît **une** fois, sur le même effectif que la stratégie servie.

    C'est la non-régression du tableau publié : deux noms pour un même algorithme donnaient deux
    lignes identiques avec un effectif doublé, et un lecteur qui additionne la colonne trouve plus
    de documents qu'il n'y en a dans le test.
    """
    documents, references, facts = test_split
    evaluator = _evaluator(
        lead_model, app_config, tiny_paths, baselines={"textrank": textrank_model}
    )

    result = evaluator.evaluate(documents, references, facts)
    per_strategy = result.per_strategy

    assert per_strategy["strategy"].is_unique
    assert set(per_strategy["strategy"]) == {"lead", "textrank"}
    assert per_strategy["n_documents"].tolist() == [float(len(documents))] * len(per_strategy)
    assert per_strategy["rouge1_f"].is_monotonic_decreasing
    assert result.metrics["n_documents"] == float(len(documents))
    assert result.strategies == ("lead", "textrank")
    # Les références publiées se lisent par leur nom publié, pas par leur stratégie interne.
    assert result.baselines["resume_vide"]["rouge1_f"] == 0.0
    assert result.baselines["textrank"]["strategy"] == "textrank"
    assert 0.0 <= result.baselines["textrank"]["fact_coverage"] <= 1.0


def test_the_error_frame_archives_the_declared_number_of_documents(
    lead_model: LeadSummarizer,
    app_config: AppConfig,
    tiny_paths: ProjectPaths,
    test_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """La trame d'erreurs s'arrête au nombre déclaré et classe les documents du pire au meilleur."""
    documents, references, facts = test_split
    evaluator = _evaluator(lead_model, app_config, tiny_paths, top_errors=5)

    result = evaluator.evaluate(documents, references, facts)
    errors = result.errors

    assert len(errors) == min(5, len(documents))
    assert errors["rouge1_f"].is_monotonic_increasing
    assert {"missing_facts", "unsupported_values", "reference_summary", "prediction"} <= set(
        errors.columns
    )
    assert errors["missing_facts"].map(lambda value: isinstance(value, list)).all()


def test_a_value_absent_from_the_document_is_counted_as_unsupported(
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
) -> None:
    """Une valeur inventée est comptée, et le résumé de référence ne peut pas faire mieux.

    Deux vérifications complémentaires : un résumé qui recopie le document couvre tous ses faits
    saillants et ne contient aucune valeur étrangère (par construction) ; une pièce inventée, elle,
    est déclarée — c'est le seul garde-fou contre un résumé fluide et faux.
    """
    row = documents.iloc[0]
    document = str(row["text"])
    doc_facts = facts.loc[facts["doc_id"] == row["doc_id"]]
    reference = str(references.loc[references["doc_id"] == row["doc_id"], "summary"].iloc[0])

    faithful = coverage_scores_for(document, doc_facts)
    invented = unsupported_values("Remplacement du module ZZZ-999 terminé en 97 heures.", document)

    assert faithful["fact_coverage"] == 1.0
    assert faithful["n_unsupported"] == 0
    assert faithful["fact_precision"] == 1.0
    assert unsupported_values(document, document) == []
    assert unsupported_values(reference, document) == []
    assert "zzz-999" in invented
    assert "97" in invented
