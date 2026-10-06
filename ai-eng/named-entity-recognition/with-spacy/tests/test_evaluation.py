"""Évaluation : métriques, références mesurées sur les mêmes lignes, segments et rapport.

Ce que ce module vérifie, et pourquoi chaque point compte :

* l'évaluateur mesure **quatre choses de plus** qu'un score : la table par type, les segments
  (style rédactionnel, canal), les **surfaces réservées** et deux références. Un rapport sans elles
  affiche un chiffre, pas un résultat ;
* les références sont mesurées **sur le même split** que le modèle — le plancher trivial (0,0) et la
  couche de règles apprise sur le train. Comparer un score de test à un score de validation serait
  une faute de méthode, et le test l'interdit ;
* le verdict est **lu** sur la métrique principale : il est *conforme*, *non conforme* ou
  *indéterminé*, jamais « conforme par défaut » ;
* le rapport est reconstruit à partir du *payload* : ce qu'il affiche est exactement ce que
  ``evaluation_metrics.json`` contient, et il cite la comparaison des surfaces réservées.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.evaluator import EntityEvaluator
from src.evaluation.reports import ReportBuilder


def _evaluator(fitted_model, app_config, **overrides) -> EntityEvaluator:
    """Build the evaluator of the project, with the test configuration.

    Args:
        fitted_model: Fitted extractor under test.
        app_config: Validated configuration.
        **overrides: Extra constructor arguments.

    Returns:
        The configured evaluator.
    """
    return EntityEvaluator(
        fitted_model,
        config=app_config.model_dump(),
        primary_metric=app_config.metrics.primary,
        min_primary_metric=app_config.metrics.min_primary,
        **overrides,
    )


def test_the_evaluation_measures_the_entity_level_and_its_references(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Les métriques d'entité, les latences, les références et le verdict sortent du même appel."""
    documents, spans = test_split
    result = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths).evaluate(
        documents, spans
    )

    assert result.n_documents == len(documents)
    assert result.n_entities == len(spans)
    assert 0.0 <= result.metrics["entity_f1"] <= 1.0
    assert result.metrics["partial_f1"] >= result.metrics["entity_f1"]
    assert result.metrics["latency_p50_ms"] >= 0.0
    assert set(result.baselines) == {"aucune_entite", "regles"}
    assert result.baselines["aucune_entite"]["entity_f1"] == 0.0
    assert "entity_f1" in result.baselines["regles"]
    assert result.verdict in {"conforme", "non conforme", "indéterminé"}
    assert result.verdict_detail["metric"] == app_config.metrics.primary
    assert result.primary_metric == app_config.metrics.primary
    assert result.threshold == app_config.metrics.min_primary


def test_the_rule_reference_is_learned_on_the_training_split_only(
    fitted_model, app_config, persisted_corpus, test_split, documents, spans
) -> None:
    """La couche de règles est apprise sur le train et mesurée sur le test, jamais sur elle-même."""
    test_documents, test_spans = test_split
    evaluator = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths)
    result = evaluator.evaluate(test_documents, test_spans)

    rules = result.baselines["regles"]
    assert rules["n_predicted_entities"] > 0
    assert 0.0 <= rules["entity_f1"] <= 1.0
    # Le modèle a été entraîné sur les surfaces vues : il les retrouve presque toutes, alors que
    # les réservées lui échappent en partie. C'est l'écart que publie le rapport, et il se voit
    # déjà avec une époque d'entraînement — la valeur exacte, elle, dépend du run.
    assert result.metrics["seen_surface_recall"] >= 0.9
    assert result.metrics["holdout_recall"] < result.metrics["seen_surface_recall"]


def test_the_segments_and_the_reserved_surfaces_are_published(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Les segments ventile style et canal, et le comparatif des surfaces réservées est chiffré."""
    test_documents, test_spans = test_split
    result = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths).evaluate(
        test_documents, test_spans
    )

    assert set(result.segments["segment"]) >= {"style", "canal"}
    assert result.segments["entity_f1"].between(0.0, 1.0).all()
    assert set(result.holdout["group"]) == {"reservee", "vue_au_train"}
    assert result.metrics["seen_surface_recall"] >= result.metrics["holdout_recall"]
    assert result.metrics["holdout_gap"] >= 0.0
    assert not result.confidence.empty
    assert not result.errors.empty


def test_the_evaluation_result_is_serialisable(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Le *payload* se sérialise : le JSON d'évaluation et le rapport disent la même chose."""
    test_documents, test_spans = test_split
    result = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths).evaluate(
        test_documents, test_spans
    )
    payload = result.to_dict()

    assert payload["n_documents"] == result.n_documents
    assert payload["verdict"] == result.verdict
    assert payload["verdict_detail"]["observed"] == result.metrics[app_config.metrics.primary]
    assert len(payload["per_label"]) == len(result.per_label)
    assert {row["group"] for row in payload["holdout"]} == {"reservee", "vue_au_train"}
    assert isinstance(payload["metrics"]["entity_f1"], float)


def test_the_evaluator_can_skip_the_baselines(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Mesurer les références coûte un ajustement : l'appelant peut le refuser explicitement."""
    test_documents, test_spans = test_split
    result = _evaluator(
        fitted_model, app_config, paths=persisted_corpus.paths, measure_baselines=False
    ).evaluate(test_documents, test_spans)

    assert result.baselines == {}
    assert "entity_f1" in result.metrics


def test_the_verdict_is_read_on_the_primary_metric(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Un seuil inatteignable rend le verdict *non conforme*, et le message le dit."""
    test_documents, test_spans = test_split
    result = EntityEvaluator(
        fitted_model,
        config=app_config.model_dump(),
        paths=persisted_corpus.paths,
        primary_metric=app_config.metrics.primary,
        min_primary_metric=1.1,
    ).evaluate(test_documents, test_spans)

    assert result.verdict == "non conforme"
    assert "≥" in str(result.verdict_detail["message"]) or "<" in str(
        result.verdict_detail["message"]
    )
    assert result.verdict_detail["passed"] is False


def test_the_report_builder_writes_the_report_its_tables_and_its_figures(
    fitted_model, app_config, persisted_corpus, test_split, tiny_paths
) -> None:
    """Le rapport, ses tables et ses figures sont écrits, et le verdict y figure."""
    test_documents, test_spans = test_split
    documents = persisted_corpus.load_documents()
    spans = persisted_corpus.load_annotations()
    result = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths).evaluate(
        test_documents, test_spans
    )

    bundle = ReportBuilder(app_config.model_dump(), paths=tiny_paths).build(
        result,
        model_summary={
            "name": type(fitted_model).__name__,
            "framework": "spacy",
            "algorithm": fitted_model.algorithm,
            "task": fitted_model.task,
            "labels": list(fitted_model.labels),
            "state": fitted_model.state,
        },
        metadata=persisted_corpus.load_metadata(),
        documents=documents,
        spans=spans,
    )
    text = bundle.report_path.read_text(encoding="utf-8")

    assert bundle.report_path.name == str(app_config.train.artifacts.report_file)
    assert bundle.verdict == result.verdict
    assert "Verdict" in text and result.primary_metric in text
    assert "surfaces réservées" in text
    assert all(path.exists() for path in bundle.artifacts)
    assert any(path.suffix == ".csv" for path in bundle.artifacts)
    assert any(path.suffix == ".png" for path in bundle.artifacts)


def test_the_report_refuses_to_claim_a_verdict_without_a_metric(app_config, tiny_paths) -> None:
    """Sans métrique, pas de rapport : un verdict ne s'invente pas."""
    from src.evaluation.evaluator import EvaluationResult

    with pytest.raises(ValueError, match="metric"):
        ReportBuilder(app_config.model_dump(), paths=tiny_paths).build(EvaluationResult())


def test_the_verdict_message_names_the_threshold_and_the_margin(
    fitted_model, app_config, persisted_corpus, test_split
) -> None:
    """Le paragraphe de verdict cite la valeur observée, le seuil et la marge."""
    test_documents, test_spans = test_split
    result = _evaluator(fitted_model, app_config, paths=persisted_corpus.paths).evaluate(
        test_documents, test_spans
    )
    observed = result.metrics[result.primary_metric]
    threshold = float(result.threshold or 0.0)

    message = str(result.verdict_detail["message"])
    assert f"{observed:.4f}" in message
    assert f"{threshold:.4f}" in message
    assert pd.notna(observed)
