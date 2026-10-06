"""Évaluation et rapport : mesurer une fois, publier lisible, et ne rien cacher.

Trois exigences, dans l'ordre où elles comptent pour un lecteur du dépôt :

* le verdict est **contractuel** — il se lit sur la métrique principale comparée à un seuil, et
  quand le seuil n'est pas défini le verdict dit « indéterminé » au lieu d'inventer une réussite ;
* les chiffres sont **situés** — la table des références triviales (classe majoritaire, tirage
  stratifié) est publiée à côté des métriques, sans quoi une F1 macro ne veut rien dire ;
* le rapport existe **vraiment** — un Markdown, des tables CSV et des figures PNG, écrits sur le
  disque, pas une promesse dans un docstring.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.loaders import TextLabelLoader
from src.evaluation.evaluator import ClassificationEvaluator, EvaluationResult
from src.evaluation.reports import ReportBuilder
from src.models.contract import BaseTextClassifier
from src.schemas.config import AppConfig
from src.utils.config_access import node
from src.utils.paths import ProjectPaths


def _evaluate(
    model: BaseTextClassifier, app_config: AppConfig, frame: pd.DataFrame, paths: ProjectPaths
) -> EvaluationResult:
    """Measure a model on a frame with the project's evaluator."""
    evaluator = ClassificationEvaluator(
        model,
        config=app_config.model_dump(),
        metrics_config=app_config.metrics.model_dump(),
        paths=paths,
        text_column="text",
        target_column=str(app_config.data.target or "label"),
    )
    return evaluator.evaluate(frame)


def test_the_evaluation_publishes_the_contracted_metrics(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Les métriques principales, les latences et la couverture sont toutes mesurées."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    assert result.n_documents == len(test_split)
    assert result.n_classes == len(fitted_model.labels)
    assert result.primary_metric == app_config.metrics.primary
    assert result.threshold == app_config.metrics.min_primary
    for name in ("accuracy", "macro_f1", "balanced_accuracy", "latency_p50_ms", "coverage"):
        assert name in result.metrics, name
    assert 0.0 <= result.metrics["macro_f1"] <= 1.0
    assert result.metrics["coverage"] == 1.0


def test_the_verdict_follows_the_contract(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Le verdict se lit sur la comparaison métrique / seuil, jamais sur une impression."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    assert result.verdict in {"conforme", "non conforme", "indéterminé"}
    if result.threshold is None:
        assert result.verdict == "indéterminé"
    else:
        observed = result.metrics[result.primary_metric]
        expected = "conforme" if observed >= result.threshold else "non conforme"
        assert result.verdict == expected


def test_the_trivial_references_are_measured_next_to_the_model(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Sans référence triviale, une exactitude n'est pas interprétable."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    assert set(result.baselines) == {"classe_majoritaire", "tirage_stratifie"}
    for scores in result.baselines.values():
        assert "accuracy" in scores
    assert result.metrics["accuracy"] >= result.baselines["classe_majoritaire"]["accuracy"] - 0.1


def test_the_tables_are_consistent_with_the_evaluated_corpus(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Per-class, matrice de confusion et prédictions décrivent les mêmes lignes."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    assert len(result.per_class) == len(fitted_model.labels)
    assert int(result.per_class["support"].sum()) == len(test_split)
    assert result.confusion.to_numpy().sum() == len(test_split)
    assert len(result.predictions) == len(test_split)
    assert {"doc_id", "text", "expected_label", "prediction", "confidence", "correct"} <= set(
        result.predictions.columns
    )


def test_the_segments_split_the_metrics_by_declared_column(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """La ventilation par canal, style et priorité est publiée : c'est là qu'on lit les échecs."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    assert not result.segments.empty
    assert {"segment", "n_documents", "accuracy", "macro_f1"} <= set(result.segments.columns)
    prefixes = {str(value).split("=")[0] for value in result.segments["segment"]}
    assert {"source", "style", "priority"} <= prefixes
    assert int(result.segments["n_documents"].sum()) >= len(test_split)


def test_the_error_table_is_ready_to_read(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """La table des erreurs remplace le texte par un extrait : elle se relit dans un terminal."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    if not result.top_errors.empty:
        assert "text_preview" in result.top_errors.columns
        assert result.top_errors["text_preview"].str.len().max() <= 160
        assert (result.top_errors["correct"] == 0).all()


def test_the_payload_is_json_serialisable(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """``to_dict`` alimente ``evaluation_metrics.json`` : il doit passer ``json.dumps``."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)

    payload = json.dumps(result.to_dict())

    assert "macro_f1" in payload
    assert "classe_majoritaire" in payload


def test_an_empty_frame_is_refused(
    fitted_model: BaseTextClassifier, app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Évaluer zéro ticket produirait un rapport plausible sur rien."""
    with pytest.raises(ValueError, match="empty frame"):
        _evaluate(fitted_model, app_config, pd.DataFrame({"text": [], "label": []}), tiny_paths)


def test_the_report_bundles_a_markdown_and_its_artefacts(
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    test_split: pd.DataFrame,
    documents: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Le rapport écrit sur disque : un Markdown, des tables, des figures."""
    result = _evaluate(fitted_model, app_config, test_split, tiny_paths)
    metadata = {"seed": 7, "n_documents": len(documents), "rule_baseline_accuracy": 0.55}

    bundle = ReportBuilder(app_config.model_dump(), paths=tiny_paths).build(
        result,
        model_summary={
            "name": fitted_model.name,
            "framework": fitted_model.framework,
            "algorithm": fitted_model.algorithm,
            "task": fitted_model.task,
            "n_features": fitted_model.n_features,
            "classes": fitted_model.labels,
        },
        metadata=metadata,
        documents=documents,
    )

    assert bundle.report_path.exists()
    assert bundle.verdict == result.verdict
    assert bundle.summary
    text = bundle.report_path.read_text(encoding="utf-8")
    assert result.verdict in text
    assert result.primary_metric in text
    assert "Reproductibilité" in text
    names = {path.name for path in bundle.artifacts}
    assert {
        "per_class.csv",
        "segment_metrics.csv",
        "predictions.csv",
        "confusion_matrix.png",
    } <= names
    assert (tiny_paths.figures_dir / "confusion_matrix.png").exists()


def test_the_report_refuses_an_empty_evaluation(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Sans métrique, il n'y a pas de rapport : le constructeur lève plutôt que d'écrire du vide."""
    with pytest.raises(ValueError, match="no metric"):
        ReportBuilder(app_config.model_dump(), paths=tiny_paths).build(EvaluationResult(metrics={}))


def test_the_evaluation_of_the_persisted_corpus_uses_the_test_split_once(
    persisted_corpus: TextLabelLoader,
    fitted_model: BaseTextClassifier,
    app_config: AppConfig,
    tiny_paths: ProjectPaths,
) -> None:
    """Le split de test vient du corpus écrit par ``mode=generate-data``, pas d'un tirage."""
    test = persisted_corpus.split("test")

    result = _evaluate(fitted_model, app_config, test, tiny_paths)

    assert result.n_documents == len(test)
    assert node(app_config, "metrics")["primary"] == result.primary_metric
    assert set(result.predictions["doc_id"]) == set(test["doc_id"])
