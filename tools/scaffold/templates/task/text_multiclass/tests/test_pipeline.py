"""Pipelines : le chemin complet ``generate-data`` -> ``train`` -> ``evaluate`` -> ``predict``.

Ce module exécute réellement les quatre pipelines dans un projet jetable (``tmp_path``) : c'est le
seul test qui prouve que les modes du ``Makefile`` fonctionnent, que les artefacts se chaînent
(le modèle écrit par ``train`` est bien celui que ``evaluate`` recharge) et qu'aucun pipeline
n'écrit dans le dépôt. Un pipeline qui passe ses tests unitaires mais échoue en enchaînement est un
pipeline cassé : c'est ici qu'on s'en aperçoit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines import (
    DataGenerationPipeline,
    EvaluationPipeline,
    InferencePipeline,
    PipelineResult,
    TrainPipeline,
)
from src.schemas.config import AppConfig
from src.utils.paths import ProjectPaths


def test_the_generation_pipeline_writes_a_validated_corpus(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``generate-data`` écrit le corpus et sa recette, dans le projet qu'on lui donne."""
    result = DataGenerationPipeline(app_config, paths=tiny_paths).run()

    assert result.name == "generate-data"
    assert result.succeeded
    assert (tiny_paths.raw_dir / "support_tickets.parquet").exists()
    assert (tiny_paths.raw_dir / "generation_metadata.json").exists()
    assert result.payload.metadata["n_documents"] == result.metrics["n_documents"]
    assert 0.0 < result.metrics["majority_accuracy"] < 1.0
    # Aucun artefact ne sort du projet jetable : le dépôt n'est jamais écrit par un test.
    assert all(Path(path).is_relative_to(tiny_paths.root) for path in result.artifacts)


def test_the_four_pipelines_chain_each_other(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """L'enchaînement complet produit un modèle, un rapport et des prédictions validées."""
    generation = DataGenerationPipeline(app_config, paths=tiny_paths).run()
    training = TrainPipeline(app_config, paths=tiny_paths).run()
    evaluation = EvaluationPipeline(app_config, paths=tiny_paths).run()
    inference = InferencePipeline(app_config, paths=tiny_paths).run()

    for result in (generation, training, evaluation, inference):
        assert isinstance(result, PipelineResult)
        assert result.artifacts, f"le pipeline {result.name} n'a rien écrit"
        assert all(Path(path).exists() for path in result.artifacts), result.name

    model_file = tiny_paths.models_dir / str(app_config.train.artifacts.model_file)
    assert model_file.exists(), "le pipeline d'entraînement doit écrire l'artefact du modèle"
    assert (tiny_paths.metrics_dir / "training_metrics.json").exists()
    assert (tiny_paths.metrics_dir / "evaluation_metrics.json").exists()
    assert (tiny_paths.reports_dir / "evaluation_report.md").exists()
    assert (tiny_paths.reports_dir / "predictions.csv").exists()
    assert (tiny_paths.figures_dir / "confusion_matrix.png").exists()


def test_the_training_pipeline_archives_the_recipe_of_the_run(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Le modèle est archivé avec sa configuration résolue et sa fiche : sinon, orphelin."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()

    result = TrainPipeline(app_config, paths=tiny_paths).run()

    assert (tiny_paths.models_dir / "model_card.json").exists()
    assert (tiny_paths.models_dir / "resolved_config.json").exists()
    assert (tiny_paths.models_dir / "preprocessing.joblib").exists()
    assert f"val_{app_config.metrics.primary}" in result.metrics
    assert any("tickets" in message for message in result.messages)


def test_the_evaluation_pipeline_refuses_to_run_without_a_model(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Évaluer sans modèle produirait un rapport sur rien : le pipeline doit refuser.

    ``BasePipeline.run`` encapsule les exceptions dans un ``PipelineResult`` en échec : le refus se
    lit donc dans le résultat, et c'est cette forme-là qui remonte à ``main.py`` et à la CI.
    """
    DataGenerationPipeline(app_config, paths=tiny_paths).run()

    result = EvaluationPipeline(app_config, paths=tiny_paths).run()

    assert not result.succeeded
    assert any("Model artefact not found" in message for message in result.messages)


def test_the_inference_pipeline_writes_a_contract_valid_table(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``predict`` valide ses prédictions contre le contrat Pandera avant de les écrire."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()

    result = InferencePipeline(app_config, paths=tiny_paths).run()
    written = pd.read_csv(tiny_paths.reports_dir / "predictions.csv")

    assert len(written) == int(result.metrics["n_predictions"])
    assert int(result.metrics["n_predictions"]) == app_config.predict.n_samples
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.metrics["latency_p95_ms"] >= 0.0
    assert {"doc_id", "prediction", "confidence"} <= set(written.columns)


def test_the_pipelines_are_replayable(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Rejouer ``generate-data`` produit exactement le même corpus : le seed est dans la config."""
    first = DataGenerationPipeline(app_config, paths=tiny_paths).run()
    corpus_first = first.payload.documents.copy()
    second = DataGenerationPipeline(app_config, paths=tiny_paths).run()

    pd.testing.assert_frame_equal(corpus_first, second.payload.documents)
