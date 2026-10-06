"""Pipelines : le chemin complet ``generate-data`` -> ``train`` -> ``evaluate`` -> ``predict``.

Ce module exécute réellement les quatre pipelines dans un projet jetable (``tmp_path``) : c'est le
seul test qui prouve que les modes du ``Makefile`` fonctionnent, que les artefacts se chaînent (le
répertoire d'artefact écrit par ``train`` est bien celui que ``evaluate`` recharge) et qu'aucun
pipeline n'écrit dans le dépôt.

Deux choses sont vérifiées en plus de l'enchaînement, parce qu'elles sont propres à cette famille :

* l'artefact du modèle est un **répertoire** (``nlp.to_disk``) : le chaînage échouerait avec un
  fichier unique, donc le test regarde le type du chemin écrit ;
* la **configuration résolue** est archivée avec l'artefact : sans elle, un run n'est pas
  reproductible, même avec la graine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loaders import EntityCorpusLoader
from src.pipelines import (
    DataGenerationPipeline,
    EvaluationPipeline,
    InferencePipeline,
    PipelineResult,
    TrainPipeline,
)
from src.schemas.config import AppConfig
from src.utils.paths import ProjectPaths


def test_the_generation_pipeline_writes_the_two_tables_and_the_recipe(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``generate-data`` écrit les messages, les annotations et la recette, dans le projet donné."""
    result = DataGenerationPipeline(app_config, paths=tiny_paths).run()

    assert result.name == "generate-data"
    assert result.succeeded, result.messages
    loader = EntityCorpusLoader(tiny_paths)
    assert loader.documents_path.exists()
    assert loader.annotations_path.exists()
    assert loader.metadata_path.exists()
    documents, spans = loader.load_corpus()
    assert result.metrics["n_documents"] == float(len(documents))
    assert result.metrics["n_entities"] == float(len(spans))
    assert result.metrics["holdout_entities"] > 0
    assert 0.0 < result.metrics["holdout_share"] < 1.0
    # Aucun artefact ne sort du projet jetable : le dépôt n'est jamais écrit par un test.
    assert all(Path(path).is_relative_to(tiny_paths.root) for path in result.artifacts)


def test_the_four_pipelines_chain_each_other(app_config: AppConfig, tiny_paths: ProjectPaths) -> None:
    """L'enchaînement complet produit un artefact, un rapport et des mentions validées."""
    generation = DataGenerationPipeline(app_config, paths=tiny_paths).run()
    training = TrainPipeline(app_config, paths=tiny_paths).run()
    evaluation = EvaluationPipeline(app_config, paths=tiny_paths).run()
    inference = InferencePipeline(app_config, paths=tiny_paths).run()

    for result in (generation, training, evaluation, inference):
        assert isinstance(result, PipelineResult)
        assert result.succeeded, f"{result.name}: {result.messages}"
        assert result.artifacts, f"le pipeline {result.name} n'a rien écrit"
        assert all(Path(path).exists() for path in result.artifacts), result.name

    artefact = tiny_paths.models_dir / str(app_config.train.artifacts.model_file)
    assert artefact.is_dir(), "un pipeline spaCy se persiste en répertoire (``nlp.to_disk``)"
    assert (artefact / "manifest.json").exists()
    assert (tiny_paths.metrics_dir / "training_metrics.json").exists()
    assert (tiny_paths.metrics_dir / "evaluation_metrics.json").exists()
    assert (tiny_paths.reports_dir / str(app_config.train.artifacts.report_file)).exists()
    assert (tiny_paths.reports_dir / str(app_config.train.artifacts.predictions_file)).exists()
    assert (tiny_paths.reports_dir / "inference_summary.csv").exists()
    assert (tiny_paths.figures_dir / "f1_per_label.png").exists()


def test_the_training_pipeline_archives_the_recipe_of_the_run(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Le modèle est archivé avec sa configuration résolue et sa fiche : sinon, orphelin."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()

    resolved = json.loads((tiny_paths.models_dir / "resolved_config.json").read_text())
    card = json.loads((tiny_paths.models_dir / "model_card.json").read_text())

    assert resolved["project"]["stack"] == "spacy"
    assert resolved["metrics"]["primary"] == app_config.metrics.primary
    assert card["framework"] == "spacy"
    assert card["params"]["resolved_architecture"]["pipe_names"]
    assert card["metrics"]["val_entity_f1"] >= 0.0
    assert card["notes"]


def test_the_evaluation_pipeline_reads_the_test_split_once(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Le rapport est construit sur le test, avec ses références et ses segments."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()
    evaluation = EvaluationPipeline(app_config, paths=tiny_paths).run()

    payload = evaluation.payload
    report = (tiny_paths.reports_dir / str(app_config.train.artifacts.report_file)).read_text()
    metrics = json.loads((tiny_paths.metrics_dir / "evaluation_metrics.json").read_text())

    assert payload.n_documents == int(metrics["n_documents"])
    assert metrics["verdict"] in {"conforme", "non conforme", "indéterminé"}
    assert set(metrics["baselines"]) == {"aucune_entite", "regles"}
    assert metrics["holdout"][0]["group"] == "reservee"
    assert "Verdict" in report
    assert "surfaces réservées" in report
    assert "Reproductibilité" in report


def test_the_inference_pipeline_writes_validated_mentions(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``predict`` écrit une table de mentions validée, avec sa provenance et sa latence."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()
    inference = InferencePipeline(app_config, paths=tiny_paths).run()

    written = tiny_paths.reports_dir / str(app_config.train.artifacts.predictions_file)
    mentions = pd.read_csv(written)

    assert len(mentions) == int(inference.metrics["n_mentions"])
    assert mentions["source"].isin(["regle", "modele"]).all()
    assert mentions["confidence"].between(0.0, 1.0).all()
    assert (mentions["end"] > mentions["start"]).all()
    assert mentions["latency_ms"].ge(0.0).all()
    assert {"msg_id", "start", "end", "label", "surface"} <= set(mentions.columns)


def test_the_evaluation_pipeline_refuses_to_run_without_an_artefact(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Évaluer un modèle vide produirait un rapport plausible sur rien : le pipeline échoue."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    result = EvaluationPipeline(app_config, paths=tiny_paths).run()

    assert not result.succeeded
    assert result.messages
    assert "tagger" in result.messages[0] or "FileNotFound" in result.messages[0]
