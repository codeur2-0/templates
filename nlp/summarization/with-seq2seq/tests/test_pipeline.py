"""Pipelines : le chemin complet ``data`` -> ``train`` -> ``evaluate`` -> ``predict``.

Ce module exécute réellement les quatre pipelines dans un projet jetable (``tmp_path``) : c'est le
seul test qui prouve que les modes du ``Makefile`` fonctionnent, que les artefacts se chaînent (le
résumé entraîné par ``train`` est bien celui que ``evaluate`` recharge) et qu'aucun pipeline n'écrit
dans le dépôt.

Trois choses sont vérifiées en plus de l'enchaînement, parce qu'elles sont propres à cette famille :

* les trois tables du corpus sont écrites **ensemble** (documents, résumés, faits) : une évaluation
  sans table de faits publierait une couverture nulle, et personne ne le verrait ;
* le tableau des stratégies publié contient **une ligne par algorithme**, sur le même effectif : la
  stratégie servie vient de son artefact, et l'alias ``baseline_extractive`` de ``lead`` ne peut pas
  produire une seconde ligne pour le même algorithme ;
* la table d'inférence est écrite **dans le projet du pipeline**, pas dans le répertoire courant —
  c'est ce qui permet de faire tourner un pipeline depuis un bac à sable sans écraser les artefacts
  publiés du dépôt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipelines import (
    DataGenerationPipeline,
    EvaluationPipeline,
    InferencePipeline,
    PipelineResult,
    TrainPipeline,
)
from src.schemas.config import AppConfig
from src.utils.config_access import node
from src.utils.paths import ProjectPaths


def _run_all(app_config: AppConfig, tiny_paths: ProjectPaths) -> dict[str, PipelineResult]:
    """Run the four pipelines in order, in the throw-away project."""
    return {
        name: pipeline(app_config, paths=tiny_paths).run()
        for name, pipeline in (
            ("data", DataGenerationPipeline),
            ("train", TrainPipeline),
            ("evaluate", EvaluationPipeline),
            ("predict", InferencePipeline),
        )
    }


def test_the_generation_pipeline_writes_the_three_tables_and_the_recipe(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``data`` écrit les documents, leurs résumés, les faits annotés et la recette du corpus."""
    result = DataGenerationPipeline(app_config, paths=tiny_paths).run()

    assert result.name == "generate-data"
    assert result.succeeded, result.messages
    assert {"n_documents", "n_references", "n_facts", "n_salient_facts"} <= set(result.metrics)
    assert result.metrics["n_documents"] == result.metrics["n_references"]
    assert 0.0 < result.metrics["n_salient_facts"] <= result.metrics["n_facts"]
    assert (tiny_paths.raw_dir / "generation_metadata.json").exists() or any(
        Path(path).name == "generation_metadata.json" for path in result.artifacts
    )
    # Les trois tables sont écrites : une table absente rendrait l'évaluation muette, pas bruyante.
    written = {Path(path).name for path in result.artifacts}
    assert {
        "intervention_reports.parquet",
        "reference_summaries.parquet",
        "salient_facts.parquet",
    } <= (written)
    # Aucun artefact ne sort du projet jetable : le dépôt n'est jamais écrit par un test.
    assert all(Path(path).is_relative_to(tiny_paths.root) for path in result.artifacts)


def test_the_four_pipelines_chain_each_other(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """L'enchaînement complet produit un modèle, des métriques, un rapport et des prédictions."""
    results = _run_all(app_config, tiny_paths)

    for name, result in results.items():
        assert result.succeeded, f"{name}: {result.messages}"
        assert result.artifacts, f"le pipeline {name} n'a rien écrit"
        assert all(Path(path).exists() for path in result.artifacts), name
        assert all(Path(path).is_relative_to(tiny_paths.root) for path in result.artifacts), name

    artefact = tiny_paths.models_dir / str(app_config.train.artifacts.model_file)
    assert artefact.is_file()
    assert (tiny_paths.models_dir / "model_card.json").exists()
    assert (tiny_paths.models_dir / "resolved_config.json").exists()
    assert (tiny_paths.metrics_dir / "training_metrics.json").exists()
    assert (tiny_paths.metrics_dir / "evaluation_metrics.json").exists()
    assert (tiny_paths.reports_dir / str(app_config.train.artifacts.report_file)).exists()
    assert (tiny_paths.reports_dir / "per_strategy.csv").exists()
    assert (tiny_paths.reports_dir / "predictions.csv").exists()
    assert (tiny_paths.figures_dir / "rouge_par_strategie.png").exists()


def test_the_training_pipeline_archives_the_recipe_of_the_run(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Le modèle est archivé avec sa configuration résolue et sa fiche : sinon, orphelin."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    training = TrainPipeline(app_config, paths=tiny_paths).run()

    resolved = json.loads((tiny_paths.models_dir / "resolved_config.json").read_text())
    card = json.loads((tiny_paths.models_dir / "model_card.json").read_text())

    assert training.succeeded, training.messages
    assert resolved["project"]["problem"] == "summarization"
    assert resolved["metrics"]["primary"] == app_config.metrics.primary
    assert resolved["seed"] == app_config.seed
    assert card["algorithm"] == app_config.model.algorithm
    assert card["vocabulary"]["shared"] is True
    assert card["vocabulary"]["pretrained"] is False
    assert card["n_parameters"] > 0
    assert card["architecture"]["decoding"].startswith("greedy")
    params = node(app_config, "model").get("params", {}) or {}
    assert card["budget"]["max_output_tokens"] == int(params["max_output_tokens"])


def test_the_evaluation_pipeline_publishes_one_row_per_algorithm(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Le tableau des stratégies est mesuré sur le test, une ligne par algorithme, même effectif.

    C'est la non-régression de la référence publiée deux fois : ``baseline_extractive`` est le nom
    publié de ``lead``, donc compter les lignes du tableau doit donner le nombre d'algorithmes
    réellement mesurés — et l'effectif de chaque ligne, la taille du test.
    """
    generation = DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()
    evaluation = EvaluationPipeline(app_config, paths=tiny_paths).run()

    per_strategy = pd.read_csv(tiny_paths.reports_dir / "per_strategy.csv")
    metrics = json.loads((tiny_paths.metrics_dir / "evaluation_metrics.json").read_text())
    report = (tiny_paths.reports_dir / str(app_config.train.artifacts.report_file)).read_text()
    test_documents = int(generation.metrics["split_test"])

    assert evaluation.succeeded, evaluation.messages
    assert per_strategy["strategy"].is_unique
    assert per_strategy["n_documents"].tolist() == [float(test_documents)] * len(per_strategy)
    assert set(per_strategy["strategy"]) == set(evaluation.payload.strategies)
    compared = {
        str(app_config.model.algorithm),
        *[str(name) for name in node(app_config, "model").get("strategies_to_compare", []) or []],
    }
    assert set(per_strategy["strategy"]) == compared
    assert metrics["verdict"] in {"conforme", "non conforme", "indéterminé"}
    assert metrics["n_documents"] == float(test_documents)
    assert "Verdict" in report
    assert "couverture" in report


def test_the_inference_pipeline_writes_validated_predictions_in_the_project(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``predict`` écrit sa table dans le projet du pipeline, avec sa fidélité et sa latence.

    Le contrat de la table d'inférence n'est pas celui de la table d'évaluation : il publie la
    longueur en **mots** (celle d'un résumé servi) et les colonnes ROUGE calculées quand la
    référence est connue, jamais ``n_tokens`` — un test qui lit ``n_tokens`` ici lit la table d'un
    autre mode.
    """
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    TrainPipeline(app_config, paths=tiny_paths).run()
    inference = InferencePipeline(app_config, paths=tiny_paths).run()

    written = tiny_paths.reports_dir / str(app_config.train.artifacts.predictions_file)
    predictions = pd.read_csv(written)

    assert inference.succeeded, inference.messages
    assert str(inference.artifacts[0]) == str(written)
    assert len(predictions) == int(inference.metrics["n_documents"]) >= 1
    assert {"doc_id", "strategy", "reference_summary", "prediction"} <= set(predictions.columns)
    assert {"compression", "fact_coverage", "unsupported_facts", "latency_ms"} <= set(
        predictions.columns
    )
    assert "n_tokens" not in predictions.columns
    assert predictions["compression"].gt(0.0).all()
    assert predictions["compression"].le(1.0).all()
    assert predictions["fact_coverage"].between(0.0, 1.0).all()
    assert predictions["latency_ms"].ge(0.0).all()
    assert predictions["strategy"].nunique() == 1
    # La lecture rapide accompagne la table : sans elle, un fichier de résumés ne dit rien.
    assert (tiny_paths.reports_dir / "inference_summary.json").exists()


def test_the_evaluation_pipeline_refuses_to_run_without_an_artefact(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Évaluer sans modèle produirait un rapport plausible sur rien : le pipeline échoue."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    result = EvaluationPipeline(app_config, paths=tiny_paths).run()

    assert not result.succeeded
    assert result.messages
    assert "introuvable" in result.messages[0]


def test_the_inference_pipeline_refuses_to_run_without_an_artefact(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """``predict`` sans artefact échoue de la même façon, avec le mode à exécuter d'abord."""
    DataGenerationPipeline(app_config, paths=tiny_paths).run()
    result = InferencePipeline(app_config, paths=tiny_paths).run()

    assert not result.succeeded
    assert result.messages
    assert "introuvable" in result.messages[0]
