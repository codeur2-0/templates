"""Pipelines : le parcours complet, exécuté sur un corpus minuscule.

Ce module est le test d'intégration du projet : il joue ``generate-data`` → ``train`` →
``evaluate`` → ``predict`` dans un répertoire jetable, avec la vraie configuration Hydra. Il
vérifie ce qu'aucun test unitaire ne peut vérifier : que les maillons s'enchaînent, que les
artefacts attendus existent, et surtout que le contrat de non-régression tient — **le pipeline
d'évaluation ne réentraîne rien** et le seuil appris par ``train`` est bien celui qu'applique
``evaluate``.

Ces tests écrivent dans ``tmp_path`` : lancer la suite ne doit jamais modifier les données du
projet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models import load_model
from src.pipelines.data_pipeline import DataGenerationPipeline
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.train_pipeline import TrainPipeline
from src.schemas.config import AppConfig
from src.utils.paths import ProjectPaths


@pytest.fixture
def pipeline_paths(tmp_path: Path) -> ProjectPaths:
    """Return an isolated layout, created and ready for the pipelines."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    return paths


def test_generate_data_writes_a_valid_dataset(
    app_config: AppConfig, pipeline_paths: ProjectPaths
) -> None:
    """``generate-data`` valide puis écrit le corpus, les questions et les métadonnées."""
    result = DataGenerationPipeline(app_config, paths=pipeline_paths).run()
    assert result.succeeded, result.messages
    assert pipeline_paths.raw_dir.joinpath("documents.parquet").exists()
    assert pipeline_paths.raw_dir.joinpath("queries.parquet").exists()
    assert pipeline_paths.raw_dir.joinpath("generation_metadata.json").exists()

    documents = pd.read_parquet(pipeline_paths.raw_dir / "documents.parquet")
    queries = pd.read_parquet(pipeline_paths.raw_dir / "queries.parquet")
    assert len(documents) == int(result.metrics["n_documents"])
    assert set(queries["split"]) == {"calibration", "val", "test"}
    assert result.metrics["n_questions"] == len(queries)


def test_train_then_evaluate_then_predict(
    app_config: AppConfig, pipeline_paths: ProjectPaths
) -> None:
    """Le parcours complet doit produire les artefacts et rester cohérent de bout en bout."""
    assert DataGenerationPipeline(app_config, paths=pipeline_paths).run().succeeded

    trained = TrainPipeline(app_config, paths=pipeline_paths).run()
    assert trained.succeeded, trained.messages
    artefact = pipeline_paths.models_dir / str(app_config.train.artifacts.model_file)
    assert artefact.exists()
    assert pipeline_paths.processed_dir.joinpath("chunks.parquet").exists()
    assert pipeline_paths.metrics_dir.joinpath("training_metrics.json").exists()

    evaluated = EvaluationPipeline(app_config, paths=pipeline_paths).run()
    assert evaluated.succeeded, evaluated.messages
    assert pipeline_paths.metrics_dir.joinpath("evaluation_metrics.json").exists()
    report = pipeline_paths.reports_dir / "evaluation_report.md"
    assert report.exists()
    assert "Synthèse" in report.read_text(encoding="utf-8")

    predicted = InferencePipeline(app_config, paths=pipeline_paths).run()
    assert predicted.succeeded, predicted.messages
    output = pipeline_paths.root / str(app_config.predict.output)
    assert output.exists()
    predictions = pd.read_csv(output)
    assert len(predictions) >= 1
    assert set(predictions.columns) >= {"query_id", "question", "answer", "abstained"}


def test_evaluate_reuses_the_threshold_learned_by_train(
    app_config: AppConfig, pipeline_paths: ProjectPaths
) -> None:
    """Le seuil calibré doit être celui de l'artefact : réentraîner fausserait la comparaison."""
    assert DataGenerationPipeline(app_config, paths=pipeline_paths).run().succeeded
    assert TrainPipeline(app_config, paths=pipeline_paths).run().succeeded
    artefact = pipeline_paths.models_dir / str(app_config.train.artifacts.model_file)
    trained_model = load_model(artefact, config=app_config.model_dump())
    assert EvaluationPipeline(app_config, paths=pipeline_paths).run().succeeded
    reloaded = load_model(artefact, config=app_config.model_dump())
    assert reloaded.abstention_threshold == pytest.approx(trained_model.abstention_threshold)


def test_evaluation_report_states_the_verdict(
    app_config: AppConfig, pipeline_paths: ProjectPaths
) -> None:
    """Le rapport doit trancher explicitement (conforme / non conforme), pas suggérer."""
    assert DataGenerationPipeline(app_config, paths=pipeline_paths).run().succeeded
    assert TrainPipeline(app_config, paths=pipeline_paths).run().succeeded
    assert EvaluationPipeline(app_config, paths=pipeline_paths).run().succeeded
    report = (pipeline_paths.reports_dir / "evaluation_report.md").read_text(encoding="utf-8")
    assert "conforme" in report
    assert str(app_config.metrics.primary) in report
    assert "abstention" in report

    metrics = pipeline_paths.metrics_dir / "evaluation_metrics.json"
    payload = metrics.read_text(encoding="utf-8")
    assert str(app_config.metrics.primary) in payload


def test_predict_accepts_a_user_question_file(
    app_config: AppConfig, pipeline_paths: ProjectPaths, tmp_path: Path
) -> None:
    """Le chemin de service : un fichier de questions fourni par l'utilisateur."""
    assert DataGenerationPipeline(app_config, paths=pipeline_paths).run().succeeded
    assert TrainPipeline(app_config, paths=pipeline_paths).run().succeeded
    questions = tmp_path / "questions.csv"
    pd.DataFrame(
        {"question": ["Combien de jours de congés payés par an pour les salariés en CDI ?"]}
    ).to_csv(questions, index=False)
    config = app_config.model_copy(
        update={"predict": app_config.predict.model_copy(update={"input": str(questions)})}
    )
    result = InferencePipeline(config, paths=pipeline_paths).run()
    assert result.succeeded, result.messages
    predictions = result.payload
    assert len(predictions) == 1
    assert "congés" in str(predictions.loc[0, "question"])


def test_train_without_validation_questions_is_possible(
    app_config: AppConfig, pipeline_paths: ProjectPaths
) -> None:
    """Un corpus sans question annotée reste indexable : c'est le cas d'un vrai client.

    Le pipeline doit alors réussir **et le dire** : un entraînement sans métrique de validation
    n'est pas un échec, mais il ne doit pas produire un rapport qui laisse croire à une mesure.
    """
    assert DataGenerationPipeline(app_config, paths=pipeline_paths).run().succeeded
    queries_path = pipeline_paths.raw_dir / "queries.parquet"
    queries = pd.read_parquet(queries_path)
    queries = queries.assign(split="test")
    queries.to_parquet(queries_path, index=False)

    result = TrainPipeline(app_config, paths=pipeline_paths).run()
    assert result.succeeded, result.messages
    # Seuls les compteurs de l'index restent (``index_*``) : aucune métrique de qualité ne doit
    # être publiée quand aucune question n'a été évaluée.
    quality_metrics = [
        name
        for name in result.metrics
        if name.startswith("val_") and not name.startswith("val_index_")
    ]
    assert not quality_metrics
    assert result.metrics["index_n_documents"] > 0
    assert (pipeline_paths.models_dir / str(app_config.train.artifacts.model_file)).exists()
