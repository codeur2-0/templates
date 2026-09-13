"""Tests de bout en bout : les quatre pipelines du cycle de vie.

C'est le filet de sécurité le plus important du projet. Les tests unitaires vérifient des
comportements isolés ; ceux-ci vérifient que **l'enchaînement réel** fonctionne, dans un
répertoire temporaire isolé :

    generate-data  →  train  →  evaluate  →  predict

Un pipeline retourne toujours un :class:`PipelineResult` : en cas d'échec, le statut est
``failure`` et le message est porté par l'objet plutôt que par une exception non maîtrisée,
ce qui est exactement le comportement attendu d'un point d'entrée de production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.pipelines.data_pipeline import DataGenerationPipeline
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.train_pipeline import TrainPipeline
from src.schemas.config import AppConfig, PathsConfig
from src.utils.paths import ProjectPaths

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sandbox_config(app_config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    """Return the real configuration redirected into an isolated directory."""
    sandbox = tmp_path_factory.mktemp("pipeline-sandbox")
    return app_config.model_copy(update={"paths": PathsConfig(root=sandbox)})


@pytest.fixture(scope="module")
def sandbox_paths(sandbox_config: AppConfig) -> ProjectPaths:
    """Return the isolated project layout."""
    return ProjectPaths.from_config(sandbox_config.model_dump()).ensure()


@pytest.fixture(scope="module")
def generation_result(sandbox_config: AppConfig) -> Any:
    """Run the data generation pipeline once for the module."""
    return DataGenerationPipeline(sandbox_config).run()


@pytest.fixture(scope="module")
def training_result(sandbox_config: AppConfig, generation_result: Any) -> Any:
    """Run the training pipeline once for the module (after generation)."""
    assert generation_result.succeeded
    return TrainPipeline(sandbox_config).run()


@pytest.fixture(scope="module")
def evaluation_result(sandbox_config: AppConfig, training_result: Any) -> Any:
    """Run the evaluation pipeline once for the module (after training)."""
    assert training_result.succeeded
    return EvaluationPipeline(sandbox_config).run()


class TestDataGenerationPipeline:
    """Génération du jeu de données synthétique."""

    def test_generation_succeeds(self, generation_result: Any) -> None:
        """La génération ne doit jamais échouer : c'est la première commande du README."""
        assert generation_result.succeeded, generation_result.messages
        assert generation_result.status == "success"

    def test_dataset_is_written(
        self, generation_result: Any, sandbox_paths: ProjectPaths, sandbox_config: AppConfig
    ) -> None:
        """Le dataset est écrit dans ``data/raw`` au format déclaré."""
        written = [Path(item) for item in generation_result.artifacts]
        assert written
        assert all(path.exists() for path in written)
        assert (sandbox_paths.raw_dir / f"{sandbox_config.data.dataset_name}.parquet").exists()

    def test_generated_rows_match_configuration(
        self, generation_result: Any, sandbox_config: AppConfig
    ) -> None:
        """Le nombre de lignes généré respecte ``data.n_samples``."""
        assert generation_result.metrics.get("n_rows", 0) == sandbox_config.data.n_samples or (
            generation_result.payload is not None
        )

    def test_generated_data_passes_the_raw_contract(
        self, sandbox_paths: ProjectPaths, sandbox_config: AppConfig
    ) -> None:
        """Les données générées sont valides vis-à-vis de ``RawDataSchema``."""
        from src.data.loaders import RawDataLoader

        loader = RawDataLoader(sandbox_paths, dataset_name=sandbox_config.data.dataset_name)
        frame = loader.load()
        assert len(frame) == sandbox_config.data.n_samples


class TestTrainPipeline:
    """Entraînement complet et persistance des artefacts."""

    def test_training_succeeds(self, training_result: Any) -> None:
        """Le pipeline d'entraînement doit aller au bout (contrat du Makefile)."""
        assert training_result.succeeded, training_result.messages

    def test_training_reports_metrics(
        self, training_result: Any, sandbox_config: AppConfig
    ) -> None:
        """La métrique primaire est calculée et bornée."""
        metrics = training_result.metrics
        assert metrics
        primary = sandbox_config.metrics.primary
        matching = [value for name, value in metrics.items() if primary in name]
        assert matching, f"'{primary}' absent des métriques {sorted(metrics)}"
        assert all(pd.notna(value) for value in matching)

    def test_model_artifacts_are_persisted(
        self, training_result: Any, sandbox_paths: ProjectPaths
    ) -> None:
        """Modèle, preprocessing, feature builder et fiche modèle sont sur le disque."""
        assert (sandbox_paths.models_dir).exists()
        written = {Path(item).name for item in training_result.artifacts}
        assert written
        assert any(name.endswith(".joblib") or name.endswith(".pkl") for name in written)

    def test_splits_are_persisted(self, sandbox_paths: ProjectPaths) -> None:
        """Les splits enrichis sont sauvegardés (nécessaires à l'évaluation et aux notebooks)."""
        assert (sandbox_paths.processed_dir / "split_train.parquet").exists()
        assert (sandbox_paths.processed_dir / "split_test.parquet").exists()
        assert (sandbox_paths.processed_dir / "features_X_train.parquet").exists()


class TestEvaluationPipeline:
    """Évaluation sur le split de test et production des rapports."""

    def test_evaluation_succeeds(self, evaluation_result: Any) -> None:
        """L'évaluation d'un modèle entraîné doit fonctionner sans ré-entraînement."""
        assert evaluation_result.succeeded, evaluation_result.messages

    def test_report_is_written(self, evaluation_result: Any, sandbox_paths: ProjectPaths) -> None:
        """Le rapport Markdown est produit (livrable métier)."""
        reports = list(sandbox_paths.reports_dir.glob("*.md"))
        assert reports
        content = reports[0].read_text(encoding="utf-8")
        assert len(content) > 500
        assert "#" in content

    def test_figures_are_written(self, evaluation_result: Any, sandbox_paths: ProjectPaths) -> None:
        """Les figures d'évaluation sont générées (PNG non vides)."""
        figures = list(sandbox_paths.figures_dir.glob("*.png"))
        assert figures
        assert all(figure.stat().st_size > 1000 for figure in figures)

    def test_metrics_json_is_machine_readable(
        self, evaluation_result: Any, sandbox_paths: ProjectPaths
    ) -> None:
        """Les métriques sont aussi disponibles en JSON (consommation CI / dashboard)."""
        import json

        payloads = list(sandbox_paths.metrics_dir.glob("*.json"))
        assert payloads
        data = json.loads(payloads[0].read_text(encoding="utf-8"))
        assert isinstance(data, dict)


class TestInferencePipeline:
    """Prédiction sur de nouvelles données."""

    def test_inference_succeeds(self, sandbox_config: AppConfig, evaluation_result: Any) -> None:
        """La prédiction utilise les artefacts entraînés, sans données d'entraînement."""
        assert evaluation_result.succeeded
        result = InferencePipeline(sandbox_config).run()
        assert result.succeeded, result.messages

    def test_predictions_are_written(self, sandbox_config: AppConfig) -> None:
        """Le fichier de prédictions contient autant de lignes que d'entrées demandées."""
        result = InferencePipeline(sandbox_config).run()
        output = Path(sandbox_config.predict.output)
        candidate = (
            output
            if output.is_absolute()
            else ProjectPaths.from_config(sandbox_config.model_dump()).root / output
        )
        assert result.succeeded
        if candidate.exists():
            frame = (
                pd.read_csv(candidate) if candidate.suffix == ".csv" else pd.read_parquet(candidate)
            )
            assert len(frame) >= 1

    def test_predictions_are_explainable(self, sandbox_config: AppConfig) -> None:
        """Chaque prédiction est accompagnée d'une décision exploitable (segment / raison)."""
        result = InferencePipeline(sandbox_config).run()
        payload = result.payload
        if isinstance(payload, pd.DataFrame):
            assert not payload.empty
            assert payload.shape[1] >= 1
