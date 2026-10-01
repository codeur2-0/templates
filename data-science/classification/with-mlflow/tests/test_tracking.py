"""Tests of the MLflow tracker: one run per training, test metrics on it, reloadable model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest

from src.models.base import BaseModel
from src.schemas.config import AppConfig
from src.tracking.tracker import MlflowTracker, build_tracker, flatten
from src.utils.io import read_json, write_json
from src.utils.paths import ProjectPaths


@pytest.fixture
def tracker(tmp_path: Path, app_config: AppConfig) -> MlflowTracker:
    """Return a tracker writing into a throw-away project layout."""
    return build_tracker(app_config.model_dump(), ProjectPaths.from_root(tmp_path).ensure())


@pytest.fixture
def trained_run(
    tracker: MlflowTracker,
    app_config: AppConfig,
    fitted_model: BaseModel,
    prepared: dict[str, Any],
    tmp_path: Path,
) -> str:
    """Record one training run and return its id."""
    card = write_json(tmp_path / "card.json", {"model": fitted_model.summary()})
    tracker.log_training(
        config=app_config.model_dump(),
        metrics={"val_roc_auc": 0.8, "train_roc_auc": 0.9},
        model=fitted_model,
        artifacts=[card],
        sample=prepared["X_train"],
    )
    return str(read_json(tracker.run_file)["run_id"])


class TestFlatten:
    """Aplatissement de la configuration en paramètres MLflow."""

    def test_nested_keys_become_dotted_names(self) -> None:
        """Chaque feuille devient un paramètre `a.b.c`."""
        assert flatten({"model": {"params": {"n_estimators": 300}}, "seed": 42}) == {
            "model.params.n_estimators": "300",
            "seed": "42",
        }

    def test_long_values_are_truncated(self) -> None:
        """MLflow borne la taille d'une valeur : on tronque plutôt que d'échouer."""
        assert len(flatten({"x": "a" * 5000})["x"]) == 500


class TestTracking:
    """Un run par entraînement, l'évaluation rattachée au même run."""

    def test_backend_is_local(self, tracker: MlflowTracker, tmp_path: Path) -> None:
        """Aucun serveur : la base SQLite vit dans les artefacts du projet."""
        assert tracker.tracking_uri.startswith("sqlite:///")
        assert str(tmp_path.as_posix()) in tracker.tracking_uri

    def test_training_run_records_params_metrics_and_model(
        self, tracker: MlflowTracker, trained_run: str
    ) -> None:
        """Paramètres aplatis, métriques et modèle sont retrouvables dans le run."""
        mlflow.set_tracking_uri(tracker.tracking_uri)
        run = mlflow.get_run(trained_run)
        assert run.data.metrics["val_roc_auc"] == pytest.approx(0.8)
        assert "model.algorithm" in run.data.params
        assert run.data.tags["pipeline"] == "train"

    def test_evaluation_is_attached_to_the_training_run(
        self, tracker: MlflowTracker, trained_run: str
    ) -> None:
        """Les métriques de test arrivent, préfixées, dans le run d'entraînement lui-même."""
        tracker.log_evaluation(metrics={"roc_auc": 0.77}, artifacts=[])
        mlflow.set_tracking_uri(tracker.tracking_uri)
        run = mlflow.get_run(trained_run)
        assert run.data.metrics["test_roc_auc"] == pytest.approx(0.77)
        experiment = mlflow.get_experiment_by_name(tracker.experiment_name)
        assert experiment is not None
        assert len(mlflow.search_runs([experiment.experiment_id])) == 1

    def test_logged_model_predicts_like_the_trained_one(
        self,
        tracker: MlflowTracker,
        trained_run: str,
        fitted_model: BaseModel,
        prepared: dict[str, Any],
    ) -> None:
        """Le modèle rechargé par `mlflow.pyfunc` prédit exactement comme l'estimateur."""
        mlflow.set_tracking_uri(tracker.tracking_uri)
        reloaded = mlflow.pyfunc.load_model(f"runs:/{trained_run}/model")
        sample = prepared["X_val"].head(20)
        expected = np.asarray(fitted_model.estimator_.predict(sample))
        np.testing.assert_array_equal(np.asarray(reloaded.predict(sample)), expected)

    def test_evaluation_without_training_run_is_skipped(self, tracker: MlflowTracker) -> None:
        """Sans run d'entraînement connu, l'évaluation n'invente pas de run."""
        tracker.log_evaluation(metrics={"roc_auc": 0.5}, artifacts=[])
        assert not tracker.run_file.exists()
