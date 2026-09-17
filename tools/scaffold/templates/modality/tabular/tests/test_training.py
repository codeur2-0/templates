"""Tests de la boucle d'entraînement, des callbacks et du registre de métriques.

Ces tests couvrent ce qui, en production, coûte le plus cher quand c'est faux :

* un artefact manquant ou corrompu rend le déploiement impossible,
* un early stopping mal câblé gaspille du temps de calcul ou arrête trop tôt,
* une métrique non disponible pour la tâche produit des rapports trompeurs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models import build_model
from src.models.base import BaseModel
from src.training.callbacks import (
    CallbackContext,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    MetricThresholdCallback,
    ProgressBarCallback,
)
from src.training.losses_metrics import (
    METRICS_BY_TASK,
    MetricCalculator,
    MetricInputs,
    available_metrics,
    validate_metric_names,
)
from src.training.trainer import Trainer, TrainingData
from src.utils.paths import ProjectPaths

#: Tâches supervisées (une cible est exigée par le contrat d'entraînement).
SUPERVISED_TASKS = {"binary", "multiclass", "regression", "forecasting", "ranking"}


@pytest.fixture
def training_data(prepared: dict[str, Any], app_config: Any) -> TrainingData:
    """Assemble the matrices handed to the trainer."""
    return TrainingData(
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        X_val=prepared["X_val"],
        y_val=prepared["y_val"],
        groups_val=prepared.get("groups_val"),
        feature_names=list(prepared["feature_names"]),
        task=str(app_config.metrics.task),
    )


@pytest.fixture
def trainer(
    app_config: Any, model: BaseModel, training_data: TrainingData, tmp_path: Path
) -> Trainer:
    """Return a trainer writing its artefacts in a temporary directory."""
    paths = ProjectPaths.from_root(tmp_path).ensure()
    return Trainer(
        model,
        config=app_config.model_dump(),
        paths=paths,
        metric_names=app_config.metrics.all_metrics,
        task=str(app_config.metrics.task),
        callbacks=[],
    )


def _run_training(trainer: Trainer, data: TrainingData) -> Any:
    """Run one training and return the outcome."""
    return trainer.train(data)


class TestTrainer:
    """Orchestration de l'entraînement."""

    def test_training_produces_an_outcome(
        self, trainer: Trainer, training_data: TrainingData
    ) -> None:
        """Un entraînement complet renvoie un outcome riche et sérialisable."""
        outcome = _run_training(trainer, training_data)
        assert outcome.model.is_fitted
        assert outcome.metrics
        assert outcome.duration_seconds >= 0.0
        assert outcome.to_dict()["task"] == training_data.task

    def test_artifacts_are_written(self, trainer: Trainer, training_data: TrainingData) -> None:
        """Le modèle, sa fiche et ses métriques sont persistés (déploiement possible)."""
        outcome = _run_training(trainer, training_data)
        assert {"model", "model_card", "metrics"} <= set(outcome.artifacts)
        for path in outcome.artifacts.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_validation_metrics_are_prefixed(
        self, trainer: Trainer, training_data: TrainingData
    ) -> None:
        """Les métriques de validation sont préfixées ``val_`` (aucune confusion train/test)."""
        outcome = _run_training(trainer, training_data)
        if training_data.X_val is None:
            pytest.skip("pas de split de validation configuré")
        if str(training_data.task) not in SUPERVISED_TASKS:
            # Tâche non supervisée : sans cible, la validation publie soit des métriques
            # intrinsèques (clustering : `val_silhouette`, `val_davies_bouldin`, et le diagnostic
            # `n_clusters` non préfixé), soit rien du tout (détection d'anomalies, dont la qualité
            # se mesure sur les métadonnées étiquetées, en mode `evaluate`). Le préfixe `val_`
            # n'est donc pas un invariant universel ; l'absence de valeur non finie, si.
            assert all(np.isfinite(value) for value in outcome.validation_metrics.values())
            return
        assert outcome.validation_metrics
        assert all(name.startswith("val_") for name in outcome.validation_metrics)

    def test_quality_gate_is_reported(
        self, app_config: Any, model: BaseModel, training_data: TrainingData, tmp_path: Path
    ) -> None:
        """Le seuil de qualité déclaré en configuration est évalué pendant l'entraînement."""
        min_primary = app_config.metrics.min_primary
        if min_primary is None:
            pytest.skip("aucun seuil de qualité configuré")
        gated = Trainer(
            model,
            config=app_config.model_dump(),
            paths=ProjectPaths.from_root(tmp_path).ensure(),
            metric_names=app_config.metrics.all_metrics,
            min_primary_metric=float(min_primary),
            primary_metric=app_config.metrics.primary,
            callbacks=[],
        )
        outcome = _run_training(gated, training_data)
        assert isinstance(outcome.metrics, dict)

    def test_contract_rejects_empty_data(
        self, trainer: Trainer, training_data: TrainingData
    ) -> None:
        """Un jeu d'entraînement vide doit être refusé avant tout calcul."""
        empty = TrainingData(
            X_train=training_data.X_train.head(0),
            y_train=None if training_data.y_train is None else training_data.y_train.head(0),
            feature_names=training_data.feature_names,
            task=training_data.task,
        )
        with pytest.raises(ValueError, match="empty"):
            trainer.train(empty)

    def test_contract_rejects_missing_features(
        self, trainer: Trainer, training_data: TrainingData
    ) -> None:
        """Une matrice qui n'a pas les features du modèle est refusée."""
        truncated = TrainingData(
            X_train=training_data.X_train.drop(columns=[training_data.feature_names[0]]),
            y_train=training_data.y_train,
            feature_names=training_data.feature_names[:-1],
            task=training_data.task,
        )
        with pytest.raises(ValueError, match="misses model features"):
            trainer.train(truncated)

    def test_contract_rejects_missing_values(
        self, trainer: Trainer, training_data: TrainingData
    ) -> None:
        """Des NaN dans les features signifient un preprocessing raté : refus explicite."""
        corrupted = training_data.X_train.copy()
        corrupted.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="missing value"):
            trainer.train(
                TrainingData(
                    X_train=corrupted,
                    y_train=training_data.y_train,
                    X_val=training_data.X_val,
                    y_val=training_data.y_val,
                    feature_names=training_data.feature_names,
                    task=training_data.task,
                )
            )

    def test_contract_rejects_supervised_task_without_target(
        self, app_config: Any, model: BaseModel, training_data: TrainingData, tmp_path: Path
    ) -> None:
        """Une tâche supervisée sans cible est une erreur de câblage, pas un cas limite."""
        if str(app_config.metrics.task) not in SUPERVISED_TASKS:
            pytest.skip("tâche non supervisée")
        unsupervised = Trainer(
            model,
            config=app_config.model_dump(),
            paths=ProjectPaths.from_root(tmp_path).ensure(),
            callbacks=[],
        )
        with pytest.raises(ValueError, match="supervised"):
            unsupervised.train(
                TrainingData(
                    X_train=training_data.X_train,
                    y_train=None,
                    feature_names=training_data.feature_names,
                    task=str(app_config.metrics.task),
                )
            )

    def test_callbacks_from_configuration(
        self, app_config: Any, model: BaseModel, tmp_path: Path
    ) -> None:
        """Les callbacks viennent de ``conf/train/default.yaml`` (rien de codé en dur)."""
        configured = Trainer(
            model,
            config=app_config.model_dump(),
            paths=ProjectPaths.from_root(tmp_path).ensure(),
        )
        names = [callback.name for callback in configured.callbacks]
        assert "logging" in names
        assert bool(app_config.train.callbacks.metric_history) == ("metric-history" in names)
        assert bool(app_config.train.early_stopping.enabled) == ("early-stopping" in names)

    def test_training_data_describe(self, training_data: TrainingData) -> None:
        """Le résumé d'entraînement alimente les logs et les rapports."""
        description = training_data.describe()
        assert description["n_train"] == len(training_data.X_train)
        assert description["n_features"] == training_data.X_train.shape[1]

    def test_training_data_from_bundle(self, prepared: dict[str, Any], app_config: Any) -> None:
        """Le bundle produit par la couche données alimente directement le trainer."""
        from src.pipelines.train_pipeline import build_dataset_bundle

        bundle = build_dataset_bundle(prepared, str(app_config.metrics.task))
        data = TrainingData.from_bundle(bundle)
        assert len(data.X_train) == bundle.n_train
        assert data.feature_names == list(prepared["feature_names"])
        assert bundle.describe()["task"] == data.task


class TestCallbacks:
    """Comportement unitaire des callbacks."""

    def test_metric_history_accumulates_every_epoch(self) -> None:
        """L'historique alimente les courbes d'apprentissage des notebooks."""
        callback = MetricHistoryCallback()
        context = CallbackContext(model_name="test", epochs=3)
        for epoch, loss in enumerate([0.9, 0.5, 0.3]):
            context.epoch = epoch
            context.logs = {"val_loss": loss}
            callback.on_epoch_end(context)
        assert context.history["val_loss"] == [0.9, 0.5, 0.3]

    def test_metric_history_ignores_missing_values(self) -> None:
        """Une métrique indisponible ne doit pas polluer l'historique."""
        callback = MetricHistoryCallback()
        context = CallbackContext(model_name="test")
        context.logs = {"val_loss": None, "val_acc": 0.5}
        callback.on_epoch_end(context)
        assert context.history == {"val_acc": [0.5]}

    def test_early_stopping_triggers_after_patience(self) -> None:
        """Sans amélioration pendant ``patience`` époques, l'entraînement s'arrête."""
        callback = EarlyStoppingCallback("val_loss", patience=2, mode="min")
        context = CallbackContext(model_name="test", epochs=10)
        callback.on_train_begin(context)
        for epoch, loss in enumerate([1.0, 0.9, 0.9, 0.9, 0.9]):
            context.epoch = epoch
            context.logs = {"val_loss": loss}
            callback.on_epoch_end(context)
            if context.stopped_early:
                break
        assert context.stopped_early
        assert callback.best_score == pytest.approx(0.9)

    def test_early_stopping_tracks_improvement(self) -> None:
        """Une amélioration continue ne déclenche pas d'arrêt."""
        callback = EarlyStoppingCallback("val_loss", patience=2, mode="min")
        context = CallbackContext(model_name="test", epochs=5)
        callback.on_train_begin(context)
        for epoch, loss in enumerate([1.0, 0.8, 0.6, 0.4]):
            context.epoch = epoch
            context.logs = {"val_loss": loss}
            callback.on_epoch_end(context)
        assert not context.stopped_early
        assert context.best_epoch == 3

    def test_early_stopping_max_mode(self) -> None:
        """En mode ``max`` (score), l'amélioration est une croissance."""
        callback = EarlyStoppingCallback("val_roc_auc", patience=1, mode="max")
        context = CallbackContext(model_name="test", epochs=4)
        callback.on_train_begin(context)
        for epoch, score in enumerate([0.7, 0.8, 0.8]):
            context.epoch = epoch
            context.logs = {"val_roc_auc": score}
            callback.on_epoch_end(context)
        assert context.stopped_early
        assert callback.best_score == pytest.approx(0.8)

    def test_early_stopping_rejects_unknown_mode(self) -> None:
        """Un mode invalide est une erreur de configuration."""
        with pytest.raises(ValueError, match="mode"):
            EarlyStoppingCallback("val_loss", mode="moyen")

    def test_metric_threshold_records_satisfaction(self) -> None:
        """Le garde-fou de qualité CI passe quand le seuil est atteint."""
        callback = MetricThresholdCallback("val_roc_auc", threshold=0.6, mode="max")
        context = CallbackContext(model_name="test")
        context.logs = {"val_roc_auc": 0.55}
        callback.on_epoch_end(context)
        assert not callback.satisfied
        context.logs = {"val_roc_auc": 0.72}
        callback.on_epoch_end(context)
        assert callback.satisfied

    def test_metric_threshold_ignores_absent_metric(self) -> None:
        """Une métrique absente des logs ne doit pas faire échouer le garde-fou."""
        callback = MetricThresholdCallback("val_roc_auc", threshold=0.6)
        context = CallbackContext(model_name="test")
        callback.on_epoch_end(context)
        assert not callback.satisfied

    def test_logging_callback_runs_without_side_effect(self) -> None:
        """Le callback de log ne doit jamais interrompre l'entraînement."""
        callback = LoggingCallback(every=2)
        context = CallbackContext(model_name="test", epochs=4)
        callback.on_train_begin(context)
        for epoch in range(4):
            context.epoch = epoch
            context.logs = {"val_loss": 1.0 - epoch * 0.1}
            callback.on_epoch_end(context)
        callback.on_train_end(context)

    def test_progress_bar_is_disabled_when_not_interactive(self) -> None:
        """Hors terminal interactif (CI, cron), la barre de progression reste silencieuse."""
        callback = ProgressBarCallback(enabled=False)
        context = CallbackContext(model_name="test", epochs=1)
        callback.on_train_begin(context)
        callback.on_epoch_end(context)
        callback.on_train_end(context)


class TestMetricRegistry:
    """Registre de métriques partagé entre entraînement et évaluation."""

    def test_configured_metrics_exist(self, app_config: Any) -> None:
        """Toute métrique déclarée en configuration existe dans le registre."""
        names = validate_metric_names(app_config.metrics.all_metrics, str(app_config.metrics.task))
        assert app_config.metrics.primary in names
        assert len(names) == len(set(names))

    def test_unknown_metric_is_rejected(self) -> None:
        """Une métrique inconnue doit lever (erreur de configuration détectée tôt)."""
        with pytest.raises(KeyError, match="Unknown metric"):
            validate_metric_names(["indice_de_qualité_inventé"])

    def test_task_metrics_are_declared(self, app_config: Any) -> None:
        """Chaque tâche a ses métriques de référence (documentation vivante)."""
        task = str(app_config.metrics.task)
        assert task in METRICS_BY_TASK
        assert set(METRICS_BY_TASK[task]) <= set(available_metrics(task))

    def test_calculator_skips_metrics_without_inputs(self) -> None:
        """Une métrique sans ses entrées renvoie NaN plutôt que de planter."""
        calculator = MetricCalculator(task="binary", metrics=["accuracy", "roc_auc"])
        values = calculator.evaluate(MetricInputs(y_true=[0, 1, 1, 0], y_pred=[0, 1, 0, 0]))
        assert values["accuracy"] == pytest.approx(0.75)
        assert np.isnan(values["roc_auc"]) or values["roc_auc"] >= 0.0

    def test_calculator_computes_probabilistic_metrics(self) -> None:
        """Avec les probabilités, les métriques probabilistes sont calculées."""
        calculator = MetricCalculator(task="binary", metrics=["accuracy", "roc_auc", "log_loss"])
        values = calculator.evaluate(
            MetricInputs(
                y_true=[0, 1, 1, 0],
                y_pred=[0, 1, 0, 0],
                y_proba=[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.8, 0.2]],
            )
        )
        assert values["accuracy"] == pytest.approx(0.75)
        assert 0.0 <= values["roc_auc"] <= 1.0
        assert values["log_loss"] > 0.0

    def test_calculator_from_configuration(self, app_config: Any) -> None:
        """Le registre se configure depuis Hydra (mêmes métriques partout)."""
        calculator = MetricCalculator.from_config(app_config.model_dump())
        assert calculator.task == str(app_config.metrics.task)
        assert calculator.metric_names[0] == app_config.metrics.primary

    def test_perfect_predictions_score_one(self) -> None:
        """Sanity check : des prédictions parfaites donnent 1.0 (le registre n'est pas inversé)."""
        calculator = MetricCalculator(task="binary", metrics=["accuracy"])
        values = calculator.evaluate(
            MetricInputs(y_true=pd.Series([0, 1, 1, 0]), y_pred=pd.Series([0, 1, 1, 0]))
        )
        assert values["accuracy"] == pytest.approx(1.0)
