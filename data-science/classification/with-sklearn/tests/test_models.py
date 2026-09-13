"""Tests du contrat modèle (:class:`src.models.base.BaseModel`) et de son implémentation.

Le contrat est ce qui rend le framework interchangeable : si ces tests passent, on peut
remplacer l'implémentation (sklearn → XGBoost → PyTorch) sans toucher au reste du projet.
Ils vérifient donc le *comportement*, pas l'algorithme :

* un modèle non entraîné refuse de prédire (pas de résultat silencieusement faux),
* ``predict`` / ``predict_proba`` ont les formes attendues,
* la persistance est fidèle (mêmes prédictions après ``save`` / ``load``),
* l'entraînement est reproductible à graine fixée.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models import build_model, load_model
from src.models.base import BaseModel, FitResult

#: Tâches qui nécessitent une cible (les tâches non supervisées sautent les tests concernés).
SUPERVISED_TASKS = {"binary", "multiclass", "regression", "forecasting", "ranking"}
#: Tâches dont les prédictions sont des probabilités normalisées.
PROBA_TASKS = {"binary", "multiclass"}


def _task(app_config: Any) -> str:
    """Return the learning task declared in the configuration."""
    return str(app_config.metrics.task)


class TestModelContract:
    """Le contrat partagé par toutes les implémentations."""

    def test_base_model_cannot_be_instantiated(self) -> None:
        """``BaseModel`` est abstrait : il impose ``fit`` / ``predict`` / ``save`` / ``load``."""
        with pytest.raises(TypeError):
            BaseModel(feature_names=["a"], target_name="y")  # type: ignore[abstract]

    def test_model_declares_its_identity(self, model: BaseModel, app_config: Any) -> None:
        """Framework, tâche et cible sont déclarés (utilisés par les rapports et l'inférence)."""
        assert model.framework and model.framework != "base"
        assert model.task == _task(app_config)
        assert model.target_name == app_config.data.target or model.target_name

    def test_model_records_the_feature_contract(
        self, model: BaseModel, prepared: dict[str, Any]
    ) -> None:
        """Le modèle connaît exactement les features attendues, dans l'ordre."""
        assert model.feature_names == prepared["feature_names"]

    def test_unfitted_model_refuses_to_predict(
        self, model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Prédire avec un modèle non entraîné doit lever, pas renvoyer n'importe quoi."""
        assert not model.is_fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            model.check_is_fitted()
        with pytest.raises(RuntimeError):
            model.predict(matrices["X_test"])

    def test_summary_is_human_readable(self, fitted_model: BaseModel) -> None:
        """``summary()`` alimente les logs : il mentionne l'état d'entraînement."""
        text = fitted_model.summary()
        assert type(fitted_model).__name__ in text
        assert "fitted" in text


class TestFitting:
    """Entraînement et résultat."""

    def test_fit_returns_a_fit_result(self, fitted_model: BaseModel) -> None:
        """L'entraînement produit un :class:`FitResult` exploitable (métriques, durée)."""
        result = fitted_model.fit_result_
        assert isinstance(result, FitResult)
        assert result.n_samples > 0
        assert result.n_features == len(fitted_model.feature_names)
        assert result.duration_seconds >= 0.0
        assert result.finished_at

    def test_fit_result_is_json_serialisable(self, fitted_model: BaseModel, tmp_path: Path) -> None:
        """Le résultat doit pouvoir être archivé (traçabilité des runs)."""
        assert fitted_model.fit_result_ is not None
        payload = fitted_model.fit_result_.to_dict()
        path = fitted_model.fit_result_.to_json(tmp_path / "fit.json")
        assert json.loads(path.read_text(encoding="utf-8"))["model_name"] == payload["model_name"]

    def test_training_is_reproducible(self, app_config: Any, prepared: dict[str, Any]) -> None:
        """À configuration et graine fixées, deux entraînements donnent les mêmes prédictions."""
        if _task(app_config) not in SUPERVISED_TASKS:
            pytest.skip("reproductibilité vérifiée sur les tâches supervisées")
        first = build_model(app_config, feature_names=prepared["feature_names"])
        second = build_model(app_config, feature_names=prepared["feature_names"])
        for candidate in (first, second):
            candidate.fit(
                prepared["X_train"],
                prepared["y_train"],
                X_val=prepared["X_val"],
                y_val=prepared["y_val"],
                callbacks=[],
            )
        np.testing.assert_allclose(
            first.predict(prepared["X_test"]).astype("float64"),
            second.predict(prepared["X_test"]).astype("float64"),
        )

    def test_model_can_be_refitted(self, app_config: Any, prepared: dict[str, Any]) -> None:
        """Ré-entraîner un modèle déjà entraîné doit fonctionner (warm restart)."""
        if _task(app_config) not in SUPERVISED_TASKS:
            pytest.skip("ré-entraînement vérifié sur les tâches supervisées")
        candidate = build_model(app_config, feature_names=prepared["feature_names"])
        candidate.fit(prepared["X_train"], prepared["y_train"], callbacks=[])
        candidate.fit(prepared["X_train"], prepared["y_train"], callbacks=[])
        assert candidate.is_fitted


class TestPrediction:
    """Sorties de prédiction."""

    def test_predict_returns_one_value_per_row(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """``predict`` renvoie un vecteur aligné sur les lignes, sans NaN."""
        predictions = fitted_model.predict(matrices["X_test"])
        assert np.asarray(predictions).shape == (len(matrices["X_test"]),)
        assert np.isfinite(np.asarray(predictions, dtype="float64")).all()

    def test_predictions_stay_in_the_observed_label_space(
        self, fitted_model: BaseModel, matrices: dict[str, Any], app_config: Any
    ) -> None:
        """Les classes prédites sont un sous-ensemble des classes d'entraînement."""
        if _task(app_config) not in {"binary", "multiclass"}:
            pytest.skip("espace de labels propre à la classification")
        predicted = set(np.unique(fitted_model.predict(matrices["X_test"])).tolist())
        observed = set(np.unique(matrices["y_train"]).tolist())
        assert predicted <= observed

    def test_predict_proba_is_normalised(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Les probabilités sont positives et somment à 1 (contrat des modèles probabilistes)."""
        if not fitted_model.supports_proba:
            pytest.skip("ce modèle n'expose pas de probabilités")
        probabilities = fitted_model.predict_proba(matrices["X_test"])
        assert probabilities.shape[0] == len(matrices["X_test"])
        assert probabilities.shape[1] >= 2
        assert (probabilities >= 0.0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_is_refused_when_unsupported(
        self, model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Sans support des probabilités, l'appel doit lever explicitement."""
        if model.supports_proba:
            pytest.skip("ce modèle supporte predict_proba")
        with pytest.raises(NotImplementedError):
            model.predict_proba(matrices["X_test"])

    def test_prediction_on_a_single_row(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Le scénario d'inférence unitaire (une requête) doit fonctionner."""
        single = matrices["X_test"].head(1)
        assert np.asarray(fitted_model.predict(single)).shape == (1,)

    def test_prediction_is_deterministic_at_inference(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Deux appels sur les mêmes données donnent exactement le même résultat."""
        first = np.asarray(fitted_model.predict(matrices["X_test"]), dtype="float64")
        second = np.asarray(fitted_model.predict(matrices["X_test"]), dtype="float64")
        np.testing.assert_allclose(first, second)


class TestFeatureAlignment:
    """Robustesse aux frames mal formés."""

    def test_missing_feature_is_refused(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Une feature manquante doit être nommée dans l'erreur (diagnostic immédiat)."""
        truncated = matrices["X_test"].drop(columns=[fitted_model.feature_names[0]])
        with pytest.raises(ValueError, match="missing"):
            fitted_model.align_features(truncated)

    def test_extra_columns_are_ignored_and_order_restored(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """Colonnes en trop ignorées, ordre restauré : l'inférence ne dépend pas du payload."""
        noisy = matrices["X_test"].copy()
        noisy["colonne_inutile"] = 1.0
        reordered = noisy[list(reversed(noisy.columns))]
        aligned = fitted_model.align_features(reordered)
        assert list(aligned.columns) == fitted_model.feature_names

    def test_predict_accepts_reordered_columns(
        self, fitted_model: BaseModel, matrices: dict[str, Any]
    ) -> None:
        """``predict`` aligne lui-même les colonnes (l'appelant n'a pas à le faire)."""
        reordered = matrices["X_test"][list(reversed(fitted_model.feature_names))]
        np.testing.assert_allclose(
            np.asarray(fitted_model.predict(reordered), dtype="float64"),
            np.asarray(fitted_model.predict(matrices["X_test"]), dtype="float64"),
        )


class TestPersistence:
    """Sauvegarde et rechargement."""

    def test_save_load_roundtrip_preserves_predictions(
        self, fitted_model: BaseModel, matrices: dict[str, Any], tmp_path: Path
    ) -> None:
        """Le modèle rechargé prédit exactement comme le modèle en mémoire."""
        path = fitted_model.save(tmp_path / "model.joblib")
        assert Path(path).exists()
        restored = type(fitted_model).load(path)
        assert restored.is_fitted
        assert restored.feature_names == fitted_model.feature_names
        np.testing.assert_allclose(
            np.asarray(restored.predict(matrices["X_test"]), dtype="float64"),
            np.asarray(fitted_model.predict(matrices["X_test"]), dtype="float64"),
        )

    def test_load_model_helper_returns_a_base_model(
        self, fitted_model: BaseModel, tmp_path: Path
    ) -> None:
        """Le helper ``load_model`` respecte le contrat polymorphe."""
        path = fitted_model.save(tmp_path / "model.joblib")
        restored = load_model(path)
        assert isinstance(restored, BaseModel)

    def test_load_rejects_a_foreign_artifact(self, tmp_path: Path) -> None:
        """Charger un artefact qui n'est pas un modèle doit échouer proprement."""
        import joblib

        foreign = tmp_path / "not_a_model.joblib"
        joblib.dump({"pas": "un modèle"}, foreign)
        with pytest.raises((TypeError, ValueError)):
            load_model(foreign)

    def test_model_card_documents_the_run(self, fitted_model: BaseModel, tmp_path: Path) -> None:
        """La fiche modèle est sérialisable et trace features, paramètres et versions."""
        card = fitted_model.model_card(metrics={"roc_auc": 0.5})
        payload = json.loads(card.to_json(tmp_path / "card.json").read_text(encoding="utf-8"))
        assert payload["feature_names"] == fitted_model.feature_names
        assert payload["framework"] == fitted_model.framework
        assert payload["metrics"] == {"roc_auc": 0.5}
        assert "python" in payload["library_versions"]


class TestModelFactory:
    """Construction depuis la configuration Hydra."""

    def test_build_model_uses_the_configuration(
        self, app_config: Any, prepared: dict[str, Any]
    ) -> None:
        """L'algorithme, la graine et les hyperparamètres viennent de la configuration."""
        candidate = build_model(app_config, feature_names=prepared["feature_names"])
        assert isinstance(candidate, BaseModel)
        assert candidate.params == app_config.model.params
        assert candidate.random_state in {app_config.model.random_state, app_config.seed}

    def test_build_model_rejects_an_empty_configuration(self) -> None:
        """Une configuration sans noeud ``model`` est refusée avec un message clair."""
        with pytest.raises(ValueError, match="model"):
            build_model({})

    def test_build_model_defaults_feature_names(self, app_config: Any) -> None:
        """Sans ``feature_names``, le modèle reste constructible (cas des notebooks)."""
        candidate = build_model(app_config)
        assert isinstance(candidate, BaseModel)
        assert isinstance(candidate.feature_names, list)

    def test_every_configured_algorithm_can_be_built(
        self, app_config: Any, prepared: dict[str, Any]
    ) -> None:
        """L'algorithme déclaré dans ``conf/model/default.yaml`` est bien supporté."""
        candidate = build_model(app_config, feature_names=prepared["feature_names"])
        candidate.fit(
            prepared["X_train"],
            prepared["y_train"],
            X_val=prepared["X_val"],
            y_val=prepared["y_val"],
            callbacks=[],
        )
        assert isinstance(candidate.predict(prepared["X_test"]), np.ndarray | pd.Series)
