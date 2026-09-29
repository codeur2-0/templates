"""Tests de la couche tâche multi-classes : décision, évaluation, inférence et rapport.

Les tests partagés (``test_models.py``, ``test_pipeline.py``…) vérifient le contrat commun à toutes
les tâches. Ceux-ci vérifient ce qui n'existe qu'en multi-classes :

* la **politique de décision** (``diagnosis``) : validation, matrice de coûts, décision à coût
  minimal, coût réalisé d'une revue experte ;
* l'**évaluateur** : ordre métier des classes, références, verdict, robustesse à une classe absente
  du split et refus d'une classe inconnue du modèle ;
* le **predictor** : colonnes publiées, probabilités normalisées, cohérence de la revue experte ;
* le **rapport** : artefacts écrits et sections attendues.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation.decision import (
    DiagnosisSettings,
    minimum_cost_decision,
    realised_costs,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.reports import ReportBuilder
from src.inference.predictor import REVIEW_TEAM, Predictor
from src.models.base import BaseModel
from src.utils.paths import ProjectPaths


def _config(app_config: Any) -> dict[str, Any]:
    """Return the root configuration as a plain mapping."""
    return app_config.model_dump()


def _diagnosis(app_config: Any) -> dict[str, Any]:
    """Return a copy of the ``diagnosis`` block of the configuration."""
    return json.loads(json.dumps(_config(app_config)["diagnosis"]))


@pytest.fixture(scope="module")
def settings(app_config: Any, fitted_model: BaseModel) -> DiagnosisSettings:
    """Decision settings resolved against the classes of the fitted model."""
    return DiagnosisSettings.resolve(_config(app_config), classes=list(fitted_model.classes_))


@pytest.fixture(scope="module")
def evaluation(app_config: Any, fitted_model: BaseModel, matrices: dict[str, Any], enriched_splits: dict[str, pd.DataFrame], project_paths: ProjectPaths) -> Any:
    """Evaluate the fitted model on the tiny test split, with its raw context."""
    evaluator = Evaluator.from_config(fitted_model, _config(app_config), project_paths)
    return evaluator.evaluate(matrices["X_test"], matrices["y_test"], split="test", context=enriched_splits["test"])


class TestDiagnosisSettings:
    """Le bloc ``diagnosis`` est le contrat métier : il doit échouer bruyamment s'il est faux."""

    def test_class_order_follows_the_configuration(self, settings: DiagnosisSettings, app_config: Any) -> None:
        assert list(settings.class_order) == list(_diagnosis(app_config)["class_order"])

    def test_cost_matrix_encodes_the_business_asymmetry(self, settings: DiagnosisSettings) -> None:
        labels = list(settings.class_order)
        matrix = settings.cost_matrix(labels)
        costs = settings.costs
        no_failure = labels.index(str(settings.no_failure_class))
        failure = next(index for index, label in enumerate(labels) if index != no_failure)
        assert matrix.shape == (len(labels), len(labels))
        assert matrix[no_failure, no_failure] == costs.correct_acknowledgement
        assert matrix[failure, failure] == costs.correct_intervention
        assert matrix[failure, no_failure] == costs.missed_failure
        assert matrix[no_failure, failure] == costs.useless_dispatch
        # Acquitter une panne réelle doit rester l'erreur la plus chère de la matrice.
        assert matrix[failure, no_failure] == matrix.max()

    def test_unknown_class_in_the_routing_rule_is_refused(self, app_config: Any) -> None:
        diagnosis = _diagnosis(app_config)
        diagnosis["alarm_code_rule"] = {"E999": "classe_inexistante"}
        with pytest.raises(ValueError, match="unknown class"):
            DiagnosisSettings.resolve({"diagnosis": diagnosis})

    def test_review_threshold_out_of_range_is_refused(self, app_config: Any) -> None:
        diagnosis = _diagnosis(app_config)
        diagnosis["review_threshold"] = 1.2
        with pytest.raises(ValueError, match="review_threshold"):
            DiagnosisSettings.resolve({"diagnosis": diagnosis})

    def test_negative_cost_is_refused(self, app_config: Any) -> None:
        diagnosis = _diagnosis(app_config)
        diagnosis["costs"] = {**diagnosis["costs"], "wrong_team": -1.0}
        with pytest.raises(ValueError, match=">= 0"):
            DiagnosisSettings.resolve({"diagnosis": diagnosis})

    def test_model_class_absent_from_the_order_is_refused(self, app_config: Any) -> None:
        with pytest.raises(ValueError, match=r"absent from diagnosis\.class_order"):
            DiagnosisSettings.resolve(_config(app_config), classes=["classe_fantome"])


class TestDecision:
    """La décision à coût minimal n'est pas l'argmax dès que les erreurs n'ont pas le même prix."""

    def test_minimum_cost_prefers_a_dispatch_over_a_risky_acknowledgement(self, settings: DiagnosisSettings) -> None:
        labels = list(settings.class_order)
        no_failure = str(settings.no_failure_class)
        failure = next(label for label in labels if label != no_failure)
        probabilities = np.zeros((1, len(labels)))
        probabilities[0, labels.index(no_failure)] = 0.55
        probabilities[0, labels.index(failure)] = 0.45
        decided = minimum_cost_decision(probabilities, labels, settings)
        # L'argmax acquitterait l'alarme ; le coût attendu d'une panne manquée l'interdit.
        assert decided[0] == failure

    def test_minimum_cost_follows_a_confident_prediction(self, settings: DiagnosisSettings) -> None:
        labels = list(settings.class_order)
        probabilities = np.full((len(labels), len(labels)), 0.001)
        np.fill_diagonal(probabilities, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        assert list(minimum_cost_decision(probabilities, labels, settings)) == labels

    def test_a_reviewed_alarm_costs_the_review_plus_the_right_action(self, settings: DiagnosisSettings) -> None:
        labels = list(settings.class_order)
        no_failure = str(settings.no_failure_class)
        failure = next(label for label in labels if label != no_failure)
        costs = realised_costs([failure], [no_failure], labels, settings, reviewed=np.array([True]))
        assert costs[0] == pytest.approx(settings.costs.correct_intervention + settings.costs.expert_review)


class TestEvaluator:
    """Ce que l'évaluateur multi-classes doit produire, et ce qu'il doit refuser."""

    def test_labels_follow_the_business_order(self, evaluation: Any, settings: DiagnosisSettings) -> None:
        expected = [label for label in settings.class_order if label in set(evaluation.labels)]
        assert evaluation.labels == expected

    def test_probability_columns_are_normalised(self, evaluation: Any) -> None:
        columns = [f"proba_{label}" for label in evaluation.labels]
        np.testing.assert_allclose(evaluation.predictions[columns].sum(axis=1), 1.0, atol=1e-6)

    def test_references_include_the_current_routing_rule(self, evaluation: Any) -> None:
        names = set(evaluation.references["référence"])
        assert {"modèle", "classe majoritaire", "routage actuel (code automate)"} <= names

    def test_verdict_covers_every_objective(self, evaluation: Any, settings: DiagnosisSettings) -> None:
        assert len(evaluation.verdict) == 7
        assert set(evaluation.verdict["status"]) <= {"atteint", "non atteint", "n/a"}

    def test_decision_policies_are_compared_on_the_same_alarms(self, evaluation: Any) -> None:
        policies = set(evaluation.decision["politique"])
        assert {"argmax", "coût minimal"} <= policies
        assert (evaluation.decision["cost_per_alarm"] > 0).all()

    def test_a_class_missing_from_the_split_yields_finite_metrics(self, app_config: Any, fitted_model: BaseModel, matrices: dict[str, Any], project_paths: ProjectPaths) -> None:
        truth = pd.Series(np.asarray(matrices["y_test"])).astype(str)
        rarest = truth.value_counts().idxmin()
        keep = (truth != rarest).to_numpy()
        evaluator = Evaluator.from_config(fitted_model, _config(app_config), project_paths)
        result = evaluator.evaluate(matrices["X_test"].loc[keep], truth[keep].to_numpy())
        # Les probabilités couvrent toujours toutes les classes du modèle : log loss et AUC restent
        # calculables au lieu de devenir NaN.
        assert np.isfinite(result.metrics["log_loss"])
        assert np.isfinite(result.metrics["f1_macro"])

    def test_an_unknown_label_is_refused(self, app_config: Any, fitted_model: BaseModel, matrices: dict[str, Any], project_paths: ProjectPaths) -> None:
        truth = np.asarray(matrices["y_test"]).astype(object)
        truth[0] = "mode_inconnu"
        evaluator = Evaluator.from_config(fitted_model, _config(app_config), project_paths)
        with pytest.raises(ValueError, match="unknown to the model"):
            evaluator.evaluate(matrices["X_test"], truth)

    def test_result_is_json_serialisable(self, evaluation: Any) -> None:
        payload = json.loads(json.dumps(evaluation.to_dict(), default=str))
        assert payload["task"] == "multiclass"
        assert payload["verdict"]


class TestPredictor:
    """Le predictor publie une décision, pas seulement une classe."""

    @pytest.fixture(scope="class")
    def predictor(self, app_config: Any, fitted_model: BaseModel, preprocessing: Any, feature_builder: Any) -> Predictor:
        return Predictor(model=fitted_model, preprocessing=preprocessing, feature_builder=feature_builder, config=_config(app_config))

    @pytest.fixture(scope="class")
    def payload(self, split_frames: Any, app_config: Any) -> pd.DataFrame:
        return split_frames.test.drop(columns=[app_config.data.target]).head(25).reset_index(drop=True)

    def test_published_columns(self, predictor: Predictor, payload: pd.DataFrame) -> None:
        output = predictor.predict(payload)
        expected = {"predicted_mode", "second_choice", "confidence", "margin", "decision", "routed_team", "needs_review", "decision_reason"}
        assert expected <= set(output.columns)
        assert {f"proba_{label}" for label in predictor.labels} <= set(output.columns)
        assert len(output) == len(payload)

    def test_probabilities_sum_to_one_in_business_order(self, predictor: Predictor, payload: pd.DataFrame) -> None:
        output = predictor.predict(payload)
        columns = [f"proba_{label}" for label in predictor.labels]
        np.testing.assert_allclose(output[columns].sum(axis=1), 1.0, atol=1e-3)
        assert predictor.labels == predictor.settings.ordered(predictor.classes)

    def test_review_flag_matches_the_threshold(self, predictor: Predictor, payload: pd.DataFrame) -> None:
        output = predictor.predict(payload)
        threshold = predictor.settings.review_threshold
        assert (output["needs_review"] == (output["confidence"] < threshold)).all()
        assert (output.loc[output["needs_review"], "routed_team"] == REVIEW_TEAM).all()
        assert set(output["decision"]) <= set(predictor.labels)

    def test_predict_one_is_json_serialisable(self, predictor: Predictor, payload: pd.DataFrame) -> None:
        record = payload.iloc[0].to_dict()
        response = predictor.predict_one(record)
        encoded = json.dumps(response, default=str)
        assert response["decision"] in predictor.labels
        assert sum(response["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert "decision_reason" in json.loads(encoded)


class TestReport:
    """Le rapport écrit ses artefacts et expose le verdict."""

    def test_report_artefacts(self, evaluation: Any, app_config: Any, tmp_path: Any) -> None:
        paths = ProjectPaths.from_root(tmp_path).ensure()
        written = ReportBuilder(paths, config=_config(app_config)).build(evaluation)
        report = written["report"].read_text(encoding="utf-8")
        assert "## 2. Verdict par objectif" in report
        assert "## 7. Décision : argmax ou coût minimal" in report
        assert written["json"].exists()
        assert "confusion_matrix" in written
