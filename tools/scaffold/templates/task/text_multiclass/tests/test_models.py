"""Contrat du classifieur : ce qu'un modèle de texte promet, indépendamment de la pile.

Ces tests s'exécutent contre la pile active (celle du projet) et n'importent donc que la fabrique
``src.models.build_model`` : ils doivent passer telle quels avec un modèle lexical comme avec un
modèle à base de transformeurs. Ce qui est vérifié ici est le contrat — la forme des sorties, les
gardes — et non les performances, qui appartiennent au pipeline d'évaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models import load_model
from src.models.contract import BaseTextClassifier


def test_a_fresh_model_is_not_fitted(fitted_model: BaseTextClassifier) -> None:
    """Un modèle neuf refuse de prédire : mieux vaut une erreur qu'une décision au hasard."""
    from src.models import build_model

    fresh = build_model({"model": {"algorithm": fitted_model.algorithm}})

    assert not fresh.is_fitted
    assert fresh.state == "unfitted"
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.predict(["Bonjour, ma facture est erronée."])


def test_the_contract_publishes_the_labels_in_a_stable_order(
    fitted_model: BaseTextClassifier, predictions_frame: pd.DataFrame
) -> None:
    """Les colonnes de probabilité suivent l'ordre de ``labels``, trié une fois pour toutes."""
    labels = fitted_model.labels

    assert labels == sorted(labels)
    for label in labels:
        assert f"p_{label}" in predictions_frame.columns


def test_predict_frame_returns_one_row_per_document(
    fitted_model: BaseTextClassifier, test_split: pd.DataFrame
) -> None:
    """``predict_frame`` rend une table alignée sur l'entrée, prête à être jointe."""
    frame = fitted_model.predict_frame(test_split)

    assert len(frame) == len(test_split)
    assert {"prediction", "confidence"} <= set(frame.columns)
    assert frame["prediction"].isin(fitted_model.labels).all()
    assert frame["confidence"].between(0.0, 1.0).all()


def test_predict_frame_refuses_a_missing_text_column(
    fitted_model: BaseTextClassifier, test_split: pd.DataFrame
) -> None:
    """Une frame sans la bonne colonne est refusée, avec la liste des colonnes reçues."""
    with pytest.raises(ValueError, match="not found in the frame"):
        fitted_model.predict_frame(test_split.rename(columns={"text": "body"}))


def test_the_explanation_is_bounded_and_named(
    fitted_model: BaseTextClassifier, test_split: pd.DataFrame
) -> None:
    """L'explication rend au plus ``k`` termes, triés par influence décroissante."""
    texts = test_split["text"].astype(str).head(3).tolist()

    explanations = fitted_model.explain(texts, k=4)

    assert len(explanations) == len(texts)
    for terms in explanations:
        assert len(terms) <= 4
        weights = [abs(weight) for _, weight in terms]
        assert weights == sorted(weights, reverse=True)


def test_the_fit_result_is_archived_in_the_model(
    fitted_model: BaseTextClassifier, train_split: pd.DataFrame
) -> None:
    """Le compte rendu d'ajustement reste attaché au modèle : c'est ce que lit la fiche."""
    result = fitted_model.fit_result_

    assert result is not None
    assert result.n_samples == len(train_split)
    assert result.n_features == fitted_model.n_features > 0
    assert result.duration_seconds >= 0.0
    assert result.extra["n_classes"] == len(fitted_model.labels)


def test_a_single_class_corpus_is_refused(fitted_model: BaseTextClassifier) -> None:
    """Un corpus mono-classe ne produit pas un classifieur : le contrat le refuse à l'entrée."""
    frame = pd.DataFrame({"text": ["un ticket" * 5] * 4, "label": ["autre"] * 4})

    with pytest.raises(ValueError, match="single class"):
        fitted_model.fit(frame)


def test_an_empty_corpus_is_refused(fitted_model: BaseTextClassifier) -> None:
    """Ajuster sur rien produirait un modèle vide et des métriques inventées."""
    with pytest.raises(ValueError, match="empty frame"):
        fitted_model.fit(pd.DataFrame({"text": [], "label": []}))


def test_the_model_card_documents_the_run(
    fitted_model: BaseTextClassifier, train_split: pd.DataFrame
) -> None:
    """La fiche de modèle est l'objet que relit un auditeur : elle doit être complète."""
    card = fitted_model.model_card(
        metrics={"val_macro_f1": 0.5}, artifact="classifier.joblib", notes=["corpus synthétique"]
    )

    assert card.model_name == type(fitted_model).__name__
    assert card.framework == fitted_model.framework
    assert card.algorithm == fitted_model.algorithm
    assert card.task == "multiclass"
    assert card.target_name == "label"
    assert card.feature_names == [fitted_model.text_column]
    assert card.n_samples == train_split.shape[0]
    assert card.n_features == fitted_model.n_features
    assert card.notes == ["corpus synthétique"]
    assert card.to_dict()["metrics"] == {"val_macro_f1": 0.5}


def test_the_artefact_is_written_in_the_directory_it_receives(
    fitted_model: BaseTextClassifier, tmp_path: Path
) -> None:
    """Un dossier reçoit le nom de fichier par défaut : le pipeline n'a pas à le composer."""
    path = fitted_model.save(tmp_path)

    assert path.name == "classifier.joblib"
    assert path.exists()


def test_the_artefact_reloaded_by_the_factory_is_usable(
    fitted_model: BaseTextClassifier, tmp_path: Path, test_split: pd.DataFrame
) -> None:
    """``load_model`` rend un objet qui prédit, explique et se décrit comme l'original."""
    path = fitted_model.save(tmp_path / "classifier.joblib")

    reloaded = load_model(path)

    assert reloaded.is_fitted
    assert reloaded.labels == fitted_model.labels
    assert reloaded.name == fitted_model.name
    texts = test_split["text"].astype(str).head(3).tolist()
    assert reloaded.predict(texts) == fitted_model.predict(texts)
    assert reloaded.explain(texts, k=3) == fitted_model.explain(texts, k=3)
    assert "classes=" in reloaded.summary()


def test_the_factory_refuses_a_missing_artefact(tmp_path: Path) -> None:
    """Recharger un artefact absent est une erreur explicite : pas de modèle vide silencieux."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "classifier.joblib")


def test_the_model_is_a_contract_implementation(fitted_model: BaseTextClassifier) -> None:
    """La pile respecte le contrat : c'est ce qui permet de comparer deux piles."""
    assert isinstance(fitted_model, BaseTextClassifier)
    assert fitted_model.task in {"multiclass", "binary"}
    assert fitted_model.algorithm
