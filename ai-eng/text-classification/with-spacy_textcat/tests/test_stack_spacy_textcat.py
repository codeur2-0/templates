"""Tests de la stack `spacy_textcat` : ce que la stack promet, et seulement cela.

Quatre familles de tests, dans l'ordre où un défaut coûte cher en production :

1. **la décision** — le pipeline vierge apprend le libellé de ses documents d'entraînement, et un
   texte fait de mots inconnus ne fait pas planter l'inférence ;
2. **les probabilités** — une ligne par texte, une colonne par classe, des sommes à 1 : la fiche
   de modèle et la mesure de calibration reposent sur cette forme ;
3. **la persistance** — un artefact rechargé (configuration du pipeline + poids) classe *exactement*
   comme le modèle ajusté, explication comprise ;
4. **le déterminisme** — deux ajustements à graine fixée donnent les mêmes probabilités, ce qui est
   la condition pour que les chiffres du README soient reproductibles.

Le budget d'époques est réduit (`EPOCHS`) : ces tests vérifient des propriétés, pas une performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.models import available_algorithms, build_model, load_model
from src.models.classifier import SpacyTextClassifier
from src.models.contract import BaseTextClassifier

#: Époques des tests : assez pour apprendre le train, assez peu pour rester rapide.
EPOCHS = 6


def _train_split(documents: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of the training split."""
    return documents[documents["split"].astype(str) == "train"].reset_index(drop=True)


def _model(algorithm: str = "textcat_bow", **params: Any) -> SpacyTextClassifier:
    """Build a classifier with the reduced test budget."""
    config = {"model": {"algorithm": algorithm}, "train": {"epochs": EPOCHS}, "seed": 7}
    return cast(SpacyTextClassifier, build_model(config, params=params or None))


def test_the_stack_declares_its_three_architectures() -> None:
    """Les architectures de la stack viennent du registre, pas d'une liste écrite dans le test."""
    assert {"textcat_bow", "textcat_reduce", "textcat_ensemble"} <= set(available_algorithms())


def test_the_classifier_learns_the_label_of_its_training_documents(
    documents: pd.DataFrame,
) -> None:
    """Le pipeline retrouve la classe de la majorité de ses documents d'entraînement."""
    train = _train_split(documents)
    model = _model()
    model.fit(train)

    predictions = model.predict(train["text"].astype(str).tolist())
    accuracy = float(np.mean(np.asarray(predictions) == train["label"].astype(str).to_numpy()))
    assert accuracy > 0.8, accuracy


def test_probabilities_are_shaped_like_the_label_list(documents: pd.DataFrame) -> None:
    """Une ligne par texte, une colonne par classe, des sommes à 1 : la forme promise."""
    train = _train_split(documents)
    model = _model()
    model.fit(train)
    texts = documents["text"].astype(str).head(3).tolist()

    probabilities = model.predict_proba(texts)

    assert probabilities.shape == (len(texts), len(model.labels))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(texts)), atol=1e-9)
    assert model.labels == sorted(model.labels)


def test_an_unknown_word_does_not_break_the_inference(documents: pd.DataFrame) -> None:
    """Des mots jamais vus produisent une décision, pas une exception."""
    model = _model()
    model.fit(_train_split(documents))

    probabilities = model.predict_proba(["bloubiboulga zorglub xyzzy"])

    assert probabilities.shape == (1, len(model.labels))
    assert np.isfinite(probabilities).all()


def test_the_explanation_names_terms_of_the_text(documents: pd.DataFrame) -> None:
    """L'explication par occlusion ne cite que des termes réellement présents dans le texte."""
    train = _train_split(documents)
    model = _model()
    model.fit(train)
    text = str(train["text"].iloc[0])

    explanation = model.explain([text], k=5)[0]

    assert explanation, "l'occlusion doit trouver au moins un terme influent"
    for term, weight in explanation:
        assert term in text, (term, text)
        assert isinstance(weight, float)


def test_the_artefact_classes_like_the_fitted_model(
    documents: pd.DataFrame, tmp_path: Path
) -> None:
    """Un artefact rechargé (configuration + poids) classe, score et explique comme le modèle."""
    model = _model()
    model.fit(_train_split(documents))
    texts = documents["text"].astype(str).head(4).tolist()
    expected = model.predict(texts)
    expected_probabilities = model.predict_proba(texts)
    expected_explanation = model.explain(texts, k=3)

    path = model.save(tmp_path / "classifier.joblib")
    reloaded = cast(SpacyTextClassifier, load_model(path))

    assert reloaded.is_fitted
    assert reloaded.labels == model.labels
    assert reloaded.n_features == model.n_features > 0
    assert reloaded.predict(texts) == expected
    np.testing.assert_allclose(reloaded.predict_proba(texts), expected_probabilities)
    assert reloaded.explain(texts, k=3) == expected_explanation


def test_two_fits_at_the_same_seed_agree(documents: pd.DataFrame) -> None:
    """Deux ajustements à graine fixée donnent les mêmes probabilités (reproductibilité)."""
    train = _train_split(documents)
    texts = documents["text"].astype(str).head(5).tolist()
    first = _model()
    second = _model()

    first.fit(train)
    second.fit(train)

    np.testing.assert_allclose(first.predict_proba(texts), second.predict_proba(texts), atol=1e-6)


@pytest.mark.parametrize("algorithm", ["textcat_reduce", "textcat_ensemble"])
def test_every_architecture_trains_and_scores(documents: pd.DataFrame, algorithm: str) -> None:
    """Les architectures à encodeur s'entraînent et rendent des probabilités bien formées."""
    model = _model(algorithm)
    model.fit(_train_split(documents))

    probabilities = model.predict_proba(documents["text"].astype(str).head(2).tolist())

    assert probabilities.shape == (2, len(model.labels))
    assert model.n_features > 0


def test_a_grid_override_keeps_the_other_defaults() -> None:
    """Surcharger le seul `dropout` ne fait pas perdre le reste du bloc `training`."""
    model = _model("textcat_reduce", training={"dropout": 0.4})

    assert model.params["training"]["dropout"] == 0.4
    assert model.params["training"]["learning_rate"] == 0.001
    assert model.params["architecture"] == "reduce"


def test_registry_defaults_do_not_leak_between_architectures() -> None:
    """`params={}` donne les défauts du registre, jamais la configuration du modèle servi."""
    config = {
        "model": {"algorithm": "textcat_bow", "params": {"training": {"learning_rate": 0.5}}},
        "train": {"epochs": EPOCHS},
    }
    ensemble = cast(
        SpacyTextClassifier, build_model(config, algorithm="textcat_ensemble", params={})
    )

    assert ensemble.params["training"]["learning_rate"] == 0.003
    assert ensemble.params["architecture"] == "ensemble"


def test_an_empty_ticket_gets_a_decision(documents: pd.DataFrame) -> None:
    """Un ticket vide reçoit une distribution valide au lieu de faire planter l'inférence."""
    model = _model()
    model.fit(_train_split(documents))

    probabilities = model.predict_proba(["", "facture en double", "   "])

    assert probabilities.shape == (3, len(model.labels))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3), atol=1e-9)


def test_occlusion_removes_a_token_not_a_substring(documents: pd.DataFrame) -> None:
    """Retirer « mon » ne doit pas abîmer « montant » : seul le jeton visé disparaît."""
    model = _model()
    model.fit(_train_split(documents))
    text = "Le montant de mon remboursement"

    variant = model._without_token(text, 3)

    assert variant.split() == ["Le", "montant", "de", "remboursement"]


def test_an_unknown_algorithm_is_refused() -> None:
    """Une stack refuse un algorithme qu'elle ne sert pas, avec un message qui liste les autres."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_model({"model": {"algorithm": "tfidf_logreg"}})


def test_the_contract_is_the_documented_one(documents: pd.DataFrame) -> None:
    """Le modèle est bien un `BaseTextClassifier`, et son résumé dit ce qu'il utilise."""
    model = _model()
    model.fit(_train_split(documents))

    assert isinstance(model, BaseTextClassifier)
    assert "textcat_bow" in model.summary()
    assert len(model.labels) == documents["label"].nunique()
