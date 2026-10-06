"""Tests de la stack `tfidf_classifier` : ce que la stack promet, et seulement cela.

Quatre familles de tests, dans l'ordre où un défaut coûte cher en production :

1. **la décision** — le classifieur apprend le libellé de ses documents d'entraînement, et un
   document hors vocabulaire ne fait pas planter l'inférence (il produit une décision, et le taux
   de termes hors vocabulaire est publié pour dire à quel point cette décision est fragile) ;
2. **les probabilités** — une ligne par texte, une colonne par classe, des sommes à 1 : la fiche
   de modèle et la mesure de calibration reposent sur cette forme ;
3. **la persistance** — un artefact rechargé classe *exactement* comme le modèle ajusté, y compris
   son explication ; sans ce test, un `load` silencieux peut rendre un classifieur vide ;
4. **le déterminisme** — deux ajustements à graine fixée donnent les mêmes probabilités, ce qui est
   la condition pour que les chiffres du README soient reproductibles.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.models import available_algorithms, build_model, load_model
from src.models.classifier import TfidfClassifier
from src.models.contract import BaseTextClassifier
from src.preprocessing.transformers import tokenize


def _train_split(documents: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of the training split."""
    return documents[documents["split"].astype(str) == "train"].reset_index(drop=True)


def test_the_stack_declares_its_three_classifiers() -> None:
    """Les classifieurs de la stack viennent du registre, pas d'une liste écrite dans le test."""
    assert {"tfidf_logreg", "tfidf_nb", "tfidf_centroid"} <= set(available_algorithms())


def test_the_classifier_learns_the_label_of_its_training_documents(
    documents: pd.DataFrame,
) -> None:
    """Un texte déjà vu est classé dans sa classe : la représentation est bien ajustée."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)

    sample = train.head(5)
    predictions = model.predict(sample["text"].astype(str).tolist())
    assert predictions == sample["label"].astype(str).tolist()


def test_probabilities_are_shaped_like_the_label_list(documents: pd.DataFrame) -> None:
    """Une ligne par texte, une colonne par classe, des sommes à 1 : la forme promise."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)
    texts = documents["text"].astype(str).head(3).tolist()

    probabilities = model.predict_proba(texts)

    assert probabilities.shape == (len(texts), len(model.labels))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(texts)), atol=1e-9)
    assert model.labels == sorted(model.labels)


def test_an_unknown_word_does_not_break_the_inference(documents: pd.DataFrame) -> None:
    """Un terme jamais vu n'a pas de poids : il ne doit pas lever, il doit être compté."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)

    probabilities = model.predict_proba(["bloubiboulga zorglub xyzzy"])

    assert probabilities.shape == (1, len(model.labels))
    extra = model._extra_metadata()
    assert 0.0 <= extra["oov_term_rate"] <= 1.0


def test_the_explanation_names_terms_of_the_text(documents: pd.DataFrame) -> None:
    """L'explication d'un classifieur linéaire nomme des termes réellement présents."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)
    text = str(train["text"].iloc[0])

    explanation = model.explain([text], k=5)[0]

    assert explanation, "un classifieur linéaire doit pouvoir expliquer sa décision"
    tokens = set(tokenize(text))
    for term, weight in explanation:
        # Le vecteur indexe des unigrammes **et** des bigrammes : « ne correspond » n'est donc pas
        # un jeton, mais chacune de ses parties doit se retrouver dans le texte expliqué.
        assert term.split(), term
        assert all(part in tokens for part in term.split()), (term, sorted(tokens))
        assert isinstance(weight, float)


def test_the_artefact_classes_like_the_fitted_model(
    documents: pd.DataFrame, tmp_path: Path
) -> None:
    """Un artefact rechargé classe, score et explique comme le modèle ajusté.

    C'est le test qui manquait à la stack lexicale : un `load` qui reconstruit un vocabulaire vide
    échoue seulement à la première requête, en production.
    """
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)
    texts = documents["text"].astype(str).head(4).tolist()
    expected = model.predict(texts)
    expected_probabilities = model.predict_proba(texts)
    expected_explanation = model.explain(texts, k=3)

    path = model.save(tmp_path / "classifier.joblib")
    reloaded = cast(TfidfClassifier, load_model(path))

    assert reloaded.is_fitted
    assert reloaded.labels == model.labels
    assert reloaded.n_features == model.n_features
    assert reloaded.predict(texts) == expected
    np.testing.assert_allclose(reloaded.predict_proba(texts), expected_probabilities)
    assert reloaded.explain(texts, k=3) == expected_explanation


def test_two_fits_at_the_same_seed_agree(documents: pd.DataFrame) -> None:
    """Deux ajustements à graine fixée donnent les mêmes probabilités (reproductibilité)."""
    train = _train_split(documents)
    texts = documents["text"].astype(str).head(5).tolist()
    first = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    second = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))

    first.fit(train)
    second.fit(train)

    np.testing.assert_allclose(first.predict_proba(texts), second.predict_proba(texts))


def test_the_centroid_classifier_needs_no_learned_weight(documents: pd.DataFrame) -> None:
    """Le centroïde classe sans apprentissage de poids : c'est le plancher de la famille."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_centroid"}}))
    model.fit(train)

    accuracy = float(
        np.mean(
            np.asarray(model.predict(train["text"].astype(str).tolist()))
            == train["label"].astype(str).to_numpy()
        )
    )

    assert accuracy > 0.5, "un centroïde doit au moins retrouver la classe de ses documents"
    assert model._extra_metadata()["is_transparent"] == 1.0


def test_an_unknown_algorithm_is_refused() -> None:
    """Une stack refuse un algorithme qu'elle ne sert pas, avec un message qui liste les autres."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_model({"model": {"algorithm": "transformers_bert"}})


def test_the_contract_is_the_documented_one(documents: pd.DataFrame) -> None:
    """Le modèle est bien un `BaseTextClassifier`, et son résumé dit ce qu'il utilise."""
    train = _train_split(documents)
    model = cast(TfidfClassifier, build_model({"model": {"algorithm": "tfidf_logreg"}}))
    model.fit(train)

    assert isinstance(model, BaseTextClassifier)
    assert "tfidf_logreg" in model.summary()
    assert len(model.labels) == documents["label"].nunique()
