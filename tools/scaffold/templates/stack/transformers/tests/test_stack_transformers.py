"""Tests de la stack `transformers` : ce que la stack promet, et seulement cela.

Cinq familles de tests, dans l'ordre où un défaut coûte cher en production :

1. **la décision** — l'encodeur apprend le libellé de ses documents d'entraînement, et un texte
   fait de mots inconnus ne fait pas planter l'inférence (il produit une décision, et le taux de
   jetons ``[UNK]`` est publié pour dire à quel point cette décision est fragile) ;
2. **le vocabulaire** — il est appris sur le **train** : un terme absent du train devient ``[UNK]``,
   et le pré-entraînement masqué ne voit aucun libellé ;
3. **les probabilités** — une ligne par texte, une colonne par classe, des sommes à 1 : la fiche de
   modèle et la mesure de calibration reposent sur cette forme ;
4. **la persistance** — un artefact rechargé prédit et explique *exactement* comme le modèle ajusté,
   sans quoi un `load` silencieux rendrait un encodeur aux poids aléatoires ;
5. **le déterminisme** — deux ajustements à graine fixée donnent les mêmes probabilités, ce qui est
   la condition pour que les chiffres du README soient reproductibles.

Les architectures des tests sont volontairement minuscules (vocabulaire de 400 pièces, deux
couches de 64 unités, une époque) : la suite doit rester exécutable à chaque commit, et ce n'est pas
elle qui mesure la qualité — c'est le pipeline d'évaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.models import available_algorithms, build_model, load_model
from src.models.classifier import TransformerTextClassifier
from src.models.contract import BaseTextClassifier

#: Paramètres d'un encodeur de test : minuscules, mais structurellement identiques à la production.
_TINY_PARAMS: dict[str, Any] = {
    "tokenizer": {"vocab_size": 400, "min_frequency": 2, "max_length": 32},
    "network": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "dropout": 0.1,
    },
    "pretraining": {"epochs": 1, "mask_probability": 0.15},
    "training": {"class_weight": "balanced", "threads": 1},
}

#: Paramètres du sac d'embeddings (aucun pré-entraînement, aucun contexte).
_BAG_PARAMS: dict[str, Any] = {
    "tokenizer": {"vocab_size": 400, "min_frequency": 2, "max_length": 32},
    "network": {"embedding_dim": 32, "dropout": 0.0},
    "pretraining": {"epochs": 0},
    "training": {"class_weight": "balanced", "threads": 1},
}

#: Configuration d'entraînement des tests : une époque, petits lots.
_TRAIN_NODE: dict[str, Any] = {"epochs": 1, "batch_size": 16, "learning_rate": 0.01}


def _config(algorithm: str) -> dict[str, Any]:
    """Return the minimal configuration understood by the factory."""
    return {
        "model": {"algorithm": algorithm, "text_column": "text", "target": "label"},
        "train": dict(_TRAIN_NODE),
    }


def _train_split(documents: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of the training split."""
    return documents[documents["split"].astype(str) == "train"].reset_index(drop=True)


def _fit(
    documents: pd.DataFrame,
    algorithm: str = "bert_tiny",
    *,
    params: dict[str, Any] | None = None,
    epochs: int = 1,
) -> TransformerTextClassifier:
    """Fit a tiny model of the stack on the training split."""
    config = _config(algorithm)
    config["train"] = {**_TRAIN_NODE, "epochs": epochs}
    model = cast(
        TransformerTextClassifier,
        build_model(config, algorithm=algorithm, params=params or dict(_TINY_PARAMS)),
    )
    model.fit(_train_split(documents))
    return model


def test_the_stack_declares_its_three_architectures() -> None:
    """Les architectures de la stack viennent du registre, pas d'une liste écrite dans le test."""
    assert {"embedding_bag", "bert_tiny", "bert_small"} <= set(available_algorithms())


def test_the_tokenizer_learns_its_vocabulary_on_the_training_split(
    documents: pd.DataFrame,
) -> None:
    """Le vocabulaire est appris sur le train : il ne connaît que son alphabet et ses pièces.

    WordPiece découpe un mot inconnu en **sous-mots** connus (``bloubiboulga`` devient ``b``,
    ``##l``, ``##ou``…), et ne produit ``[UNK]`` que lorsqu'aucune pièce ne correspond — un autre
    alphabet, par exemple. C'est le comportement attendu d'un vocabulaire appris, pas un défaut :
    le taux de ``[UNK]`` est publié dans la fiche de modèle.
    """
    model = _fit(documents)
    train = _train_split(documents)

    vocabulary = model.tokenizer.get_vocab()
    word = str(train["text"].iloc[0]).split()[0]

    assert 0 < model.n_features <= 400
    assert model.tokenizer.unk_token in vocabulary
    # Un texte qui n'utilise aucun caractère du corpus tombe en [UNK]…
    assert "[UNK]" in model.tokenizer.tokenize("Спасибо за помощь")
    # …alors qu'un mot du train est découpé en pièces connues du vocabulaire.
    assert "[UNK]" not in model.tokenizer.tokenize(word)


def test_the_encoder_learns_the_label_of_its_training_documents(
    documents: pd.DataFrame,
) -> None:
    """Un texte déjà vu est classé dans sa classe : les poids ont bien été mis à jour."""
    model = _fit(documents, epochs=3)
    train = _train_split(documents)
    sample = train.head(5)

    predictions = model.predict(sample["text"].astype(str).tolist())

    assert predictions == sample["label"].astype(str).tolist()


def test_probabilities_are_shaped_like_the_label_list(documents: pd.DataFrame) -> None:
    """Une ligne par texte, une colonne par classe, des sommes à 1 : la forme promise."""
    model = _fit(documents)
    texts = documents["text"].astype(str).head(3).tolist()

    probabilities = model.predict_proba(texts)

    assert probabilities.shape == (len(texts), len(model.labels))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(texts)), atol=1e-6)
    assert model.labels == sorted(model.labels)


def test_an_unknown_word_does_not_break_the_inference(documents: pd.DataFrame) -> None:
    """Un terme jamais vu n'a pas d'embedding appris : il doit être compté, pas lever."""
    model = _fit(documents)

    probabilities = model.predict_proba(["[UNK] ".join(["bloubiboulga"] * 8)])

    assert probabilities.shape == (1, len(model.labels))
    extra = model._extra_metadata()
    assert 0.0 <= extra["unk_token_rate"] <= 1.0
    assert 0.0 <= extra["truncated_rate"] <= 1.0


def test_the_explanation_names_terms_of_the_text(documents: pd.DataFrame) -> None:
    """L'explication par occlusion ne cite que des termes réellement présents dans le texte."""
    model = _fit(documents)
    text = str(_train_split(documents)["text"].iloc[0])

    explanation = model.explain([text], k=4)[0]

    assert explanation, "même un encodeur contextuel doit pouvoir expliquer sa décision"
    for term, weight in explanation:
        assert term in text, (term, text)
        assert term.lower() != "[unk]", term
        assert isinstance(weight, float)


def test_the_explanation_is_ordered_by_absolute_influence(documents: pd.DataFrame) -> None:
    """Les termes sont triés par influence décroissante : c'est le contrat de la famille."""
    model = _fit(documents)
    texts = documents["text"].astype(str).head(2).tolist()

    for terms in model.explain(texts, k=5):
        weights = [abs(weight) for _, weight in terms]
        assert weights == sorted(weights, reverse=True)


def test_the_artefact_classes_like_the_fitted_model(
    documents: pd.DataFrame, tmp_path: Path
) -> None:
    """Un artefact rechargé prédit, score et explique comme le modèle ajusté."""
    model = _fit(documents)
    texts = documents["text"].astype(str).head(4).tolist()
    expected = model.predict(texts)
    expected_probabilities = model.predict_proba(texts)
    expected_explanation = model.explain(texts, k=3)

    path = model.save(tmp_path / "model.pt")
    reloaded = cast(TransformerTextClassifier, load_model(path))

    assert path.suffix == ".pt"
    assert reloaded.is_fitted
    assert reloaded.labels == model.labels
    assert reloaded.n_features == model.n_features
    assert reloaded.n_parameters == model.n_parameters
    assert reloaded.predict(texts) == expected
    np.testing.assert_allclose(reloaded.predict_proba(texts), expected_probabilities, atol=1e-12)
    assert reloaded.explain(texts, k=3) == expected_explanation


def test_a_directory_receives_the_default_artefact_name(
    documents: pd.DataFrame, tmp_path: Path
) -> None:
    """Un dossier reçoit ``model.pt`` : le nom vient de la stack, jamais du pipeline."""
    model = _fit(documents)

    path = model.save(tmp_path)

    assert path.name == "model.pt"
    assert load_model(tmp_path).labels == model.labels


def test_two_fits_at_the_same_seed_agree(documents: pd.DataFrame) -> None:
    """Deux ajustements à graine fixée donnent les mêmes probabilités (reproductibilité)."""
    texts = documents["text"].astype(str).head(5).tolist()
    first = _fit(documents)
    second = _fit(documents)

    np.testing.assert_allclose(
        first.predict_proba(texts), second.predict_proba(texts), atol=1e-12
    )


def test_the_bag_of_embeddings_needs_no_pre_training(documents: pd.DataFrame) -> None:
    """Le sac d'embeddings est le plancher neuronal : aucun masquage, aucun contexte."""
    model = _fit(documents, algorithm="embedding_bag", params=dict(_BAG_PARAMS))

    assert model.n_parameters > 0
    assert model._extra_metadata()["is_transparent"] == 0.0
    assert model._extra_metadata()["n_classes"] == float(len(model.labels))


def test_the_pre_training_can_be_disabled(documents: pd.DataFrame) -> None:
    """``pretraining.epochs: 0`` est un réglage documenté : l'affinage seul doit rester jouable."""
    params = dict(_TINY_PARAMS)
    params["pretraining"] = {"epochs": 0, "mask_probability": 0.15}
    model = _fit(documents, params=params)

    probabilities = model.predict_proba(documents["text"].astype(str).head(2).tolist())

    assert model.is_fitted
    assert model.n_parameters > 0
    assert probabilities.shape == (2, len(model.labels))


def test_an_unknown_algorithm_is_refused() -> None:
    """Une stack refuse un algorithme qu'elle ne sert pas, avec un message qui liste les autres."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_model({"model": {"algorithm": "tfidf_logreg"}})


def test_an_unhandled_artefact_extension_is_refused(tmp_path: Path) -> None:
    """Un fichier étranger est refusé par son extension, avant toute désérialisation."""
    stranger = tmp_path / "model.txt"
    stranger.write_text("not a model", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported model artefact"):
        load_model(stranger)


def test_a_missing_artefact_is_refused(tmp_path: Path) -> None:
    """Recharger un artefact absent est une erreur explicite : pas de modèle vide silencieux."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "model.pt")


def test_the_contract_is_the_documented_one(documents: pd.DataFrame) -> None:
    """Le modèle est bien un `BaseTextClassifier`, et son résumé dit ce qu'il utilise."""
    model = _fit(documents)

    assert isinstance(model, BaseTextClassifier)
    assert "bert_tiny" in model.summary()
    assert len(model.labels) == documents["label"].nunique()
    assert model.state == "fitted"
