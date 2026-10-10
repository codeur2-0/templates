"""Fabrique, contrat et cycle de vie d'un extracteur : les trois algorithmes de la stack.

Trois choses se testent ici, et aucune ne se voit dans un score :

* la **fabrique** — l'algorithme demandé est celui qui est construit, les hyper-paramètres du
  manifeste arrivent bien au modèle, et la liste des algorithmes est cohérente avec ce que la
  fabrique sait construire ;
* les **invariants des mentions** — triées par position, sans chevauchement, avec
  ``surface == text[start:end]``, une provenance dans ``{regle, modele}`` et une confiance bornée.
  C'est le contrat que consomme le back-office ;
* la **persistance** — un artefact rechargé prédit **exactement** comme le modèle ajusté. Un écart
  d'un seul span sur une seule mention signalerait une sérialisation incomplète, et le test compare
  les prédictions span par span, pas une métrique moyenne.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import (
    ALGORITHMS,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
)
from src.models.contract import SOURCES, sort_mentions

ALGORITHMS_UNDER_TEST = ("gazetteer", "tagger", "hybrid")


def test_the_factory_declares_the_algorithms_it_can_build(app_config) -> None:
    """Les algorithmes annoncés sont ceux que la fabrique sait réellement construire."""
    declared = set(available_algorithms(app_config.metrics.task))
    assert set(ALGORITHMS) <= declared
    for name in ALGORITHMS_UNDER_TEST:
        described = describe_algorithm(name)
        assert described["name"] == name
        assert described["display_name"]
        assert described["rationale"]
        assert "learns_weights" in described


def test_the_configured_algorithm_is_the_one_built(app_config) -> None:
    """Le modèle construit porte l'algorithme du manifeste, pas un défaut implicite."""
    model = build_model(app_config.model_dump())

    assert model.algorithm == app_config.model.algorithm
    assert model.task == app_config.metrics.task
    # Les types sont **appris** du corpus : un extracteur qui n'a pas encore vu le train n'en
    # déclare aucun. Le manifeste dit quel algorithme, pas ce que le corpus contient.
    assert model.labels == []


def test_an_unknown_algorithm_is_refused(app_config) -> None:
    """Un algorithme inconnu lève, au lieu de retomber silencieusement sur un défaut."""
    with pytest.raises(ValueError):
        build_model(app_config.model_dump(), algorithm="mystere")


@pytest.mark.parametrize("algorithm", ALGORITHMS_UNDER_TEST)
def test_every_algorithm_respects_the_mention_contract(
    app_config, train_split, test_split, algorithm: str
) -> None:
    """Quel que soit l'algorithme, la sortie est triée, sans chevauchement et ancrée."""
    train_documents, train_spans = train_split
    test_documents, _ = test_split
    model = build_model(app_config.model_dump(), algorithm=algorithm, params={})
    model.fit(train_documents, train_spans)

    mentions = model.predict(
        [str(text) for text in test_documents["text"]], ids=list(test_documents["msg_id"])
    )

    assert len(mentions) == len(test_documents)
    assert set(model.labels) == set(train_spans["label"].astype(str)), (
        "les types servis sont ceux du train, ni plus ni moins"
    )
    for text, found in zip(test_documents["text"], mentions, strict=True):
        assert found == sort_mentions(found)
        for mention in found:
            assert 0 <= mention.start < mention.end <= len(str(text))
            assert mention.surface == str(text)[mention.start : mention.end]
            assert mention.source in SOURCES
            assert 0.0 <= mention.confidence <= 1.0
            assert mention.label in set(model.labels)
        bounds = [(mention.start, mention.end) for mention in found]
        assert all(bounds[index][1] <= bounds[index + 1][0] for index in range(len(bounds) - 1)), (
            "deux mentions ne se recouvrent jamais"
        )


def test_the_rule_layer_knows_only_what_the_training_split_taught_it(
    app_config, train_split, test_split
) -> None:
    """Les surfaces de l'index viennent du train : une surface réservée est invisible."""
    train_documents, train_spans = train_split
    test_documents, test_spans = test_split
    model = build_model(app_config.model_dump(), algorithm="gazetteer", params={})
    model.fit(train_documents, train_spans)

    train_surfaces = {
        (str(row.label), str(row.surface).casefold()) for row in train_spans.itertuples(index=False)
    }
    reserved = test_spans[test_spans["holdout"].astype(bool)]
    assert not reserved.empty
    predictions = model.predict(
        [str(text) for text in test_documents["text"]], ids=list(test_documents["msg_id"])
    )
    predicted = {
        (mention.label, mention.surface.casefold()) for found in predictions for mention in found
    }

    leaked = {
        (str(row.label), str(row.surface).casefold()) for row in reserved.itertuples(index=False)
    } & train_surfaces
    assert leaked == set()
    # Aucune mention dont la surface n'était pas dans le train n'est attribuée à un type « nom ».
    invented = {
        (label, surface)
        for label, surface in predicted
        if label in {"produit", "transporteur"} and (label, surface) not in train_surfaces
    }
    assert invented == set()


def test_a_saved_artefact_predicts_exactly_like_the_fitted_model(
    fitted_model, app_config, test_split, tmp_path
) -> None:
    """L'artefact rechargé rend les mêmes mentions, span par span."""
    test_documents, _ = test_split
    texts = [str(text) for text in test_documents["text"]]
    identifiers = list(test_documents["msg_id"])
    before = fitted_model.predict(texts, ids=identifiers)

    directory = fitted_model.save(tmp_path / str(app_config.train.artifacts.model_file))
    reloaded = load_model(directory, config=app_config.model_dump())
    after = reloaded.predict(texts, ids=identifiers)

    assert directory.is_dir(), "l'artefact d'un pipeline spaCy est un répertoire"
    assert reloaded.algorithm == fitted_model.algorithm
    assert reloaded.labels == fitted_model.labels
    for reference, candidate in zip(before, after, strict=True):
        assert [mention.as_tuple() for mention in reference] == [
            mention.as_tuple() for mention in candidate
        ]
        assert [mention.source for mention in reference] == [
            mention.source for mention in candidate
        ]


def test_the_model_card_publishes_the_recipe_of_the_run(fitted_model, training_outcome) -> None:
    """La fiche de modèle porte la version des bibliothèques, les métriques et l'architecture."""
    card = fitted_model.model_card(
        metrics=training_outcome.metrics, artifact="tagger", notes=["test"]
    )
    payload = card.to_dict()

    assert payload["algorithm"] == fitted_model.algorithm
    assert payload["framework"] == "spacy"
    assert payload["artifact"] == "tagger"
    assert payload["library_versions"].get("spacy")
    assert payload["params"]["resolved_architecture"]["pipe_names"]
    assert payload["params"]["training"]["epochs"] >= 1
    assert payload["notes"] == ["test"]


def test_the_architecture_is_published_instead_of_being_redeclared(fitted_model) -> None:
    """L'architecture résolue est **publiée** : une config partielle décrirait un autre modèle."""
    params = fitted_model._effective_params()

    assert params["resolved_architecture"]["pipe_names"]
    assert "ner" in fitted_model.architecture_report().get("pipe_names", [])


def test_predicting_a_frame_missing_its_text_column_is_refused(fitted_model) -> None:
    """Une colonne de texte absente est signalée avec les colonnes présentes."""
    frame = pd.DataFrame({"msg_id": ["MSG-0001"]})

    with pytest.raises(ValueError):
        fitted_model.predict_frame(frame)
