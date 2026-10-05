"""Tests propres à la stack **scikit-learn (TF-IDF / BM25)**.

Ce module vit dans la couche `stack/tfidf` : il a le droit de connaître les noms d'algorithmes et
les objets internes de la stack. Les tests partagés de la modalité texte (`modality/text/tests`)
ne les connaissent pas, ce qui permet à une seconde stack de servir la même famille sans les
réécrire.

Ce qui est vérifié ici n'est pas le contrat (la modalité le teste déjà) mais ce qui **distingue**
cette stack : les deux scorers qu'elle sert, la persistance du vocabulaire appris, et la
traçabilité de l'algorithme dans la fiche de modèle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import available_algorithms, build_model, load_model
from src.models.base import BaseModel


def test_the_lexical_stack_declares_its_two_scorers() -> None:
    """BM25 et TF-IDF cosinus viennent du registre de la stack, pas d'une liste écrite ici."""
    assert {"bm25", "tfidf_cosine"} <= set(available_algorithms())


def test_the_two_scorers_rank_differently(documents: pd.DataFrame) -> None:
    """Deux scorers distincts classent différemment : sinon l'un des deux ne sert à rien."""
    question = "Quel est le solde annuel de congés payés pour les salariés en CDI ?"
    rankings = {}
    for algorithm in ("bm25", "tfidf_cosine"):
        model = build_model({"model": {"algorithm": algorithm}})
        model.fit(documents)
        rankings[algorithm] = [passage.chunk_id for passage in model.retrieve(question, 10)]
    assert rankings["bm25"] != rankings["tfidf_cosine"]


def test_the_vocabulary_is_persisted_with_the_index(
    documents: pd.DataFrame, tmp_path: Path
) -> None:
    """Le vocabulaire appris est sérialisé : un index rechargé ne réapprend rien.

    C'est le test qui manquait quand le rechargement reconstruisait un vectoriseur vide : le
    premier ``transform`` levait alors un ``AttributeError`` en production, après un ``load``
    parfaitement silencieux.
    """
    model = build_model({"model": {"algorithm": "bm25"}})
    model.fit(documents)
    question = str(documents["text"].iloc[0])[:60]
    expected = [passage.chunk_id for passage in model.retrieve(question, 5)]

    path = model.save(tmp_path / "index.joblib")
    reloaded = load_model(path, config={"model": {"algorithm": "bm25"}})
    assert reloaded.is_fitted
    assert [passage.chunk_id for passage in reloaded.retrieve(question, 5)] == expected
    np.testing.assert_allclose(
        [passage.score for passage in reloaded.retrieve(question, 5)],
        [passage.score for passage in model.retrieve(question, 5)],
    )


def test_the_model_card_names_the_configured_algorithm(fitted_model: BaseModel) -> None:
    """L'artefact dit quel scorer l'a produit : deux algorithmes, deux cartes différentes."""
    card = fitted_model.model_card().to_dict()
    assert card["framework"] == "tfidf"
    assert card["algorithm"] == fitted_model.algorithm
    assert card["algorithm"] in set(available_algorithms())
