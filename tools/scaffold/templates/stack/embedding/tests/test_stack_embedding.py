"""Tests propres à la stack **index vectoriel dense (hachage + SVD)**.

Ce module vit dans la couche `stack/embedding` : il a le droit de connaître les noms
d'algorithmes et les objets internes de la stack. Les tests partagés de la modalité texte
(`modality/text/tests`) ne les connaissent pas, ce qui permet à une seconde stack de servir la même
famille sans les réécrire.

Ce qui est vérifié ici n'est pas le contrat (la modalité le teste déjà) mais ce qui **distingue**
cette stack : les deux formes d'index qu'elle sert, la dimension réellement stockée, la fidélité de
la projection, la persistance des vecteurs, la détection de quasi-doublons et le cache de requêtes.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.models import (
    EmbeddingModel,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
)


def _fact_and_reference(title: str) -> tuple[str, str]:
    """Split a fiche title into its fact and its reference.

    Argument:
        title: Title of a fiche, as the generator writes it
            (``"Notice — autonomie — casque Aria (ARIA)"``).

    Returns:
        The fact label and the reference code.
    """
    parts = [part.strip() for part in title.split(" — ")]
    return parts[1], parts[2].rsplit("(", 1)[1].rstrip(")")


def test_the_dense_stack_declares_its_two_index_shapes() -> None:
    """Les deux formes d'index viennent du registre de la stack, pas d'une liste écrite ici."""
    assert {"hashing_svd", "hashing_raw"} <= set(available_algorithms())


def test_the_published_dimension_is_the_stored_one(documents: pd.DataFrame) -> None:
    """La dimension publiée est celle de la matrice, pas celle demandée dans la configuration.

    Le paramètre ``n_features`` est celui de l'espace de hachage ; en revanche une SVD tronquée
    réduit ses composantes quand le corpus est plus petit que la dimension visée : un index dont la
    fiche de modèle annonce 128 dimensions alors qu'il en stocke 59 documente une intention, pas un
    fait.
    """
    raw = cast(
        EmbeddingModel,
        build_model(
            {
                "model": {
                    "algorithm": "hashing_raw",
                    "params": {"embedding": {"n_features": 2048, "n_components": 0}},
                }
            }
        ),
    )
    raw.fit(documents)
    compressed = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_svd"}}))
    compressed.fit(documents)

    summary = compressed.index_summary()
    requested = int(describe_algorithm("hashing_svd")["n_components"])
    assert raw.dimension == 2048
    assert summary["fitted"] is True
    assert summary["n_passages"] == len(compressed.chunks)
    assert compressed.dimension == min(requested, summary["n_passages"] - 1)


def test_the_projection_fidelity_is_measured(documents: pd.DataFrame) -> None:
    """La fidélité vaut 1,0 sans projection et reste mesurée avec.

    L'espace de hachage brut est exact par construction ; dès qu'une SVD compresse, la fidélité
    compare l'ordre des similarités des deux espaces et devient le chiffre qui dit ce que la
    compression conserve — celui qu'un index doit publier au lieu de supposer.
    """
    projected = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_svd"}}))
    projected.fit(documents)
    exact = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_raw"}}))
    exact.fit(documents)

    assert exact.projection_fidelity == 1.0
    assert 0.0 <= projected.projection_fidelity <= 1.0
    assert projected.index_summary()["projection_fidelity"] == pytest.approx(
        projected.projection_fidelity, abs=1e-6
    )


def test_the_index_survives_a_reload(documents: pd.DataFrame, tmp_path: Path) -> None:
    """Les vecteurs sont sérialisés : un index rechargé classe comme l'index ajusté.

    C'est le test qui manquait quand le rechargement reconstruisait un embedder vide : le premier
    ``transform`` levait alors un ``AttributeError`` en production, après un ``load`` parfaitement
    silencieux.
    """
    model = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_svd"}}))
    model.fit(documents)
    question = str(documents["text"].iloc[0])
    expected = [passage.chunk_id for passage in model.retrieve(question, 5)]

    path = model.save(tmp_path / "vector_index.joblib")
    reloaded = cast(EmbeddingModel, load_model(path))

    assert reloaded.is_fitted
    assert reloaded.dimension == model.dimension
    assert [passage.chunk_id for passage in reloaded.retrieve(question, 5)] == expected
    np.testing.assert_allclose(
        [passage.score for passage in reloaded.retrieve(question, 5)],
        [passage.score for passage in model.retrieve(question, 5)],
    )
    np.testing.assert_allclose(reloaded.embed([question]), model.embed([question]))


def test_nearest_neighbours_returns_the_other_fiches_of_the_fact(
    documents: pd.DataFrame,
) -> None:
    """Les plus proches voisins d'une fiche sont les autres écritures du même fait.

    C'est le second usage de l'index : retrouver les trois versions d'une même information (notice,
    fiche commerciale, note SAV) sans écrire de recherche par mots-clés. Le test exclut la fiche
    interrogée, comme le fait un vrai travail de dédoublonnage, et il le fait pour **toutes** les
    fiches du corpus de test : un échantillon ne dirait rien de la propriété annoncée au README
    (les deux plus proches voisins sont les deux autres écritures), et le corpus de test reste
    petit — la mesure est gratuite.
    """
    model = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_svd"}}))
    model.fit(documents)
    titles = dict(zip(documents["doc_id"], documents["title"], strict=True))

    checked = 0
    for record in documents.to_dict(orient="records"):
        neighbours = model.nearest_neighbours(
            str(record["text"]), 2, exclude_doc_id=str(record["doc_id"])
        )
        assert len(neighbours) == 2
        fact, reference = _fact_and_reference(str(record["title"]))
        for neighbour in neighbours:
            assert neighbour.doc_id != record["doc_id"]
            assert _fact_and_reference(str(titles[neighbour.doc_id])) == (fact, reference)
        checked += len(neighbours)
    assert checked == 2 * len(documents)


def test_the_query_cache_serves_the_second_lookup(documents: pd.DataFrame) -> None:
    """Le cache de vecteurs de requêtes est mesuré, jamais supposé.

    Le débit d'un index de recherche dépend de la charge : servir deux fois la même requête doit
    coûter une seule vectorisation, et l'artefact publie le taux de succès correspondant.
    """
    model = cast(EmbeddingModel, build_model({"model": {"algorithm": "hashing_svd"}}))
    model.fit(documents)
    model.clear_cache()
    question = str(documents["text"].iloc[0])

    first = model.scores_for(question)
    second = model.scores_for(question)
    stats = model.query_cache_stats()

    assert stats["misses"] == 1.0
    assert stats["hits"] == 1.0
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["entries"] == 1.0
    np.testing.assert_allclose(first, second)


def test_the_model_card_publishes_the_index_properties(fitted_model: EmbeddingModel) -> None:
    """L'artefact publie l'index qu'il contient : algorithme, dimension, fidélité, taille.

    Un index dont on ne connaît ni la dimension ni la fidélité ne peut être comparé à un autre : la
    carte de modèle est l'endroit où ces propriétés deviennent des faits vérifiables.
    """
    card = fitted_model.model_card().to_dict()
    summary = fitted_model.index_summary()

    assert card["framework"] == "embedding"
    assert card["algorithm"] == fitted_model.algorithm
    assert card["algorithm"] in set(available_algorithms())
    assert summary["fitted"] is True
    assert summary["n_passages"] > 0
    assert summary["dimension"] == card["params"]["embedding"]["dimension"]
    assert card["params"]["embedding"]["projection_fidelity"] == pytest.approx(
        summary["projection_fidelity"], abs=1e-6
    )
    fit_result = fitted_model.fit_result_
    assert fit_result is not None
    assert summary["matrix_bytes"] > 0
    assert fit_result.extra["matrix_bytes"] == summary["matrix_bytes"]
    assert fit_result.extra["matrix_mb"] == pytest.approx(summary["matrix_mb"])
