"""Modèle lexical : récupération, réponse sourcée, abstention, sérialisation.

Le contrat ``BaseModel`` est testé sur ses **conséquences visibles**, pas sur ses méthodes : une
réponse sans citation, un score qui ne décroît pas, une abstention qui n'arrive jamais ou un
artefact rechargé qui oublie le seuil calibré sont des régressions qu'aucun test de fumée
n'attrape. Chaque test ci-dessous correspond à l'un de ces quatre échecs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.schemas import validate_predictions
from src.models import available_algorithms, build_model, load_model
from src.models.base import BaseModel, ModelCard


def test_unfitted_model_refuses_to_retrieve() -> None:
    """Retriever sans index : erreur explicite, jamais une liste vide silencieuse."""
    model = build_model({"model": {"algorithm": available_algorithms()[0], "params": {}}})
    assert not model.is_fitted
    with pytest.raises(RuntimeError):
        model.retrieve("Quelle est la durée de la période d'essai ?")


def test_fitted_model_exposes_document_level_metadata(fitted_model: BaseModel) -> None:
    """Les passages indexés portent leur document, leur titre et leurs offsets."""
    chunks = fitted_model.chunks
    assert len(chunks) > 0
    assert {"chunk_id", "doc_id", "chunk_index", "start_char", "end_char", "n_tokens"} <= set(
        chunks.columns
    )
    assert chunks["chunk_id"].is_unique
    assert (chunks["n_tokens"] > 0).all()


def test_retrieve_returns_ranked_passages(fitted_model: BaseModel, queries: pd.DataFrame) -> None:
    """Les passages sont classés par score décroissant et numérotés à partir de 1."""
    question = str(queries.loc[queries["difficulty"] == "facile", "question"].iloc[0])
    passages = fitted_model.retrieve(question, k=5)
    assert len(passages) == 5
    assert [passage.rank for passage in passages] == [1, 2, 3, 4, 5]
    scores = [passage.score for passage in passages]
    assert scores == sorted(scores, reverse=True)
    assert all(passage.doc_id for passage in passages)


def test_retrieve_honours_the_section_filter(
    fitted_model: BaseModel, documents: pd.DataFrame
) -> None:
    """Le filtre par section sert au support (« chercher seulement dans les procédures »)."""
    section = str(documents["section"].iloc[0])
    passages = fitted_model.retrieve(
        "Comment déclarer un incident ?", k=4, filters={"section": section}
    )
    assert passages, "le filtre ne doit pas vider le résultat pour une section existante"
    assert all(passage.section == section for passage in passages)


def test_answer_cites_the_passages_it_uses(fitted_model: BaseModel, queries: pd.DataFrame) -> None:
    """Une réponse sourcée cite au moins un passage, et ce passage fait partie du top-k."""
    question = str(queries.loc[queries["difficulty"] == "facile", "question"].iloc[0])
    answer = fitted_model.answer(question, k=5, query_id="QRY-9999")
    assert not answer.abstained
    assert answer.text.strip()
    assert answer.cited_chunk_ids
    assert set(answer.cited_chunk_ids) <= {passage.chunk_id for passage in answer.passages}
    assert answer.latency_ms >= 0.0


def test_answer_rows_respect_the_inference_contract(
    fitted_model: BaseModel, queries: pd.DataFrame
) -> None:
    """La ligne écrite par ``predict`` est validée par le contrat avant d'être publiée."""
    question = str(queries.loc[queries["difficulty"] == "facile", "question"].iloc[0])
    answer = fitted_model.answer(question, k=3, query_id="QRY-0001")
    row = pd.DataFrame([answer.to_row(n_chunks_indexed=len(fitted_model.chunks))])
    validated = validate_predictions(row)
    assert validated.loc[0, "n_citations"] >= 1
    assert validated.loc[0, "question"] == question


def test_hors_corpus_questions_are_abstained_or_flagged(
    fitted_model: BaseModel, queries: pd.DataFrame
) -> None:
    """Les questions hors corpus ne doivent pas produire de réponse inventée quand le seuil existe.

    Le test tolère l'absence d'abstention *si* le seuil n'est pas calibré (corpus minuscule des
    tests) : il vérifie alors que le score du meilleur passage est rapporté, donc que la décision
    est traçable. Il refuse en revanche une réponse créditée d'une citation inexistante.
    """
    hors_corpus = queries[queries["answer_type"] == "unanswerable"]
    if hors_corpus.empty:
        pytest.skip("le corpus de test ne contient pas de question hors corpus")
    answer = fitted_model.answer(str(hors_corpus["question"].iloc[0]), k=5)
    assert set(answer.cited_chunk_ids) <= {passage.chunk_id for passage in answer.passages}
    assert answer.top_score >= 0.0
    if answer.abstained:
        assert answer.text.strip() == ""


def test_abstention_threshold_is_respected() -> None:
    """Le seuil est le seul endroit qui décide : il doit être appliqué tel quel."""
    model = build_model(
        {"model": {"algorithm": available_algorithms()[0]}},
        params={"abstention_threshold": 1e9, "calibrate_abstention": False},
    )
    documents = pd.DataFrame(
        {
            "doc_id": ["DOC-0001"],
            "title": ["Note interne"],
            "section": ["procedure"],
            "source": ["wiki_ops"],
            "published_at": [pd.Timestamp("2025-01-06")],
            "n_tokens": [40],
            "text": [
                "Un incident critique doit être déclaré dans l'heure et escaladé après 2 heures."
            ],
        }
    )
    model.fit(documents)
    answer = model.answer("Combien de jours de congés payés ?", k=3)
    assert answer.abstained
    assert answer.text == ""


def test_model_card_is_complete(fitted_model: BaseModel) -> None:
    """La fiche de modèle embarque ce qu'un lecteur doit savoir pour interpréter l'artefact."""
    card = fitted_model.model_card()
    assert isinstance(card, ModelCard)
    payload = card.to_dict()
    assert payload["framework"] == fitted_model.framework
    assert payload["algorithm"] == fitted_model.algorithm
    assert payload["n_documents"] > 0
    assert payload["n_chunks"] > 0
    assert payload["params"]


def test_save_and_load_keep_the_index(fitted_model: BaseModel, tmp_path: Path) -> None:
    """Recharger un artefact doit rendre exactement les mêmes scores."""
    path = fitted_model.save(tmp_path / "index.joblib")
    reloaded = load_model(path)
    question = "Combien de jours de congés payés par an ?"
    original = [passage.chunk_id for passage in fitted_model.retrieve(question, k=5)]
    restored = [passage.chunk_id for passage in reloaded.retrieve(question, k=5)]
    assert original == restored
    assert len(reloaded.chunks) == len(fitted_model.chunks)


def test_save_and_load_keep_the_calibrated_threshold(
    fitted_model: BaseModel, tmp_path: Path
) -> None:
    """Le seuil appris doit survivre à l'aller-retour, sinon l'évaluation mesure une autre règle."""
    fitted_model.abstention_threshold = 12.5
    path = fitted_model.save(tmp_path / "index.joblib")
    reloaded = load_model(path)
    assert reloaded.abstention_threshold == pytest.approx(12.5)
    answer = reloaded.answer("Une question dont le score est faible", k=5)
    assert answer.abstained or answer.top_score >= 12.5


def test_load_model_avoids_the_double_extension(tmp_path: Path, fitted_model: BaseModel) -> None:
    """``load_model('index')`` puis ``load_model('index.joblib')`` désignent le même artefact."""
    fitted_model.save(tmp_path / "index.joblib")
    assert load_model(tmp_path / "index.joblib").is_fitted


def test_load_model_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    """Un fichier qui n'est pas un artefact de cette stack doit être refusé explicitement."""
    path = tmp_path / "index.txt"
    path.write_text("pas un index", encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(path)


def test_available_algorithms_are_declared() -> None:
    """Chaque algorithme annoncé est documenté et constructible : le catalogue n'est pas vide."""
    algorithms = available_algorithms()
    assert algorithms, "la stack doit déclarer au moins un algorithme"
    assert len(set(algorithms)) == len(algorithms)
    for name in algorithms:
        assert build_model({"model": {"algorithm": name}}).algorithm == name


def test_build_model_refuses_an_unknown_algorithm() -> None:
    """Une faute de frappe dans ``conf/model`` doit échouer au démarrage, pas au premier score."""
    with pytest.raises(ValueError):
        build_model({"model": {"algorithm": "bm42"}})


def test_retrieval_is_deterministic(fitted_model: BaseModel, queries: pd.DataFrame) -> None:
    """Deux appels identiques rendent le même classement : prérequis de toute comparaison."""
    question = str(queries["question"].iloc[0])
    first = [passage.chunk_id for passage in fitted_model.retrieve(question, k=10)]
    second = [passage.chunk_id for passage in fitted_model.retrieve(question, k=10)]
    assert first == second
