"""Contrats Pandera : ce que la suite vérifie, et pourquoi.

Un schéma n'est utile que s'il **refuse** ce qu'il prétend interdire. Les tests ci-dessous
cassent donc chaque contrat volontairement (une colonne manquante, une date impossible, un
identifiant mal formé, une question hors corpus qui porte un extrait) et vérifient que Pandera
lève. Le test qui valide un jeu conforme est là pour l'autre moitié du contrat : aucune règle ne
doit être assez stricte pour rejeter les données réellement produites par le générateur.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from src.data.schemas import (
    ChunksSchema,
    DocumentsSchema,
    PredictedAnswersSchema,
    QueriesSchema,
    validate_chunks,
    validate_documents,
    validate_predictions,
    validate_queries,
)


def _chunks_frame(queries: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal, valid passage table pointing at the first four documents."""
    rows = []
    for index in range(4):
        rows.append(
            {
                "chunk_id": f"DOC-000{index + 1}-C01",
                "doc_id": f"DOC-{index + 1:04d}",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 120,
                "n_tokens": 24,
                "text": "Un passage de test suffisamment long pour respecter le contrat.",
            }
        )
    assert not queries.empty
    return pd.DataFrame(rows)


def _predictions_frame() -> pd.DataFrame:
    """Build a minimal, valid prediction table."""
    return pd.DataFrame(
        [
            {
                "query_id": "QRY-0001",
                "question": "Quelle est la durée de la période d'essai ?",
                "answer": "Trois mois.",
                "citations": "DOC-0001-C01|DOC-0001",
                "n_citations": 1,
                "abstained": False,
                "top_score": 12.5,
                "n_chunks_indexed": 12,
                "latency_ms": 3.2,
            }
        ]
    )


def test_generated_documents_respect_the_contract(documents: pd.DataFrame) -> None:
    """Le corpus produit par le générateur doit passer le contrat sans correction."""
    validated = validate_documents(documents)
    assert len(validated) == len(documents)
    assert validated["doc_id"].is_unique
    assert (validated["n_tokens"] >= 20).all()


def test_generated_queries_respect_the_contract(queries: pd.DataFrame) -> None:
    """Les questions annotées doivent passer le contrat, y compris les questions hors corpus."""
    validated = validate_queries(queries)
    unanswerable = validated["answer_type"] == "unanswerable"
    assert unanswerable.any(), "le jeu de test doit contenir des questions hors corpus"
    assert (validated.loc[unanswerable, "gold_span"].str.strip() == "").all()
    assert (validated.loc[~unanswerable, "gold_doc_ids"].str.len() > 0).all()


def test_documents_contract_rejects_a_missing_column(documents: pd.DataFrame) -> None:
    """Une colonne manquante doit faire échouer la validation (pas de contrat laxiste)."""
    broken = documents.drop(columns=["text"])
    with pytest.raises(pa.errors.SchemaError):
        validate_documents(broken)


def test_documents_contract_rejects_an_unknown_identifier_pattern(
    documents: pd.DataFrame,
) -> None:
    """Un identifiant qui n'est pas au format ``DOC-0000`` doit être refusé."""
    broken = documents.copy()
    broken.loc[broken.index[0], "doc_id"] = "doc-1"
    with pytest.raises(pa.errors.SchemaError):
        validate_documents(broken)


def test_documents_contract_rejects_a_duplicate_identifier(documents: pd.DataFrame) -> None:
    """Deux documents ne peuvent pas partager un identifiant : les joints seraient fausses."""
    broken = pd.concat([documents, documents.head(1)], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        validate_documents(broken)


def test_documents_contract_rejects_an_unknown_source(documents: pd.DataFrame) -> None:
    """Une source hors nomenclature doit être refusée (les rapports agrègent par source)."""
    broken = documents.copy()
    broken.loc[broken.index[0], "source"] = "wiki_inconnu"
    with pytest.raises(pa.errors.SchemaError):
        validate_documents(broken)


def test_queries_contract_rejects_an_answerable_question_without_gold(
    queries: pd.DataFrame,
) -> None:
    """Une question déclarée répondable sans document annoté est une annotation cassée."""
    broken = queries[queries["answer_type"] != "unanswerable"].head(1).copy()
    broken.loc[broken.index[0], "gold_doc_ids"] = ""
    with pytest.raises(pa.errors.SchemaError):
        validate_queries(broken)


def test_queries_contract_rejects_an_unanswerable_question_with_a_span(
    queries: pd.DataFrame,
) -> None:
    """Une question hors corpus ne doit pas porter d'extrait : le score de rappel mentirait."""
    broken = queries[queries["answer_type"] == "unanswerable"].head(1).copy()
    broken.loc[broken.index[0], "gold_span"] = "Un extrait qui n'existe pas."
    with pytest.raises(pa.errors.SchemaError):
        validate_queries(broken)


def test_queries_contract_rejects_an_unknown_split(queries: pd.DataFrame) -> None:
    """Un split inconnu fausserait la sélection de modèle ; il doit être refusé."""
    broken = queries.head(1).copy()
    broken.loc[broken.index[0], "split"] = "validation"
    with pytest.raises(pa.errors.SchemaError):
        validate_queries(broken)


def test_queries_contract_rejects_an_empty_question(queries: pd.DataFrame) -> None:
    """Une question vide (ou de trois caractères) ne doit jamais atteindre le retriever."""
    broken = queries.head(1).copy()
    broken.loc[broken.index[0], "question"] = "?"
    with pytest.raises(pa.errors.SchemaError):
        validate_queries(broken)


def test_lazy_validation_collects_every_failure(documents: pd.DataFrame) -> None:
    """En mode *lazy*, Pandera rapporte toutes les erreurs d'un coup (utile en CI)."""
    broken = documents.copy()
    broken.loc[broken.index[0], "doc_id"] = "nope"
    broken.loc[broken.index[1], "source"] = "wiki_inconnu"
    broken.loc[broken.index[2], "text"] = "court"
    with pytest.raises(pa.errors.SchemaErrors) as error:
        validate_documents(broken, lazy=True)
    assert len(error.value.failure_cases) >= 3


def test_chunks_contract_accepts_a_valid_table(queries: pd.DataFrame) -> None:
    """Le contrat des passages doit accepter une table conforme."""
    frame = _chunks_frame(queries)
    validated = validate_chunks(frame)
    assert len(validated) == 4


def test_chunks_contract_rejects_inverted_offsets(queries: pd.DataFrame) -> None:
    """Un passage qui se termine avant de commencer est un bug de découpage, pas une donnée."""
    broken = _chunks_frame(queries)
    broken.loc[broken.index[0], "end_char"] = 0
    with pytest.raises(pa.errors.SchemaError):
        validate_chunks(broken)


def test_predictions_contract_accepts_a_valid_row() -> None:
    """Le contrat d'inférence valide ce que le pipeline écrit réellement."""
    validated = validate_predictions(_predictions_frame())
    # ``abstained`` est un entier 0/1 : le contrat l'impose (``isin=[0, 1]``) parce que le CSV
    # d'inférence doit rester lisible par un tableur, où un booléen devient TRUE/FALSE.
    assert set(validated["abstained"].unique()) <= {0, 1}
    assert validated["n_citations"].dtype.kind == "i"


def test_predictions_contract_rejects_a_negative_latency() -> None:
    """Une latence négative signalerait une horloge mal lue : le contrat la refuse."""
    broken = _predictions_frame()
    broken.loc[broken.index[0], "latency_ms"] = -1.0
    with pytest.raises(pa.errors.SchemaError):
        validate_predictions(broken)


def test_every_contract_is_introspectable() -> None:
    """Les quatre contrats restent interrogeables : le README et le data card les recopient."""
    documents = DocumentsSchema.to_schema()
    assert documents.columns["doc_id"].unique
    assert set(QueriesSchema.to_schema().columns) >= {"question", "gold_span", "split"}
    assert set(ChunksSchema.to_schema().columns) >= {"chunk_id", "doc_id", "text"}
    assert set(PredictedAnswersSchema.to_schema().columns) >= {"answer", "citations", "abstained"}


def test_chunk_and_answer_schemas_declare_their_columns() -> None:
    """Chaque table publiée a un contrat complet : rien n'est validé « par convention »."""
    assert set(ChunksSchema.to_schema().columns) == {
        "chunk_id",
        "doc_id",
        "chunk_index",
        "start_char",
        "end_char",
        "n_tokens",
        "text",
    }
    assert "citations" in PredictedAnswersSchema.to_schema().columns
    assert "gold_span" in QueriesSchema.to_schema().columns
