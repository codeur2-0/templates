"""Contrats Pandera du corpus de résumé, et les liens **entre** les tables.

Ces tests vérifient deux choses différentes, et les deux comptent :

* chaque table respecte son contrat (bornes, valeurs autorisées, clés uniques), et une table qui
  le casse lève ``SchemaError`` ;
* :func:`~src.data.schemas.validate_corpus` refuse les incohérences **croisées** — un résumé de
  référence orphelin, un document sans référence, un fait qui pointe hors de son document — par un
  ``ValueError``, comme la couche d'entités nommées : ce sont les trois erreurs qui ne se voient pas
  dans un score et qui pourtant le faussent.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.schemas import (
    FACT_TYPES,
    fact_columns,
    salient_fact_counts,
    split_sizes,
    validate_corpus,
    validate_documents,
    validate_facts,
    validate_predictions,
    validate_references,
)


def test_documents_respect_their_contract(documents: pd.DataFrame) -> None:
    """The generated documents pass their own schema (identifiers, bounds, allowed values)."""
    validated = validate_documents(documents)
    assert len(validated) == len(documents)
    assert validated["doc_id"].is_unique


def test_references_respect_their_contract(references: pd.DataFrame) -> None:
    """The reference summaries pass their schema, sentence bound included."""
    validated = validate_references(references)
    assert len(validated) == len(references)
    assert validated["doc_id"].is_unique
    assert validated["n_sentences"].between(2, 8).all()


def test_facts_respect_their_contract(facts: pd.DataFrame) -> None:
    """The fact table carries the seven declared types and a boolean salience."""
    validated = validate_facts(facts)
    assert set(validated["fact_type"]) == set(FACT_TYPES)
    assert set(validated["salient"].unique()) <= {0, 1}
    assert validated["fact_id"].is_unique


def test_fact_columns_are_stable() -> None:
    """The published fidelity columns do not move without a decision."""
    assert fact_columns() == (
        "covered_equipement",
        "covered_symptome",
        "covered_cause",
        "covered_action",
        "covered_piece",
        "covered_duree",
        "covered_statut",
    )


def test_corpus_validation_accepts_the_generated_tables(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> None:
    """The three tables validate together, which is the contract of the pipeline."""
    result = validate_corpus(documents, references, facts)
    assert len(result) == 3


def test_corpus_validation_rejects_an_orphan_reference(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> None:
    """A reference summary of an unknown document is refused, with the culprit named."""
    broken = references.copy()
    broken.loc[0, "doc_id"] = "CR-9999"
    with pytest.raises(ValueError, match="unknown document"):
        validate_corpus(documents, broken, facts)


def test_corpus_validation_rejects_a_document_without_reference(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> None:
    """A document that no reference covers is refused: it cannot be scored."""
    broken = references.iloc[1:].copy()
    with pytest.raises(ValueError, match="without a reference summary"):
        validate_corpus(documents, broken, facts)


def test_corpus_validation_rejects_a_fact_outside_its_document(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> None:
    """A fact that points at a sentence the document does not have is refused.

    L'index reste **dans les bornes du contrat de la table** (0-39) : c'est bien l'incohérence
    croisée qui est testée ici, pas une valeur illégale — le document le plus court du corpus sert
    de cible, et son propre nombre de phrases est l'index juste au-delà du dernier.
    """
    sizes = documents.set_index("doc_id")["n_sentences"]
    target = str(sizes.idxmin())
    broken = facts.copy()
    broken.loc[broken["doc_id"] == target, "sentence_index"] = int(sizes.min())
    assert int(sizes.min()) <= 39, "l'index doit rester dans les bornes du contrat de la table"
    with pytest.raises(ValueError, match="outside the sentences"):
        validate_corpus(documents, references, broken)


def test_split_sizes_are_read_from_the_corpus(documents: pd.DataFrame) -> None:
    """The split is written in the corpus, so its sizes are a fact, not a draw."""
    sizes = split_sizes(documents)
    assert set(sizes) == {"train", "val", "calibration", "test"}
    assert sum(sizes.values()) == len(documents)
    assert sizes["train"] > sizes["test"]


def test_salient_fact_counts_cover_every_type(facts: pd.DataFrame) -> None:
    """Every fact type has salient examples: an unmeasurable type would hide an omission."""
    counts = salient_fact_counts(facts)
    assert set(counts) == set(FACT_TYPES)
    assert all(count > 0 for count in counts.values())


def test_predictions_contract_rejects_an_unknown_document(
    published_predictions: pd.DataFrame,
) -> None:
    """La table publiée passe son contrat, et un identifiant hors format est refusé."""
    validated = validate_predictions(published_predictions)
    assert validated["doc_id"].str.match(r"^CR-\d{4}$").all()
    broken = published_predictions.copy()
    broken.loc[0, "doc_id"] = "not-a-document"
    with pytest.raises(SchemaError):
        validate_predictions(broken)


def test_predictions_contract_refuses_an_extra_column(
    published_predictions: pd.DataFrame,
) -> None:
    """La table publiée est **exactement** celle du contrat : une colonne en trop est refusée.

    C'est la différence entre une violation de colonne (``SchemaError``) et une violation de forme
    de table (``SchemaErrors``) : les deux sont attrapées, aucune n'est supposée.
    """
    broken = published_predictions.assign(colonne_en_trop=1)
    with pytest.raises(SchemaErrors):
        validate_predictions(broken)


def test_a_table_without_its_column_raises_a_schema_error(references: pd.DataFrame) -> None:
    """A missing column breaks the table's own contract, so it raises ``SchemaError``.

    C'est la distinction que la suite vérifie : ``SchemaError`` pour une table, ``ValueError`` pour
    l'incohérence entre deux tables.
    """
    broken = references.drop(columns=["summary"])
    with pytest.raises(SchemaError):
        validate_references(broken)
