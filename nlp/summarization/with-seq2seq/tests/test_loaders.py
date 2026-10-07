"""Chargeurs : les **trois** tables du corpus, leurs splits, leurs métadonnées.

Le corpus de résumé n'est pas une table mais trois (documents, résumés de référence, faits annotés)
que tout le reste du projet lit **ensemble** : un chargeur qui les désaligne fait comparer un résumé
au document d'un autre, ou une couverture de faits à des faits qui ne viennent pas du texte résumé —
et rien n'échoue. Les tests ci-dessous vérifient donc l'alignement (mêmes identifiants, mêmes
splits), l'aller-retour Parquet/CSV, la lecture d'un split unique et les refus attendus.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import SummaryCorpusLoader
from src.utils.paths import ProjectPaths


def test_persisting_then_loading_keeps_the_three_tables_aligned(
    persisted_corpus: SummaryCorpusLoader,
) -> None:
    """L'aller-retour disque rend les trois tables alignées sur les mêmes identifiants.

    C'est le contrat dont dépend le reste du projet : ``references`` et ``facts`` portent le même
    ``doc_id`` que ``documents``, dans le même split, et l'ordre de lecture est celui de la table
    des documents (les splits sont découpés par position).
    """
    documents, references, facts = persisted_corpus.load_corpus()

    assert len(documents) == len(references)
    assert set(documents["doc_id"]) == set(references["doc_id"])
    assert set(facts["doc_id"]) <= set(documents["doc_id"])
    assert list(documents["doc_id"]) == list(references["doc_id"])
    # Le split est une propriété des documents : les faits le retrouvent par jointure.
    merged = facts.merge(documents[["doc_id", "split"]], on="doc_id", how="left")
    assert merged["split"].notna().all()
    assert set(documents["split"]) == set(merged["split"])


def test_the_published_recipe_counts_the_splits_and_the_fact_types(
    persisted_corpus: SummaryCorpusLoader,
) -> None:
    """``split_sizes`` et ``salient_facts`` publient la recette du corpus, jamais zéro type."""
    documents, _, _ = persisted_corpus.load_corpus()
    sizes = persisted_corpus.split_sizes()
    salient = persisted_corpus.salient_facts()

    assert sum(sizes.values()) == len(documents)
    assert all(name in sizes for name in ("train", "val", "calibration", "test"))
    assert sizes["train"] > sizes["test"] > 0
    assert salient and all(count > 0 for count in salient.values())
    assert sum(salient.values()) > 0


def test_a_single_split_never_borrows_rows_from_its_neighbours(
    persisted_corpus: SummaryCorpusLoader,
) -> None:
    """``load_split`` restreint les **trois** tables au même sous-ensemble de documents."""
    documents, references, facts = persisted_corpus.load_split("test")

    assert set(documents["split"]) == {"test"}
    assert set(references["doc_id"]) == set(documents["doc_id"])
    assert set(facts["doc_id"]) <= set(documents["doc_id"])
    assert not references.empty


def test_an_unknown_split_is_refused_instead_of_returning_everything(
    persisted_corpus: SummaryCorpusLoader,
) -> None:
    """Un nom de split fautif lève ``ValueError`` : rendre le corpus entier serait silencieux."""
    with pytest.raises(ValueError, match="Unknown split"):
        persisted_corpus.load_split("entrainement")


def test_the_generation_metadata_travels_with_the_corpus(
    persisted_corpus: SummaryCorpusLoader,
) -> None:
    """La recette du corpus (graine, taille, version du générateur) est relue telle qu'écrite."""
    payload = persisted_corpus.load_metadata()

    assert isinstance(payload, dict) and payload
    assert all(isinstance(key, str) for key in payload)


def test_a_corrupted_table_is_refused_before_it_reaches_the_rest_of_the_project(
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Une référence orpheline est refusée à la lecture : le contrat se vérifie à l'entrée.

    La corruption est écrite sur le disque (et non injectée dans une frame) : c'est le chemin réel
    d'``mode=evaluate`` — un fichier produit par un autre outil, ou tronqué. Le refus vient de la
    cohérence **croisée** des trois tables, celle qu'aucun score ne révèle.
    """
    loader = SummaryCorpusLoader(ProjectPaths.from_root(tmp_path))
    loader.paths.ensure()
    orphans = pd.concat(
        [references, references.head(1).assign(doc_id="CR-9999")], ignore_index=True
    )
    loader.save_documents(documents)
    loader.save_references(orphans)
    loader.save_facts(facts)

    with pytest.raises(ValueError, match="unknown document"):
        loader.load_corpus()


def test_loading_without_validation_stays_possible_for_inspection(
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """``validation_enabled=False`` sert l'inspection : la lecture reste possible, et complète."""
    loader = SummaryCorpusLoader(
        ProjectPaths.from_root(tmp_path),
        validation_enabled=False,
    )
    loader.paths.ensure()
    loader.save_documents(documents)
    loader.save_references(references)
    loader.save_facts(facts)

    loaded, loaded_references, loaded_facts = loader.load_corpus()

    assert len(loaded) == len(documents)
    assert len(loaded_references) == len(references)
    assert len(loaded_facts) == len(facts)
