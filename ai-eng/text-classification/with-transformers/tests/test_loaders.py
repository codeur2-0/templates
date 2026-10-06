"""Loader du corpus étiqueté : écriture multi-format, relecture, métadonnées obligatoires.

Le loader est la frontière du projet : tout ce qui entre dans ``data/raw`` passe par lui, et tout
ce qui en sort est validé. Ces tests vérifient les quatre promesses :

* écrire puis relire rend la même table (Parquet pour les machines, CSV pour la relecture humaine) ;
* un split demandé est un split réel — un nom inconnu ou un split vide sont des erreurs explicites,
  pas un tableau vide qui se propagerait jusqu'au rapport ;
* les métadonnées de génération sont obligatoires : un corpus sans recette ne se défend pas ;
* une prédiction hors nomenclature n'atteint jamais le disque.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.data.loaders import METADATA_FILE, TextLabelLoader
from src.data.schemas import validate_tickets
from src.utils.paths import ProjectPaths


def test_the_documents_survive_a_parquet_round_trip(
    documents: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Écrire puis relire rend la même table, colonnes et valeurs.

    La comparaison porte sur la frame *validée* : le contrat Pandera coerce les types (la date
    passe d'objet à ``datetime64``), et c'est bien cette frame-là que le loader écrit.
    """
    sample = documents.head(40)
    expected = validate_tickets(sample).reset_index(drop=True)
    loader = TextLabelLoader(tiny_paths)

    written = loader.save_documents(sample)
    reloaded = loader.load_documents().reset_index(drop=True)

    assert loader.documents_path in written.values()
    pd.testing.assert_frame_equal(reloaded, expected, check_dtype=False)


def test_the_csv_copy_is_written_when_it_is_configured(
    documents: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Le CSV est la copie lisible : il est écrit quand le format est demandé, pas tout le temps."""
    loader = TextLabelLoader(tiny_paths, formats=["parquet", "csv"])

    written = loader.save_documents(documents.head(10))

    assert {path.suffix for path in written.values()} == {".parquet", ".csv"}
    assert all(path.exists() for path in written.values())


def test_an_unsupported_format_is_refused(
    documents: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Un format non géré est refusé au moment d'écrire, avec la liste des formats permis."""
    loader = TextLabelLoader(tiny_paths, formats=["feather"])

    with pytest.raises(ValueError, match="Unsupported table format"):
        loader.save_documents(documents.head(5))


def test_the_metadata_round_trip_carries_the_recipe(tiny_paths: ProjectPaths) -> None:
    """La recette de génération est archivée telle quelle et relue à l'identique."""
    loader = TextLabelLoader(tiny_paths)
    metadata = {"seed": 7, "n_documents": 300, "nested": {"a": [1, 2, 3]}}

    path = loader.save_metadata(metadata)

    assert path.name == METADATA_FILE
    assert loader.load_metadata() == metadata


def test_a_corpus_without_its_recipe_cannot_be_read(
    documents: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Les métadonnées absentes sont tolérées tant que rien n'a été généré, jamais après."""
    loader = TextLabelLoader(tiny_paths)

    assert loader.load_metadata() == {}
    loader.save_documents(documents.head(10))

    with pytest.raises(FileNotFoundError, match="metadata missing"):
        loader.load_metadata()


def test_reading_a_missing_corpus_fails_loudly(tiny_paths: ProjectPaths) -> None:
    """Relire un corpus qui n'existe pas est une erreur explicite, jamais un tableau vide."""
    with pytest.raises(FileNotFoundError, match="generate the dataset first"):
        TextLabelLoader(tiny_paths).load_documents()


def test_the_split_helper_filters_the_corpus(
    persisted_corpus: TextLabelLoader, documents: pd.DataFrame
) -> None:
    """``split`` rend les lignes du split demandé, réindexées."""
    test_rows = persisted_corpus.split("test")

    assert len(test_rows) == int((documents["split"] == "test").sum())
    assert set(test_rows["split"]) == {"test"}
    assert list(test_rows.index) == list(range(len(test_rows)))


def test_an_unknown_split_is_an_error(persisted_corpus: TextLabelLoader) -> None:
    """Un nom de split inconnu est refusé : le test ne doit jamais valider un split vide."""
    with pytest.raises(ValueError, match="Unknown split"):
        persisted_corpus.split("inconnu")


def test_the_split_helper_also_works_on_a_frame_in_memory(loader: TextLabelLoader) -> None:
    """Le même découpage sert au trainer, qui travaille en mémoire."""
    frame = pd.DataFrame(
        {
            "doc_id": ["TKT-0001", "TKT-0002"],
            "text": ["a" * 40, "b" * 40],
            "label": ["autre", "autre"],
            "split": ["train", "test"],
        }
    )

    selected = loader.split("train", frame=frame)

    assert list(selected["doc_id"]) == ["TKT-0001"]


def test_the_label_distribution_is_a_count_table(
    persisted_corpus: TextLabelLoader, documents: pd.DataFrame
) -> None:
    """La distribution des classes est ventilée par split : le premier chiffre relu."""
    distribution = persisted_corpus.label_distribution()

    assert int(distribution["n_documents"].sum()) == len(documents)
    assert set(distribution["label"]) == set(documents["label"].unique())
    per_split = distribution.groupby("split")["share"].sum()
    assert all(abs(float(value) - 1.0) < 1e-9 for value in per_split)


def test_the_prediction_table_is_validated_before_being_written(
    predictions_frame: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Le loader persiste les prédictions **après** validation du contrat."""
    loader = TextLabelLoader(tiny_paths)

    written = loader.save_predictions(predictions_frame)

    assert written.exists()
    assert len(pd.read_csv(written)) == len(predictions_frame)


def test_the_prediction_table_rejects_an_unknown_label(
    predictions_frame: pd.DataFrame, tiny_paths: ProjectPaths
) -> None:
    """Une prédiction hors nomenclature ne doit jamais atteindre le disque."""
    broken = predictions_frame.copy()
    broken.loc[broken.index[0], "prediction"] = "reclamation"

    with pytest.raises(pa.errors.SchemaError):
        TextLabelLoader(tiny_paths).save_predictions(broken)


def test_the_loader_creates_its_directories(tmp_path: Path) -> None:
    """Le premier appel crée ``data/raw`` : aucun pré-requis manuel dans le README."""
    loader = TextLabelLoader(ProjectPaths.from_root(tmp_path))

    assert not loader.paths.raw_dir.exists()
    path = loader.save_metadata({"seed": 1})

    assert path.exists()
    assert loader.metadata_path.parent == loader.paths.raw_dir
