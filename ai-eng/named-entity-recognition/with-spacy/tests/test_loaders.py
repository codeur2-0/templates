"""Persistance et découpage : les deux tables, leurs splits, leurs distributions.

Un loader de corpus annoté a deux façons de se tromper en silence, et les deux sont testées ici :

* **désaligner les tables** — un découpage fait sur les messages mais pas sur les annotations
produit
  un entraînement sur les messages d'un split et la supervision d'un autre. Le test compare les
  identifiants des deux côtés, à chaque split ;
* **valider trop tard** — un corpus écrit sans contrat ne casse qu'au moment de l'entraînement, loin
  de la cause. Le test vérifie qu'une table non conforme est refusée **à l'écriture**.

Le reste (aller-retour Parquet/CSV, métadonnées, prédictions) vérifie que ce qui est relu est
exactement ce qui a été écrit : un corpus qui se relit « à peu près » ne rend pas les runs
comparables.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.loaders import ANNOTATIONS_STEM, METADATA_FILE, EntityCorpusLoader
from src.data.schemas import ENTITY_LABELS, SPLITS
from src.utils.io import read_table


def test_persisting_and_reloading_returns_the_same_tables(tiny_paths, documents, spans) -> None:
    """L'aller-retour Parquet/CSV rend les deux tables à l'identique."""
    loader = EntityCorpusLoader(tiny_paths)
    written_documents = loader.save_documents(documents)
    written_spans = loader.save_annotations(spans)
    loader.save_metadata({"seed": 13, "n_documents": len(documents)})

    reloaded_documents, reloaded_spans = loader.load_corpus()

    assert {"parquet", "csv"} <= set(written_documents)
    assert set(written_spans) == set(written_documents)
    pd.testing.assert_frame_equal(reloaded_documents, documents, check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(reloaded_spans, spans, check_like=True, check_dtype=False)
    assert loader.load_metadata()["seed"] == 13


def test_the_declared_paths_are_the_ones_written(loader) -> None:
    """Les chemins publiés portent les noms déclarés (``spans``, ``generation_metadata``)."""
    assert loader.documents_path.name == f"{loader.dataset_name}.parquet"
    assert loader.annotations_path.name == f"{ANNOTATIONS_STEM}.parquet"
    assert loader.metadata_path.name == METADATA_FILE
    assert loader.predictions_path.parent.name == "reports"


@pytest.mark.parametrize("split_name", SPLITS)
def test_every_split_keeps_the_two_tables_aligned(
    persisted_corpus: EntityCorpusLoader, split_name: str
) -> None:
    """Un split découpe les **deux** tables, et les identifiants coïncident exactement."""
    selected = persisted_corpus.split(split_name)
    annotated = persisted_corpus.split_annotations(split_name)

    assert not selected.empty
    assert set(annotated["msg_id"].astype(str)) <= set(selected["msg_id"].astype(str))
    assert set(selected["split"].unique()) == {split_name}
    assert annotated["start"].is_monotonic_increasing or len(annotated) > 1


def test_an_unknown_split_is_rejected(persisted_corpus: EntityCorpusLoader) -> None:
    """Un split qui n'existe pas lève, au lieu de rendre un corpus vide."""
    with pytest.raises(ValueError):
        persisted_corpus.split("validation")


def test_the_entity_distribution_describes_every_split_and_type(
    persisted_corpus: EntityCorpusLoader,
) -> None:
    """La distribution croise splits et types, et ses compteurs somment au corpus."""
    distribution = persisted_corpus.entity_distribution()

    assert set(distribution["split"]) == set(SPLITS)
    assert set(distribution["label"]) == set(ENTITY_LABELS)
    assert len(distribution) == len(SPLITS) * len(ENTITY_LABELS)
    total = int(distribution["n_mentions"].sum())
    assert total == len(persisted_corpus.load_annotations())
    assert distribution["share"].between(0.0, 1.0).all()
    assert distribution["holdout_share"].between(0.0, 1.0).all()


def test_a_non_conforming_table_is_refused_before_being_written(tiny_paths, documents) -> None:
    """Le contrat est appliqué à l'écriture : un corpus invalide ne touche pas le disque."""
    loader = EntityCorpusLoader(tiny_paths)
    broken = documents.copy()
    broken.loc[broken.index[0], "text"] = "trop court"

    with pytest.raises((SchemaError, SchemaErrors)):
        loader.save_documents(broken)

    assert not loader.documents_path.exists()


def test_the_predictions_table_is_written_and_validated(tiny_paths, predictions) -> None:
    """Les prédictions sont validées puis écrites dans ``artifacts/reports``."""
    written = EntityCorpusLoader(tiny_paths).save_predictions(predictions)

    assert written.exists()
    assert written.parent == tiny_paths.reports_dir
    # Le chemin est déclaré (``predictions.csv``) : c'est ``read_table`` qui choisit le lecteur,
    # jamais un appel direct au format supposé.
    reloaded = read_table(written)
    assert len(reloaded) == len(predictions)
    assert set(reloaded.columns) == set(predictions.columns)
    assert reloaded["latency_ms"].ge(0).all()


def test_validation_can_be_disabled_explicitly(documents, tmp_path) -> None:
    """``validation_enabled=False`` lit une table non conforme : c'est un choix explicite."""
    from src.utils.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    broken = documents.copy()
    broken.loc[broken.index[0], "msg_id"] = "pas-un-identifiant"
    loader = EntityCorpusLoader(paths, validation_enabled=False)
    loader.save_documents(broken)

    reloaded = loader.load_documents()

    assert len(reloaded) == len(broken)
    with pytest.raises((SchemaError, SchemaErrors)):
        EntityCorpusLoader(paths, validation_enabled=True).load_documents()
