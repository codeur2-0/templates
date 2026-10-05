"""Loaders : écriture, relecture et refus des fichiers incohérents.

Le loader est la frontière du projet : tout ce qui entre (corpus, questions, fichier
d'inférence) et tout ce qui sort (tables, métadonnées) passe par lui. Les tests vérifient trois
propriétés qui ont chacune un coût réel quand elles manquent :

* un aller-retour Parquet/CSV rend **exactement** la même table (sinon les métriques d'un
  notebook et celles d'un pipeline divergent) ;
* un split inexistant lève une erreur explicite au lieu de rendre un tableau vide — un rappel
  calculé sur zéro question vaut 0 et ferait croire à un modèle cassé ;
* un fichier d'inférence sans colonne ``question`` est refusé, avec la liste des colonnes
  trouvées dans le message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.data.loaders import TextCorpusLoader
from src.utils.paths import ProjectPaths


def test_documents_round_trip_is_lossless(
    loader: TextCorpusLoader, documents: pd.DataFrame
) -> None:
    """Le corpus écrit puis relu doit être identique (Parquet pour les machines)."""
    written = loader.save_documents(documents)
    assert set(written) == {"parquet", "csv"}
    reloaded = loader.load_documents()
    # Parquet stocke les horodatages à la seconde : la résolution revient en nanosecondes à la
    # lecture. Les valeurs, elles, restent identiques — c'est ce que le test vérifie.
    pd.testing.assert_frame_equal(reloaded, documents, check_dtype=False)
    assert (reloaded["published_at"] == documents["published_at"]).all()


def test_queries_round_trip_keeps_the_dtypes(
    loader: TextCorpusLoader, queries: pd.DataFrame
) -> None:
    """Les types des questions (entiers, catégories textuelles) survivent à l'aller-retour."""
    loader.save_queries(queries)
    reloaded = loader.load_queries()
    assert reloaded["n_gold_docs"].dtype.kind == "i"
    assert set(reloaded["answer_type"]) == set(queries["answer_type"])
    pd.testing.assert_frame_equal(reloaded, queries, check_dtype=False)


def test_load_queries_filters_on_the_requested_split(
    loader: TextCorpusLoader, queries: pd.DataFrame
) -> None:
    """``load_queries('test')`` ne rend que le split demandé, réindexé."""
    loader.save_queries(queries)
    test_split = loader.load_queries("test")
    assert len(test_split) == int((queries["split"] == "test").sum())
    assert set(test_split["split"]) == {"test"}
    assert test_split.index.tolist() == list(range(len(test_split)))


def test_load_queries_rejects_an_unknown_split(
    loader: TextCorpusLoader, queries: pd.DataFrame
) -> None:
    """Un split absent doit lever, en citant ceux qui existent."""
    loader.save_queries(queries)
    with pytest.raises(ValueError, match="Unknown split 'holdout'"):
        loader.load_queries("holdout")


def test_load_documents_reports_a_missing_file(tmp_path: Path) -> None:
    """Un corpus absent doit lever ``FileNotFoundError`` avec le chemin complet."""
    empty = TextCorpusLoader(ProjectPaths.from_root(tmp_path))
    with pytest.raises(FileNotFoundError):
        empty.load_documents()


def test_chunks_round_trip(loader: TextCorpusLoader, fitted_model: object) -> None:
    """Les passages produits par le modèle se relisent via le même contrat."""
    chunks = fitted_model.chunks  # type: ignore[attr-defined]
    loader.save_chunks(chunks)
    reloaded = loader.load_chunks()
    pd.testing.assert_frame_equal(reloaded, chunks)


def test_generation_metadata_round_trip(loader: TextCorpusLoader) -> None:
    """Les métadonnées de génération (graine, compteurs) sont relues telles quelles."""
    payload = {"seed": 7, "n_documents": 64, "n_questions": 120}
    path = loader.save_metadata(payload)
    assert path.name == "generation_metadata.json"
    assert loader.load_metadata() == payload


def test_load_metadata_is_tolerant_when_absent(loader: TextCorpusLoader) -> None:
    """Un corpus ancien, sans métadonnées, reste exploitable (le rapport s'adapte)."""
    loader.metadata_path.unlink(missing_ok=True)
    assert loader.load_metadata() == {}


def test_metadata_is_written_as_utf8_json(loader: TextCorpusLoader) -> None:
    """Le fichier doit rester lisible par un humain : UTF-8, accents conservés."""
    loader.save_metadata({"dataset_name": "base_de_connaissances"})
    raw = loader.metadata_path.read_text(encoding="utf-8")
    assert "base_de_connaissances" in raw
    assert json.loads(raw)["dataset_name"] == "base_de_connaissances"


def test_inference_file_accepts_a_question_column(loader: TextCorpusLoader, tmp_path: Path) -> None:
    """Un CSV de questions suffit : les identifiants manquants sont générés."""
    path = tmp_path / "questions.csv"
    pd.DataFrame({"question": ["Combien de jours de congés ?"]}).to_csv(path, index=False)
    frame = loader.load_inference_questions(path)
    assert frame.columns.tolist() == ["query_id", "question"]
    assert frame.loc[0, "query_id"].startswith("QRY-")


def test_inference_file_keeps_the_provided_identifiers(
    loader: TextCorpusLoader, tmp_path: Path
) -> None:
    """Un identifiant fourni par l'utilisateur est conservé (traçabilité de la demande)."""
    path = tmp_path / "questions.csv"
    pd.DataFrame(
        {"query_id": ["TICKET-42"], "question": ["Quelle est la politique de télétravail ?"]}
    ).to_csv(path, index=False)
    frame = loader.load_inference_questions(path)
    assert frame.loc[0, "query_id"] == "TICKET-42"


def test_inference_file_without_question_is_rejected(
    loader: TextCorpusLoader, tmp_path: Path
) -> None:
    """Le message d'erreur doit lister les colonnes trouvées : c'est ce qui fait gagner du temps."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["bonjour"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="must contain a 'question' column"):
        loader.load_inference_questions(path)


def test_inference_json_payload_is_supported(loader: TextCorpusLoader, tmp_path: Path) -> None:
    """Le JSON est accepté : c'est le format d'un service appelant qui poste une question."""
    path = tmp_path / "question.json"
    payload = json.dumps([{"question": "Quelle est la durée de la période d'essai ?"}])
    path.write_text(payload, encoding="utf-8")
    frame = loader.load_inference_questions(path)
    assert len(frame) == 1
    assert "période d'essai" in frame.loc[0, "question"]


def test_a_corrupted_file_is_rejected_on_read(
    loader: TextCorpusLoader, documents: pd.DataFrame
) -> None:
    """Un fichier corrompu sur le disque doit être refusé, avec le contrat nommé."""
    loader.save_documents(documents)
    broken = documents.copy()
    broken.loc[broken.index[0], "source"] = "wiki_inconnu"
    broken.to_parquet(loader.documents_path, index=False)

    with pytest.raises(pa.errors.SchemaError):
        loader.load_documents()


def test_validation_can_be_disabled_per_call(
    loader: TextCorpusLoader, documents: pd.DataFrame
) -> None:
    """Relire sans valider sert aux notebooks qui explorent des données volontairement cassées."""
    loader.save_documents(documents)
    broken = documents.copy()
    broken.loc[broken.index[0], "source"] = "wiki_inconnu"
    broken.to_parquet(loader.documents_path, index=False)

    without_validation = loader.load_documents(validate=False)
    assert len(without_validation) == len(documents)
    assert without_validation.loc[0, "source"] == "wiki_inconnu"


def test_csv_copy_is_written_for_humans(loader: TextCorpusLoader, documents: pd.DataFrame) -> None:
    """Le CSV accompagne le Parquet : un lecteur non équipé de pyarrow doit pouvoir ouvrir."""
    loader.save_documents(documents)
    csv_path = loader.documents_path.with_suffix(".csv")
    assert csv_path.exists()
    head = pd.read_csv(csv_path).head(3)
    assert head["text"].str.len().min() > 100
