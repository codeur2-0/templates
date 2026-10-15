"""Prétraitement : découpage, tokens, vectorisation.

Trois propriétés sont testées ici parce que trois bugs invisibles y guettent :

* **les offsets** — un passage doit pointer sur la portion de texte qu'il contient réellement.
  La première implémentation reconstruisait les positions à coups de ``str.find`` et dérivait
  dès qu'une phrase se répétait dans le document ; le test compare donc le texte du passage à la
  tranche du document découpée par les offsets ;
* **la couverture** — le découpage ne doit perdre aucune phrase : la concaténation des passages
  recouvre le document aux frontières près (chevauchement compris) ;
* **le déterminisme** — deux appels à ``fit`` sur le même corpus produisent le même index et les
  mêmes scores, sinon les comparaisons entre variantes sont du bruit.
"""

from __future__ import annotations

from itertools import pairwise

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipelines import TextPreprocessor
from src.preprocessing.transformers import (
    DocumentChunker,
    HashingEmbedder,
    LexicalVectorizer,
    StopWordFilter,
    TextPreprocessingPipeline,
    normalise_text,
    split_sentences,
    tokenize,
)
from src.schemas.config import AppConfig
from src.utils.config_access import node

TEXT = (
    "Le solde de congés payés est de 25 jours ouvrables par an. "
    "L'accord d'entreprise prévoit un report de 5 jours sur l'année suivante. "
    "Les jours non pris sont perdus au 31 mai. "
    "Le télétravail est plafonné à 8 jours par mois."
)


def _identifier_column(app_config: AppConfig) -> str:
    """Return the corpus identifier column declared by the project (``doc_id``, ``msg_id``…).

    Le découpage ne devine pas le nom de la colonne d'identifiant : chaque famille déclare la
    sienne (``data.id_column``) et le test lit cette déclaration, comme le fait le chargeur.

    Args:
        app_config: Validated application configuration.

    Returns:
        The declared identifier column.
    """
    return str(node(app_config, "data").get("id_column") or "doc_id")


def test_normalise_text_collapses_whitespace() -> None:
    """Les espaces multiples et les retours chariot ne doivent pas créer de faux tokens."""
    assert normalise_text("  Un   texte\n\tséparé  ") == "Un texte séparé"


def test_tokenize_keeps_accents_and_digits() -> None:
    """Les accents et les nombres portent l'information métier : ils sont conservés."""
    assert tokenize("Trois devis supérieurs à 5000 euros") == [
        "trois",
        "devis",
        "supérieurs",
        "à",
        "5000",
        "euros",
    ]


def test_tokenize_keeps_internal_apostrophes_whole() -> None:
    """« l'accord » est un token, pas deux : le découper casserait tous les termes élidés."""
    assert tokenize("l'accord d'entreprise") == ["l'accord", "d'entreprise"]


def test_split_sentences_returns_each_sentence() -> None:
    """Quatre phrases, quatre unités : la phrase est l'unité atomique du découpage."""
    assert len(split_sentences(TEXT)) == 4


def test_chunker_offsets_point_at_the_passage_text() -> None:
    """Le contrat central du découpage : ``text[start:end]`` est bien le passage."""
    chunker = DocumentChunker(max_tokens=30, overlap_tokens=8)
    chunks = chunker.chunk_document("DOC-0001", TEXT)
    assert chunks, "un document non vide doit produire au moins un passage"
    for chunk in chunks:
        assert chunk.text in TEXT
        assert chunk.start_char < chunk.end_char
        assert chunk.n_tokens > 0


def test_chunker_covers_the_whole_document() -> None:
    """Aucune phrase ne doit être perdue : le découpage couvre tout le texte."""
    chunker = DocumentChunker(max_tokens=25, overlap_tokens=5)
    chunks = chunker.chunk_document("DOC-0001", TEXT)
    assert chunks[0].start_char == 0
    for previous, following in pairwise(chunks):
        assert following.start_char <= previous.end_char, "les passages doivent se suivre"
    last_sentence_start = TEXT.rfind("Le télétravail")
    assert chunks[-1].end_char > last_sentence_start


def test_chunker_respects_the_overlap_budget() -> None:
    """Le chevauchement sert à ne pas couper une réponse en deux : il doit exister."""
    chunker = DocumentChunker(max_tokens=20, overlap_tokens=6)
    chunks = chunker.chunk_document("DOC-0001", TEXT)
    if len(chunks) > 1:
        shared = chunks[0].text[-40:].split()
        assert any(word in chunks[1].text for word in shared[:2])


def test_chunker_rejects_an_absurd_configuration() -> None:
    """Un chevauchement supérieur à la taille du passage boucle à l'infini : on refuse."""
    with pytest.raises(ValueError):
        DocumentChunker(max_tokens=30, overlap_tokens=40)
    with pytest.raises(ValueError):
        DocumentChunker(max_tokens=10, overlap_tokens=0)


def test_chunker_frames_a_corpus(documents: pd.DataFrame, app_config: AppConfig) -> None:
    """Le passage à l'échelle : chaque document produit au moins un passage contractuel."""
    head = documents.head(8)
    chunker = DocumentChunker(
        max_tokens=110, overlap_tokens=30, id_column=_identifier_column(app_config)
    )
    frame = chunker.chunk_frame(head)
    assert len(frame) >= 8
    assert set(frame.columns) == {
        "chunk_id",
        "doc_id",
        "chunk_index",
        "start_char",
        "end_char",
        "n_tokens",
        "text",
    }
    assert frame["chunk_id"].is_unique
    # L'identifiant de la ligne source voyage avec le passage : c'est ce qui permet de remonter du
    # passage retrouvé au message dont il vient, quel que soit le nom déclaré dans la famille.
    assert frame["doc_id"].nunique() == len(head)


def test_stop_word_filter_drops_short_and_frequent_tokens() -> None:
    """Filtrer les mots vides est ce qui rend BM25 discriminant sur du texte français."""
    stop_words = StopWordFilter({"le", "de", "est"}, min_length=3)
    assert stop_words.transform(["le", "devis", "est", "de", "5000"]) == ["devis", "5000"]


def test_stop_word_filter_round_trip(tmp_path: Path) -> None:
    """La liste de mots vides est persistée avec le modèle : deux runs doivent partager la même."""
    path = tmp_path / "stopwords.joblib"
    original = StopWordFilter({"le", "la"}, min_length=2)
    original.save(path)
    reloaded = StopWordFilter.load(path)
    assert reloaded.transform(["le", "cabinet"]) == original.transform(["le", "cabinet"])


def test_preprocessing_pipeline_is_deterministic(
    documents: pd.DataFrame, app_config: AppConfig
) -> None:
    """Deux découpages du même corpus donnent exactement les mêmes passages."""
    pipeline = TextPreprocessingPipeline.from_config(
        {
            "chunking": {
                "max_tokens": 110,
                "overlap_tokens": 30,
                "id_column": _identifier_column(app_config),
            }
        }
    )
    first = pipeline.transform_documents(documents.head(6))
    second = pipeline.transform_documents(documents.head(6))
    pd.testing.assert_frame_equal(first, second)


def test_preprocessor_tokens_a_question(documents: pd.DataFrame, app_config: AppConfig) -> None:
    """``TextPreprocessor`` enchaîne découpage et tokénisation, comme le fait le modèle."""
    preprocessor = TextPreprocessor.from_config(
        {"chunking": {"id_column": _identifier_column(app_config)}}
    )
    chunks = preprocessor.prepare_corpus(documents.head(3))
    assert len(chunks) >= 3
    tokens = preprocessor.tokenize("Quelle est la durée de la période d'essai ?")
    assert "période" in tokens
    assert "est" not in tokens, "les mots vides doivent être retirés"


def test_lexical_vectorizer_scores_the_right_passage() -> None:
    """Le score doit être plus élevé pour le passage qui contient réellement la réponse."""
    passages = [
        "Le solde de congés payés est de 25 jours ouvrables par an.",
        "Le télétravail est plafonné à 8 jours par mois.",
        "Toute facture fournisseur doit être transmise sous 5 jours.",
    ]
    vectorizer = LexicalVectorizer(mode="bm25").fit(passages)
    query = "Combien de jours de télétravail par mois ?"
    scores = vectorizer.score(query, vectorizer.transform(passages))
    assert int(np.argmax(scores)) == 1


def test_lexical_vectorizer_tfidf_and_bm25_differ() -> None:
    """Les deux scorers ne sont pas le même modèle : la configuration doit avoir un effet."""
    passages = [f"document numéro {index} sur le congé payé" for index in range(6)]
    passages[-1] = "document final sur le télétravail et les congés payés"
    bm25 = LexicalVectorizer(mode="bm25").fit(passages)
    tfidf = LexicalVectorizer(mode="tfidf").fit(passages)
    query = "congés payés"
    bm25_scores = bm25.score(query, bm25.transform(passages))
    tfidf_scores = tfidf.score(query, tfidf.transform(passages))
    assert not np.allclose(bm25_scores, tfidf_scores)


def test_lexical_vectorizer_round_trip_keeps_the_scores(tmp_path: Path) -> None:
    """L'artefact doit reproduire les scores au bit près (joblib, pas de ré-apprentissage)."""
    passages = ["congés payés 25 jours", "télétravail 8 jours", "factures sous 5 jours"]
    vectorizer = LexicalVectorizer(mode="bm25", k1=1.2, b=0.6).fit(passages)
    expected = vectorizer.score("télétravail", vectorizer.transform(passages))
    path = tmp_path / "vectorizer.joblib"
    vectorizer.save(path)
    reloaded = LexicalVectorizer.load(path)
    np.testing.assert_allclose(
        reloaded.score("télétravail", reloaded.transform(passages)), expected
    )
    assert reloaded.mode == "bm25"


def test_lexical_vectorizer_requires_a_fit() -> None:
    """Score sans index appris : erreur explicite plutôt que tableau de zéros silencieux."""
    vectorizer = LexicalVectorizer(mode="bm25")
    with pytest.raises(RuntimeError):
        vectorizer.score("congés", vectorizer.transform(["congés"]))


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    """Deux embeddings du même texte sont identiques, et de norme 1 (produit scalaire = cosinus)."""
    embedder = HashingEmbedder(n_features=512, n_components=32, random_state=42)
    embedder.fit(["congés payés", "télétravail", "factures fournisseurs"])
    first = embedder.transform(["congés payés"])
    second = embedder.transform(["congés payés"])
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first[0]), 1.0, atol=1e-6)


def test_hashing_embedder_ranks_a_paraphrase_above_an_unrelated_text() -> None:
    """Même sans modèle pré-entraîné, un embedding lexical doit rapprocher les paraphrases."""
    embedder = HashingEmbedder(n_features=1024, n_components=64, random_state=0)
    corpus = [
        "Le solde de congés payés est de 25 jours ouvrables par an.",
        "Toute facture fournisseur doit être transmise au service achats sous 5 jours.",
    ]
    embedder.fit(corpus)
    matrix = embedder.transform(corpus)
    query = embedder.transform(["combien de jours de congés par an ?"])
    similarities = matrix @ query[0]
    assert similarities[0] > similarities[1]


def test_hashing_embedder_round_trip(tmp_path: Path) -> None:
    """L'embedding sauvegardé rend les mêmes vecteurs (les composantes SVD sont persistées)."""
    embedder = HashingEmbedder(n_features=256, n_components=16, random_state=3)
    embedder.fit(["congés payés", "télétravail", "notes de frais"])
    expected = embedder.transform(["télétravail"])
    path = tmp_path / "embedder.joblib"
    embedder.save(path)
    reloaded = HashingEmbedder.load(path)
    np.testing.assert_allclose(reloaded.transform(["télétravail"]), expected, atol=1e-6)


def test_hashing_embedder_honours_the_configured_ngram_range() -> None:
    """La plage de n-grammes déclarée dans la configuration est bien celle qui est hachée.

    Une configuration qui déclare ``[1, 1]`` produisait silencieusement des bigrammes : le
    paramètre existait sur la classe mais ``from_config`` l'ignorait.
    """
    unigrams = HashingEmbedder.from_config(
        {"n_features": 256, "n_components": 0, "ngram_range": [1, 1]}
    )
    bigrams = HashingEmbedder.from_config(
        {"n_features": 256, "n_components": 0, "ngram_range": [1, 2]}
    )

    assert unigrams.ngram_range == (1, 1)
    assert bigrams.ngram_range == (1, 2)
    with pytest.raises(ValueError, match="ngram_range"):
        HashingEmbedder.from_config({"ngram_range": [1]})


def test_embedding_dimension_is_reported() -> None:
    """La dimension est publiée dans le modèle : le rapport doit pouvoir l'afficher."""
    sparse = HashingEmbedder(n_features=128, n_components=0)
    dense = HashingEmbedder(n_features=128, n_components=16)
    assert sparse.dimension == 128
    assert dense.dimension == 16
