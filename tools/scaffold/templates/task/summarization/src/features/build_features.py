"""Représentation d'un document comme **un graphe de phrases**.

Un résumé extractif ne choisit pas des mots : il choisit des phrases. La représentation utile n'est
donc pas « un texte → un vecteur », mais « un document → une matrice de similarité entre ses
phrases », et c'est cette matrice que la baseline TextRank parcourt. Construire cette représentation
une seule fois, dans un objet testé, évite le défaut classique du résumé extractif : un notebook qui
recalcule sa propre tokenisation, une baseline qui en utilise une autre, et deux scores qui ne
parlent plus du même texte.

Trois décisions sont explicites ici :

* la pondération TF-IDF est calculée **document par document**, pas sur le corpus : un terme
  rare dans le corpus entier n'est pas plus informatif pour *ce* document, et une IDF apprise
  sur le train ferait fuiter la fréquence des documents de test dans la représentation ;
* la similarité est un cosinus sur des vecteurs normalisés, avec une diagonale mise à zéro : une
  phrase n'est pas son propre voisin, sinon la marche aléatoire de TextRank converge vers la phrase
  la plus longue ;
* la position est conservée comme feature, jamais comme score : dans un compte-rendu d'intervention,
  la première phrase porte le contexte et la dernière porte la suite à donner, mais un résumé qui ne
  prendrait que les extrémités serait un résumé de forme, et la baseline le montre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from src.preprocessing.pipelines import TextPreprocessor
from src.preprocessing.transformers import split_sentences
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Smoothing constant of the IDF, identical to the one scikit-learn uses: it keeps every term
#: positive and finite, so a document made of a single repeated sentence has no zero-division case.
IDF_SMOOTHING = 1.0


@dataclass(slots=True)
class SentenceFeatures:
    """Everything the extractive model needs to know about **one** document.

    Attributes:
        doc_id: Document identifier (``CR-0001``).
        sentences: Sentences, in reading order, split with the modality's rule.
        tokens: Token list of each sentence (filtered by the corpus stop-word list).
        matrix: TF-IDF matrix, one row per sentence (``n_sentences`` x vocabulary).
        similarity: Cosine similarity between sentences, diagonal set to zero.
        centrality: Normalised degree of each sentence in the similarity graph.
        position: Relative position of the sentence in the document (0.0 first, 1.0 last).
        length: Number of tokens of each sentence.
        vocabulary: Term to column index mapping of the document.
        idf: IDF of each column of :attr:`matrix`.
    """

    doc_id: str
    sentences: list[str]
    tokens: list[list[str]]
    matrix: np.ndarray
    similarity: np.ndarray
    centrality: np.ndarray
    position: np.ndarray
    length: np.ndarray
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="float64"))

    @property
    def n_sentences(self) -> int:
        """Number of sentences of the document."""
        return len(self.sentences)

    @property
    def n_tokens(self) -> int:
        """Number of tokens of the whole document."""
        return int(self.length.sum())

    def most_central_sentences(self, k: int = 3) -> list[tuple[int, float]]:
        """Return the ``k`` most central sentences, by decreasing centrality.

        Args:
            k: Number of sentences to return.

        Returns:
            ``(index, centrality)`` pairs, ties broken by sentence order (deterministic).
        """
        order = sorted(range(self.n_sentences), key=lambda index: (-self.centrality[index], index))
        return [(index, float(self.centrality[index])) for index in order[:k]]

    def top_terms(self, sentence_index: int, k: int = 5) -> list[tuple[str, float]]:
        """Return the terms a sentence weighs the most, by decreasing TF-IDF.

        Args:
            sentence_index: Row of :attr:`matrix`.
            k: Number of terms to return.

        Returns:
            ``(term, weight)`` pairs, ties broken by term (deterministic).
        """
        if not self.vocabulary:
            return []
        row = self.matrix[sentence_index]
        inverted = {index: term for term, index in self.vocabulary.items()}
        ranked = sorted(
            ((inverted[column], float(weight)) for column, weight in enumerate(row) if weight > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]

    def summary(self) -> str:
        """One-line description of the document representation (used in logs)."""
        density = float((self.similarity > 0).mean())
        return (
            f"{self.doc_id}: {self.n_sentences} phrases, {self.n_tokens} tokens, "
            f"vocabulaire {len(self.vocabulary)}, densité du graphe {density:.2f}"
        )


class SentenceFeatureBuilder:
    """Turn documents into :class:`SentenceFeatures` objects, deterministically.

    Attributes:
        preprocessor: The corpus tokeniser, shared with the retrieval families so that a stop word
            removed for a search is also removed from a sentence graph.
        max_sentences: Safety bound on the number of sentences used per document.
    """

    def __init__(
        self, preprocessor: TextPreprocessor | None = None, *, max_sentences: int = 40
    ) -> None:
        """Configure the builder.

        Args:
            preprocessor: Configured tokeniser (defaults to the template defaults).
            max_sentences: Maximum number of sentences kept per document.
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.max_sentences = int(max_sentences)

    @classmethod
    def from_config(cls, config: Any) -> SentenceFeatureBuilder:
        """Build the builder from the ``preprocessing`` node of the configuration.

        Args:
            config: ``preprocessing`` node.

        Returns:
            The configured builder.
        """
        return cls(TextPreprocessor.from_config(config))

    # ------------------------------------------------------------------ construction -------
    def build(self, doc_id: str, text: str) -> SentenceFeatures:
        """Build the sentence graph of one document.

        Args:
            doc_id: Document identifier.
            text: Document text.

        Returns:
            The sentence representation of the document (possibly with a single sentence).
        """
        sentences = split_sentences(text)[: self.max_sentences]
        tokens = self.preprocessor.tokenize_many(sentences)
        matrix, vocabulary, idf = _tfidf(tokens)
        similarity = _cosine_similarity(matrix)
        centrality = _centrality(similarity)
        position = _relative_position(len(sentences))
        length = np.asarray([len(row) for row in tokens], dtype="int64")
        return SentenceFeatures(
            doc_id=str(doc_id),
            sentences=sentences,
            tokens=tokens,
            matrix=matrix,
            similarity=similarity,
            centrality=centrality,
            position=position,
            length=length,
            vocabulary=vocabulary,
            idf=idf,
        )

    def build_frame(
        self, documents: pd.DataFrame, *, text_column: str = "text"
    ) -> dict[str, SentenceFeatures]:
        """Build the sentence graph of every document of a frame.

        Args:
            documents: Corpus with a ``doc_id`` column and the text column.
            text_column: Name of the text column.

        Returns:
            Mapping ``doc_id -> SentenceFeatures``.
        """
        features = {
            str(row.doc_id): self.build(str(row.doc_id), str(row[text_column]))
            for row in documents.itertuples(index=False)
        }
        if features:
            mean_sentences = float(np.mean([item.n_sentences for item in features.values()]))
            logger.info(
                "Sentence graphs built for {} document(s), {:.1f} phrases en moyenne",
                len(features),
                mean_sentences,
            )
        return features

    # ------------------------------------------------------------------ profils ------------
    @staticmethod
    def profile(documents: pd.DataFrame, references: pd.DataFrame) -> dict[str, float]:
        """Summarise the length structure of a corpus (used by the report and the notebooks).

        Args:
            documents: Document table (``n_sentences``, ``n_tokens``).
            references: Reference summary table (``n_tokens``,
                ``compression``, ``n_salient_facts``).

        Returns:
            Finite metrics describing the compression the task actually asks for.
        """
        return {
            "n_documents": float(len(documents)),
            "sentences_mean": round(float(documents["n_sentences"].mean()), 2),
            "sentences_max": float(documents["n_sentences"].max()),
            "document_tokens_mean": round(float(documents["n_tokens"].mean()), 2),
            "reference_tokens_mean": round(float(references["n_tokens"].mean()), 2),
            "reference_sentences_mean": round(float(references["n_sentences"].mean()), 2),
            "compression_mean": round(float(references["compression"].mean()), 4),
            "salient_facts_mean": round(float(references["n_salient_facts"].mean()), 2),
            # Ce qu'un résumé qui recopierait le document entier obtiendrait : le plafond du ROUGE
            # par recopie, publié avec le corpus pour que le score du modèle ait une échelle.
            "copy_compression": 1.0,
        }


def _tfidf(tokens: list[list[str]]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Compute the document-level TF-IDF matrix of a tokenised document.

    Args:
        tokens: Token list of each sentence.

    Returns:
        ``(matrix, vocabulary, idf)``: the ``n_sentences`` x ``vocabulary`` matrix, the term to
        column mapping and the IDF vector. A document with no usable token yields an empty matrix.
    """
    vocabulary: dict[str, int] = {}
    for row in tokens:
        for token in row:
            vocabulary.setdefault(token, len(vocabulary))
    n_sentences = len(tokens)
    if not vocabulary or n_sentences == 0:
        return np.zeros((n_sentences, 0), dtype="float64"), {}, np.zeros(0, dtype="float64")

    counts = np.zeros((n_sentences, len(vocabulary)), dtype="float64")
    for index, row in enumerate(tokens):
        for token in row:
            counts[index, vocabulary[token]] += 1.0
    lengths = counts.sum(axis=1, keepdims=True)
    term_frequency = np.divide(counts, np.maximum(lengths, 1.0))
    document_frequency = (counts > 0).sum(axis=0).astype("float64")
    idf = np.log((1.0 + n_sentences) / (IDF_SMOOTHING + document_frequency)) + 1.0
    return term_frequency * idf, vocabulary, idf


def _cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of a matrix, diagonal zeroed.

    Args:
        matrix: ``n_sentences`` x ``vocabulary`` matrix.

    Returns:
        The ``n_sentences`` x ``n_sentences`` similarity matrix.
    """
    n_sentences = matrix.shape[0]
    if n_sentences == 0:
        return np.zeros((0, 0), dtype="float64")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = np.divide(matrix, np.maximum(norms, 1e-12))
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, 0.0)
    # Une phrase sans terme commun avec une autre n'est pas « un peu » liée : le cosinus est
    # arrondi à zéro pour que la marche aléatoire ne reçoive pas de bruit numérique.
    similarity[np.abs(similarity) < 1e-9] = 0.0
    return similarity


def _centrality(similarity: np.ndarray) -> np.ndarray:
    """Degree centrality of each sentence, normalised to sum to one.

    Args:
        similarity: Sentence similarity matrix.

    Returns:
        The centrality vector (uniform when the graph has no edge at all).
    """
    n_sentences = similarity.shape[0]
    if n_sentences == 0:
        return np.zeros(0, dtype="float64")
    degree = similarity.sum(axis=1)
    total = float(degree.sum())
    if total <= 0.0:
        return np.full(n_sentences, 1.0 / n_sentences, dtype="float64")
    return degree / total


def _relative_position(n_sentences: int) -> np.ndarray:
    """Relative position of each sentence in a document of ``n_sentences`` sentences.

    Args:
        n_sentences: Number of sentences.

    Returns:
        Positions from 0.0 (first) to 1.0 (last); a single sentence sits at 0.5 so that it is
        neither the beginning nor the end of the document.
    """
    if n_sentences <= 0:
        return np.zeros(0, dtype="float64")
    if n_sentences == 1:
        return np.asarray([0.5], dtype="float64")
    return np.linspace(0.0, 1.0, n_sentences)


__all__ = ["IDF_SMOOTHING", "SentenceFeatureBuilder", "SentenceFeatures"]
