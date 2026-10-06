"""Text preprocessing: normalisation, sentence splitting, chunking, tokenisation, embedding.

Every transformer in this module is a *fitted or configured* object with a ``transform``
method, following the scikit-learn convention that the rest of the repository uses: a
vocabulary, a singular value decomposition or a stop-word list learned on the training corpus
is never re-learned on the validation or test split.

The module deliberately avoids downloading anything: tokenisation is a regex, stop words are a
short embedded list, and the embeddings are learned locally (hashing features + truncated SVD).
That keeps the examples runnable offline, on a CPU, in a few seconds — and it makes them
deterministic, which is what allows the reports to state exact figures.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from src.utils.config_access import as_mapping
from src.utils.io import load_pickle, save_pickle
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Word pattern: letters (accents included), digits, internal apostrophes and hyphens.
TOKEN_PATTERN = re.compile(
    "[0-9a-zA-Z\u00e0-\u00f6\u00f8-\u00ff\u00c0-\u00d6\u00d8-\u00df]+"
    "(?:['\u2019][0-9a-zA-Z\u00e0-\u00f6\u00f8-\u00ff]+)*"
)

#: Sentence boundaries: end of string, or ``.!?``/``;`` followed by a space and a capital letter.
SENTENCE_PATTERN = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")

#: French stop words used by the lexical scorers. Kept short on purpose: an aggressive list
#: hurts retrieval of rare terms ("congé", "achat") far more than it helps.
FRENCH_STOP_WORDS: frozenset[str] = frozenset(
    [
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "elles",
        "en",
        "et",
        "eux",
        "il",
        "ils",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "leurs",
        "lui",
        "ma",
        "mais",
        "me",
        "meme",
        "mes",
        "moi",
        "mon",
        "ne",
        "nos",
        "notre",
        "nous",
        "on",
        "ou",
        "par",
        "pas",
        "pour",
        "qu",
        "que",
        "qui",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "tu",
        "un",
        "une",
        "vos",
        "votre",
        "vous",
        "c",
        "d",
        "j",
        "l",
        "a",
        "m",
        "n",
        "s",
        "t",
        "y",
        "ete",
        "etee",
        "etees",
        "etes",
        "etant",
        "suis",
        "es",
        "est",
        "sommes",
        "etes",
        "sont",
        "serai",
        "seras",
        "sera",
        "serons",
        "serez",
        "seront",
        "serais",
        "serait",
        "the",
        "of",
        "and",
        "to",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    ]
)


def normalise_text(text: str) -> str:
    """Normalise a text without destroying its information.

    The transformation is intentionally conservative: Unicode NFKC, curly quotes and dashes
    folded to ASCII, whitespace collapsed. Apostrophes and accents are **kept** because French
    retrieval depends on them (``congé`` vs ``conge`` are different tokens), and lower-casing is
    *not* applied here — it belongs to the tokeniser, so a title can still be displayed as-is.

    Args:
        text: Raw text.

    Returns:
        The normalised text.
    """
    folded = unicodedata.normalize("NFKC", str(text))
    for source, target in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        folded = folded.replace(source, target)
    for source, target in (("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        folded = folded.replace(source, target)
    return re.sub(r"\s+", " ", folded).strip()


def tokenize(text: str) -> list[str]:
    """Split a text into lower-cased tokens.

    Args:
        text: Text to tokenise.

    Returns:
        The token list, in order of appearance.
    """
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(normalise_text(text))]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences.

    The rule is deliberately simple (punctuation followed by a capital letter): on a synthetic
    corpus, annotated with the same rule, a parser-grade splitter would buy nothing and would
    add a dependency.

    Args:
        text: Text to split.

    Returns:
        The sentences, stripped, empty ones removed.
    """
    parts = SENTENCE_PATTERN.split(normalise_text(text))
    return [part.strip() for part in parts if part.strip()]


@dataclass
class Chunk:
    """One indexed passage.

    Attributes:
        chunk_id: Stable identifier (``DOC-0007-C02``).
        doc_id: Document the passage comes from.
        chunk_index: Position of the passage inside its document.
        text: Passage text.
        start_char: Offset of the first character inside the document.
        end_char: Offset just after the last character inside the document.
        n_tokens: Number of tokens.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    n_tokens: int

    def to_row(self) -> dict[str, Any]:
        """Return the passage as a flat mapping (one row of the chunks table)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "n_tokens": self.n_tokens,
            "text": self.text,
        }


@dataclass
class DocumentChunker:
    """Split documents into overlapping passages on sentence boundaries.

    Attributes:
        max_tokens: Maximum number of tokens per passage.
        overlap_tokens: Tokens shared by two consecutive passages (answers that straddle a
            boundary would otherwise be invisible to the retriever).
    """

    max_tokens: int = 120
    overlap_tokens: int = 30

    def __post_init__(self) -> None:
        """Reject an impossible configuration early."""
        if self.max_tokens < 20:
            msg = f"max_tokens must be >= 20, got {self.max_tokens}"
            raise ValueError(msg)
        if not 0 <= self.overlap_tokens < self.max_tokens:
            msg = (
                f"overlap_tokens must satisfy 0 <= overlap < max_tokens, got "
                f"{self.overlap_tokens} / {self.max_tokens}"
            )
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Any) -> DocumentChunker:
        """Build a chunker from a configuration node.

        Args:
            config: Mapping (or OmegaConf node) holding ``max_tokens`` and ``overlap_tokens``.

        Returns:
            The configured chunker.
        """
        settings = as_mapping(config)
        return cls(
            max_tokens=int(settings.get("max_tokens", 120)),
            overlap_tokens=int(settings.get("overlap_tokens", 30)),
        )

    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        """Split one document.

        Sentences are packed greedily until ``max_tokens``; the last ``overlap_tokens`` tokens
        are replayed at the start of the next passage. Character offsets are tracked so that a
        retrieved passage can always be mapped back to its exact position in the source.

        Args:
            doc_id: Identifier of the document.
            text: Document text.

        Returns:
            The passages, in reading order.
        """
        normalised = normalise_text(text)
        sentences = split_sentences(normalised) or [normalised]
        located = self._locate(normalised, sentences)
        chunks: list[Chunk] = []
        pending: list[tuple[str, int]] = []
        pending_tokens = 0
        for item in located:
            pending.append(item)
            pending_tokens += len(tokenize(item[0]))
            if pending_tokens >= self.max_tokens:
                chunks.append(self._build(doc_id, len(chunks), pending))
                pending = self._overlap_tail(pending)
                pending_tokens = sum(len(tokenize(sentence)) for sentence, _ in pending)
        if pending:
            tail_text = normalise_text(" ".join(sentence for sentence, _ in pending))
            if not chunks or len(tokenize(tail_text)) > 10:
                chunks.append(self._build(doc_id, len(chunks), pending))
        return chunks

    def _locate(self, text: str, sentences: Sequence[str]) -> list[tuple[str, int]]:
        """Return ``(sentence, character offset)`` pairs, in reading order."""
        located: list[tuple[str, int]] = []
        cursor = 0
        for sentence in sentences:
            position = text.find(sentence, cursor)
            position = cursor if position < 0 else position
            located.append((sentence, position))
            cursor = position + len(sentence)
        return located

    def chunk_frame(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Chunk a whole corpus frame.

        Args:
            documents: Corpus with ``doc_id`` and ``text`` columns.

        Returns:
            The chunks as a flat frame, ready for :func:`src.data.schemas.validate_chunks`.
        """
        rows: list[dict[str, Any]] = []
        for record in documents.itertuples(index=False):
            for chunk in self.chunk_document(str(record.doc_id), str(record.text)):
                rows.append(chunk.to_row())
        frame = pd.DataFrame(rows)
        logger.info(
            "Chunked {} documents into {} passages (max_tokens={}, overlap={})",
            len(documents),
            len(frame),
            self.max_tokens,
            self.overlap_tokens,
        )
        return frame

    def _overlap_tail(self, sentences: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        """Return the sentences replayed at the start of the next passage (the overlap)."""
        if self.overlap_tokens <= 0:
            return []
        kept: list[tuple[str, int]] = []
        budget = 0
        for item in reversed(sentences):
            sentence_tokens = len(tokenize(item[0]))
            if kept and budget + sentence_tokens > self.overlap_tokens:
                break
            kept.insert(0, item)
            budget += sentence_tokens
        return kept

    def _build(self, doc_id: str, index: int, sentences: Sequence[tuple[str, int]]) -> Chunk:
        """Assemble one passage from its located sentences."""
        text = normalise_text(" ".join(sentence for sentence, _ in sentences))
        start_char = sentences[0][1]
        end_char = sentences[-1][1] + len(sentences[-1][0])
        return Chunk(
            chunk_id=f"{doc_id}-C{index:02d}",
            doc_id=doc_id,
            chunk_index=index,
            text=text,
            start_char=int(start_char),
            end_char=int(max(end_char, start_char + 1)),
            n_tokens=len(tokenize(text)),
        )


class StopWordFilter:
    """Remove stop words from a token stream.

    Attributes:
        stop_words: Words that are dropped.
        min_length: Minimum token length kept (single letters carry no retrieval signal).
    """

    def __init__(
        self, stop_words: Iterable[str] = FRENCH_STOP_WORDS, *, min_length: int = 2
    ) -> None:
        """Configure the filter.

        Args:
            stop_words: Words to drop.
            min_length: Minimum length of a kept token.
        """
        self.stop_words = frozenset(stop_words)
        self.min_length = int(min_length)

    def transform(self, tokens: Sequence[str]) -> list[str]:
        """Filter a token sequence.

        Args:
            tokens: Tokens to filter.

        Returns:
            The kept tokens.
        """
        return [
            token
            for token in tokens
            if len(token) >= self.min_length and token not in self.stop_words
        ]

    def save(self, path: str | Path) -> Path:
        """Persist the filter (stop words + minimum length).

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return save_pickle(
            {"stop_words": sorted(self.stop_words), "min_length": self.min_length}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> StopWordFilter:
        """Reload a persisted filter.

        Args:
            path: Artefact written by :meth:`save`.

        Returns:
            The restored filter.
        """
        payload = load_pickle(path)
        return cls(payload["stop_words"], min_length=int(payload["min_length"]))


@dataclass
class TextPreprocessingPipeline:
    """The full text preparation pipeline: normalise -> chunk -> tokenise -> filter.

    The pipeline is *stateless* on purpose: chunking and tokenisation depend only on their
    configuration, so there is nothing to fit and therefore no leak possible between splits.
    The learned objects of the project are the vectoriser and the embedding model, which live
    in the model itself (see ``src/models``), where their train-only fitting is explicit.

    Attributes:
        chunker: Passage builder.
        stop_words: Token filter.
    """

    chunker: DocumentChunker = field(default_factory=DocumentChunker)
    stop_words: StopWordFilter = field(default_factory=StopWordFilter)

    @classmethod
    def from_config(cls, config: Any) -> TextPreprocessingPipeline:
        """Build the pipeline from a Hydra configuration node.

        Args:
            config: ``preprocessing`` node of the configuration.

        Returns:
            The configured pipeline.
        """
        chunking = as_mapping(config).get("chunking", {})
        return cls(chunker=DocumentChunker.from_config(chunking))

    def fit(self, documents: pd.DataFrame) -> TextPreprocessingPipeline:
        """No-op fit, kept for API symmetry with scikit-learn pipelines.

        Args:
            documents: Corpus (unused, the pipeline has no learned parameter).

        Returns:
            ``self``.
        """
        logger.debug(
            "TextPreprocessingPipeline has no fitted parameter ({} documents seen)", len(documents)
        )
        return self

    def transform_documents(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Chunk a corpus into passages.

        Args:
            documents: Corpus with ``doc_id`` and ``text``.

        Returns:
            The passage table.
        """
        return self.chunker.chunk_frame(documents)

    def transform(self, texts: Sequence[str]) -> list[list[str]]:
        """Tokenise texts, stop words removed.

        Args:
            texts: Texts to tokenise.

        Returns:
            One token list per text.
        """
        return [self.stop_words.transform(tokenize(text)) for text in texts]

    def save(self, path: str | Path) -> Path:
        """Persist the pipeline configuration.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return save_pickle(
            {
                "max_tokens": self.chunker.max_tokens,
                "overlap_tokens": self.chunker.overlap_tokens,
                "stop_words": sorted(self.stop_words.stop_words),
                "min_length": self.stop_words.min_length,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TextPreprocessingPipeline:
        """Reload a persisted pipeline.

        Args:
            path: Artefact written by :meth:`save`.

        Returns:
            The restored pipeline.
        """
        payload = load_pickle(path)
        return cls(
            chunker=DocumentChunker(
                max_tokens=int(payload["max_tokens"]), overlap_tokens=int(payload["overlap_tokens"])
            ),
            stop_words=StopWordFilter(payload["stop_words"], min_length=int(payload["min_length"])),
        )


@dataclass
class HashingEmbedder:
    """Deterministic, offline text embedder.

    The design is a compromise that the project states explicitly rather than hides:

    * a **hashing vectoriser** (word unigrams + bigrams) projects a text into a fixed space
      without learning a vocabulary — so it can embed a document never seen during fit;
    * a **truncated SVD** learned on the *training* corpus compresses that sparse space into a
      dense, low-dimensional one that captures co-occurrence (the "distributional" signal);
    * vectors are L2-normalised, so the cosine similarity is a dot product.

    This is not a transformer: it cannot resolve long-range paraphrases. It is the honest
    lexical-ish baseline that a RAG project must beat before claiming a neural embedding is
    worth its cost, and it runs in milliseconds on a CPU with no download.

    Attributes:
        n_features: Width of the hashing space.
        n_components: Dimension of the dense embedding (``0`` keeps the sparse space).
        ngram_range: Range of n-grams hashed.
        random_state: Seed of the SVD (``randomized`` solver).
        is_fitted: Whether the SVD has been learned.
    """

    n_features: int = 4096
    n_components: int = 192
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 42
    is_fitted: bool = False
    #: Internal estimators; not constructor arguments (derived from the fields above).
    _hasher: HashingVectorizer = field(init=False, repr=False)
    _svd: TruncatedSVD | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Instantiate the internal estimators."""
        self._hasher = HashingVectorizer(
            n_features=self.n_features,
            ngram_range=self.ngram_range,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self._svd: TruncatedSVD | None = None
        if self.n_components > 0:
            self._svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)

    @classmethod
    def from_config(cls, config: Any) -> HashingEmbedder:
        """Build the embedder from a configuration node.

        Args:
            config: ``model.params.embedding`` node (or any mapping with the same keys).

        Returns:
            The configured embedder.
        """
        settings = as_mapping(config)
        bounds = tuple(int(value) for value in settings.get("ngram_range", (1, 2)))
        if len(bounds) != 2:
            msg = f"ngram_range must hold exactly two bounds, got {bounds!r}"
            raise ValueError(msg)
        return cls(
            n_features=int(settings.get("n_features", 4096)),
            n_components=int(settings.get("n_components", 192)),
            ngram_range=(bounds[0], bounds[1]),
            random_state=int(settings.get("random_state", 42)),
        )

    @property
    def dimension(self) -> int:
        """Dimension of the produced vectors."""
        return self.n_components if self.n_components > 0 else self.n_features

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        """Learn the SVD on a corpus.

        Args:
            texts: Training texts (documents or passages).

        Returns:
            ``self``.
        """
        matrix = self._hasher.transform(list(texts))
        if self._svd is not None:
            n_components = min(self.n_components, max(2, min(matrix.shape) - 1))
            if n_components != self.n_components:
                logger.warning(
                    "Truncated SVD reduced from {} to {} components (corpus too small)",
                    self.n_components,
                    n_components,
                )
                self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            self._svd.fit(matrix)
        self.is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            A ``(n_texts, dimension)`` float32 array, L2-normalised.

        Raises:
            RuntimeError: When the embedder has not been fitted.
        """
        if not self.is_fitted:
            msg = (
                "HashingEmbedder must be fitted before transform (call fit on the training corpus)"
            )
            raise RuntimeError(msg)
        matrix = self._hasher.transform(list(texts))
        dense = self._svd.transform(matrix) if self._svd is not None else matrix.toarray()
        return normalize(np.asarray(dense, dtype="float32"), norm="l2")

    def projection_fidelity(self, texts: Sequence[str]) -> float:
        """Measure how much of the raw hashed geometry the projection keeps.

        A projection is a *compression*: it maps the sparse hashed space onto a much smaller one,
        and a similarity computed there is only useful if it still orders the passages the way the
        raw space does. The metric compares, on consecutive pairs of the corpus, the cosine of the
        raw hashed vectors with the cosine of their projections, and reports the agreement
        ``1 - mean(|difference|) / 2``. It is ``1.0`` by construction when no projection is used,
        and it falls when the compression starts confusing passages the hashed space separated.

        Args:
            texts: Texts used for the measurement (the training passages, typically).

        Returns:
            The agreement between the two similarity structures, in ``[0, 1]``.
        """
        raw = self._hasher.transform(list(texts))
        projected = self.transform(texts)
        if projected.shape[1] == raw.shape[1] or len(texts) < 2:
            return 1.0
        rows = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        # Consecutive passages of the corpus often belong to the same document: the pairs are
        # therefore taken with a stride of a third of the corpus, which crosses the documents and
        # the topics, and capped so the measurement stays cheap on a large index.
        stride = max(len(rows) // 3, 1)
        left = np.arange(0, max(len(rows) - stride, 1), dtype=int)[:512]
        right = left + stride
        raw_similarity = np.asarray(raw[left].multiply(raw[right]).sum(axis=1)).ravel()
        dense_similarity = np.einsum("ij,ij->i", rows[left], rows[right])
        deviation = float(np.mean(np.abs(raw_similarity - dense_similarity))) / 2.0
        return float(np.clip(1.0 - deviation, 0.0, 1.0))

    def save(self, path: str | Path) -> Path:
        """Persist the embedder (hasher configuration + learned SVD).

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        payload = {
            "n_features": self.n_features,
            "n_components": 0 if self._svd is None else int(self._svd.n_components),
            "ngram_range": list(self.ngram_range),
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "svd_components": None if self._svd is None else self._svd.components_.tolist(),
        }
        return save_pickle(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> HashingEmbedder:
        """Reload a persisted embedder.

        Args:
            path: Artefact written by :meth:`save`.

        Returns:
            The restored embedder, ready to ``transform``.
        """
        payload = load_pickle(path)
        embedder = cls(
            n_features=int(payload["n_features"]),
            n_components=int(payload["n_components"]),
            ngram_range=tuple(payload["ngram_range"]),
            random_state=int(payload["random_state"]),
        )
        if payload.get("svd_components") is not None and embedder._svd is not None:
            embedder._svd.components_ = np.asarray(payload["svd_components"], dtype="float64")
        embedder.is_fitted = bool(payload["is_fitted"])
        return embedder


@dataclass
class LexicalVectorizer:
    """TF-IDF / BM25 lexical index, learned on the training passages.

    BM25 is implemented on top of the same count matrix as TF-IDF rather than with a
    third-party library: it makes the difference between the two scorers readable in the code
    (saturation of the term frequency, length normalisation) instead of hidden behind a
    dependency.

    Attributes:
        mode: ``tfidf`` (cosine) or ``bm25`` (Okapi).
        ngram_range: Range of n-grams.
        min_df: Minimum document frequency of a kept term.
        max_features: Vocabulary cap (``None`` keeps everything).
        sublinear_tf: Apply ``1 + log(tf)`` (TF-IDF only).
        k1: BM25 term-frequency saturation.
        b: BM25 length normalisation.
        is_fitted: Whether the vocabulary has been learned.
    """

    mode: str = "bm25"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1
    max_features: int | None = None
    sublinear_tf: bool = True
    k1: float = 1.5
    b: float = 0.75
    is_fitted: bool = False
    #: Internal state; not constructor arguments.
    _vectorizer: TfidfVectorizer = field(init=False, repr=False)
    _matrix: Any = field(init=False, repr=False, default=None)
    _idf: np.ndarray | None = field(init=False, repr=False, default=None)
    _length_norm: float | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Validate the configuration and instantiate the vectoriser."""
        if self.mode not in {"tfidf", "bm25"}:
            msg = f"mode must be 'tfidf' or 'bm25', got '{self.mode}'"
            raise ValueError(msg)
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            norm="l2" if self.mode == "tfidf" else None,
            use_idf=self.mode == "tfidf",
        )
        self._matrix: Any = None
        self._idf: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Any) -> LexicalVectorizer:
        """Build the vectoriser from a configuration node.

        Args:
            config: ``model.params.lexical`` node (or any mapping with the same keys).

        Returns:
            The configured vectoriser.
        """
        settings = as_mapping(config)
        return cls(
            mode=str(settings.get("mode", "bm25")),
            ngram_range=tuple(settings.get("ngram_range", (1, 1))),
            min_df=int(settings.get("min_df", 1)),
            max_features=settings.get("max_features"),
            sublinear_tf=bool(settings.get("sublinear_tf", True)),
            k1=float(settings.get("k1", 1.5)),
            b=float(settings.get("b", 0.75)),
        )

    @property
    def vocabulary_size(self) -> int:
        """Number of terms in the learned vocabulary."""
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: Sequence[str]) -> LexicalVectorizer:
        """Learn the vocabulary (and the IDF weights in BM25 mode).

        Args:
            texts: Training passages.

        Returns:
            ``self``.
        """
        matrix = self._vectorizer.fit_transform(list(texts))
        self._matrix = matrix.astype("float32")
        if self.mode == "bm25":
            document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel().astype("float64")
            self._idf = np.log(
                1.0 + (matrix.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            self._length_norm = float(np.mean(np.asarray(matrix.sum(axis=1)).ravel())) or 1.0
        else:
            self._idf = np.asarray(self._vectorizer.idf_, dtype="float64")
        self.is_fitted = True
        logger.info(
            "Lexical index fitted | mode={} | vocabulary={} | passages={}",
            self.mode,
            self.vocabulary_size,
            matrix.shape[0],
        )
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Score passages against the learned vocabulary.

        Args:
            texts: Passages to score.

        Returns:
            A sparse-friendly dense matrix of lexical weights (rows = passages).

        Raises:
            RuntimeError: When the vocabulary has not been learned.
        """
        if not self.is_fitted:
            msg = (
                "LexicalVectorizer must be fitted before transform "
                "(call fit on the training corpus)"
            )
            raise RuntimeError(msg)
        matrix = self._vectorizer.transform(list(texts))
        if self.mode == "tfidf":
            return np.asarray(matrix.todense(), dtype="float32")
        counts = np.asarray(matrix.todense(), dtype="float32")
        assert self._idf is not None
        lengths = counts.sum(axis=1, keepdims=True)
        denominator = counts + self.k1 * (
            1.0 - self.b + self.b * lengths / max(self._length_norm or 1e-9, 1e-9)
        )
        weighted = counts * (self.k1 + 1.0) / np.where(denominator == 0.0, 1.0, denominator)
        return weighted * self._idf.reshape(1, -1)

    def transform_query(self, text: str) -> np.ndarray:
        """Score a single query in the same space as the passages.

        Args:
            text: Query text.

        Returns:
            A ``(n_terms,)`` weight vector.
        """
        return self.transform([text])[0]

    def score(self, query: str, matrix: np.ndarray) -> np.ndarray:
        """Score every passage of an already transformed matrix against a query.

        Args:
            query: Query text.
            matrix: Matrix produced by :meth:`transform` on the passage corpus.

        Returns:
            One similarity score per passage (cosine for TF-IDF, dot product for BM25).
        """
        weights = self.transform_query(query)
        scores = matrix @ weights
        if self.mode == "tfidf":
            norm = np.linalg.norm(weights)
            if norm > 0:
                scores = scores / (norm * np.maximum(np.linalg.norm(matrix, axis=1), 1e-9))
        return np.asarray(scores, dtype="float32")

    def save(self, path: str | Path) -> Path:
        """Persist the vectoriser state (vocabulary, IDF, corpus statistics).

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        payload = {
            "mode": self.mode,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "k1": self.k1,
            "b": self.b,
            "is_fitted": self.is_fitted,
            "vocabulary": self._vectorizer.vocabulary_,
            "idf": None if self._idf is None else self._idf.tolist(),
            "length_norm": getattr(self, "_length_norm", None),
            # L'estimateur est persisté tel quel : réinjecter une simple liste de vocabulaire
            # laisse ``TfidfVectorizer`` sans son transformeur interne (``_tfidf``), et le
            # premier ``transform`` échoue alors que l'artefact paraît valide.
            "estimator": self._vectorizer,
        }
        return save_pickle(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> LexicalVectorizer:
        """Reload a persisted vectoriser.

        Args:
            path: Artefact written by :meth:`save`.

        Returns:
            The restored vectoriser.
        """
        payload = load_pickle(path)
        vectorizer = cls(
            mode=str(payload["mode"]),
            ngram_range=tuple(payload["ngram_range"]),
            min_df=int(payload["min_df"]),
            max_features=payload["max_features"],
            sublinear_tf=bool(payload["sublinear_tf"]),
            k1=float(payload["k1"]),
            b=float(payload["b"]),
        )
        if "estimator" in payload:
            vectorizer._vectorizer = payload["estimator"]
        else:  # artefact antérieur : on reconstruit un état minimal, sans transformeur interne
            vectorizer._vectorizer.vocabulary_ = dict(payload["vocabulary"])
            vectorizer._vectorizer.fixed_vocabulary_ = True
        vectorizer._idf = (
            None if payload["idf"] is None else np.asarray(payload["idf"], dtype="float64")
        )
        if payload.get("length_norm") is not None:
            vectorizer._length_norm = float(payload["length_norm"])
        vectorizer.is_fitted = bool(payload["is_fitted"])
        return vectorizer


__all__ = [
    "Chunk",
    "DocumentChunker",
    "FRENCH_STOP_WORDS",
    "HashingEmbedder",
    "LexicalVectorizer",
    "StopWordFilter",
    "TextPreprocessingPipeline",
    "TOKEN_PATTERN",
    "normalise_text",
    "split_sentences",
    "tokenize",
]
