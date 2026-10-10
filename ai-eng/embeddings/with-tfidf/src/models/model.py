"""Lexical retrieval model: TF-IDF or BM25 over the passages of the corpus.

This model is the reference every other stack of the family must beat. Its honesty comes from
what it does *not* do:

* it never learns a synonym — a question whose wording shares nothing with the passage will
  simply not match, no matter how good the question is (that is the whole point of the
  ``paraphrase`` segment of the evaluation);
* it cannot resolve anaphora or negation — "le délai n'est pas de 30 jours" scores exactly like
  "le délai est de 30 jours";
* it keeps the vocabulary of the training corpus: a term never seen at index time has no weight
  at query time.

Within those limits it is extremely strong, instantaneous, explainable (its score decomposes term
by term) and reproducible — which is why it is the baseline, and why the report measures the gap
instead of assuming a neural retriever is better.

The scoring is implemented in :class:`src.preprocessing.transformers.LexicalVectorizer`:

* ``bm25`` — Okapi BM25, term-frequency saturation plus document-length normalisation;
* ``tfidf_cosine`` — the classic cosine similarity on L2-normalised TF-IDF vectors.

The model persists the chunk table, the learned vocabulary/IDF, the document metadata *and* the
generator configuration (``model.llm``), so a reloaded artefact ranks *and answers* exactly like
the fitted one (two round-trip tests assert it: an artefact that forgets its
``min_overlap``/``support_ratio`` is a model that was never trained).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import ChunkFeatureBuilder
from src.models.base import BaseModel, RetrievedChunk
from src.preprocessing.pipelines import TextPreprocessor
from src.preprocessing.transformers import Chunk, LexicalVectorizer
from src.utils.io import load_pickle, save_pickle
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Columns of the document metadata kept in the artefact.
DOCUMENT_COLUMNS: tuple[str, ...] = ("doc_id", "title", "section", "source", "published_at")


def _llm_node(config: Any) -> dict[str, Any]:
    """Return the ``model.llm`` node of a configuration as a plain mapping."""
    model_node = config.get("model") if isinstance(config, Mapping) else None
    node = model_node.get("llm") if isinstance(model_node, Mapping) else None
    return dict(node) if isinstance(node, Mapping) else {}


def _artifact_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the ``model`` node archived in an artefact (the generator knobs).

    The generator is part of the trained artefact: ``min_overlap`` and ``support_ratio`` decide
    how many sentences an answer may quote, so a reload that falls back on the defaults answers
    differently from the model that was fitted and evaluated. The API key is never archived: it
    stays in the environment.

    Args:
        payload: Artefact payload read from disk.

    Returns:
        A configuration carrying ``model.llm`` when the artefact archived one.
    """
    llm = payload.get("llm")
    if isinstance(llm, Mapping) and llm:
        return {"model": {"llm": dict(llm)}}
    return {}


def _merge_config(
    archived: Mapping[str, Any], provided: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge the configuration archived in an artefact with the one the caller passes.

    Args:
        archived: Configuration rebuilt from the artefact.
        provided: Configuration passed to :meth:`TfidfModel.load` (it wins).

    Returns:
        The effective configuration; ``model`` is merged key by key, every other node is replaced.
    """
    merged: dict[str, Any] = {str(key): value for key, value in archived.items()}
    for key, value in dict(provided or {}).items():
        current = merged.get(str(key))
        if str(key) == "model" and isinstance(value, Mapping) and isinstance(current, Mapping):
            merged["model"] = {**current, **dict(value)}
        else:
            merged[str(key)] = value
    return merged


def _balanced_accuracy(answered: Any, abstained: Any) -> float:
    """Mean of the true-positive and true-negative rates of an answer/abstain rule.

    Args:
        answered: Outcomes of the rule on the answerable questions.
        abstained: Outcomes of the rule on the out-of-corpus questions.

    Returns:
        The balanced accuracy, or ``0.5`` when one of the classes is empty.
    """
    positive = np.asarray(list(answered), dtype="float64")
    negative = np.asarray(list(abstained), dtype="float64")
    if not positive.size or not negative.size:
        return 0.5
    return float(0.5 * (positive.mean() + negative.mean()))


class TfidfModel(BaseModel):
    """TF-IDF / BM25 retriever over the passages of the corpus.

    Attributes:
        framework: Stack identifier archived in the model card.
    """

    framework = "tfidf"

    def __init__(self, **kwargs: Any) -> None:
        """Build the model (see :class:`~src.models.base.BaseModel` for the arguments)."""
        super().__init__(**kwargs)
        self._preprocessor = TextPreprocessor.from_config(self.config.get("preprocessing", {}))
        self._vectorizer = LexicalVectorizer.from_config(self.params.get("lexical", {}))
        self._chunks: pd.DataFrame = self.chunks
        self._matrix: np.ndarray | None = None
        self._documents: pd.DataFrame = pd.DataFrame(columns=list(DOCUMENT_COLUMNS))
        self._features = ChunkFeatureBuilder()

    # ------------------------------------------------------------------ contrat ----------
    @property
    def chunks(self) -> pd.DataFrame:
        """The indexed passages."""
        stored = getattr(self, "_chunks", None)
        if stored is None:
            return super().chunks
        return stored

    def _fit(
        self,
        documents: pd.DataFrame,
        queries: pd.DataFrame | None,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Chunk the corpus and learn the lexical index.

        Args:
            documents: Reference corpus.
            queries: Annotated questions (unused here: the lexical index has nothing to tune —
                a threshold sweep would consume the validation split for no gain).
            callbacks: Training callbacks (fired by the trainer).
            context: Mutable callback context.

        Returns:
            An empty metric mapping: the trainer computes the validation metrics.
        """
        del callbacks, context
        self._chunks = self._preprocessor.prepare_corpus(documents)
        self._documents = documents.loc[
            :, [column for column in DOCUMENT_COLUMNS if column in documents.columns]
        ].copy()
        passages = self._chunks["text"].astype(str).tolist()
        self._vectorizer.fit(passages)
        self._matrix = self._vectorizer.transform(passages)
        self._features = ChunkFeatureBuilder(idf=self._idf_mapping())
        self.abstention_threshold = self._calibrate_threshold(queries)
        return {}

    def _retrieve(
        self, question: str, k: int, filters: Mapping[str, Any]
    ) -> Sequence[RetrievedChunk]:
        """Rank the passages of the index for a question.

        Args:
            question: User question.
            k: Number of passages to return.
            filters: Document-level filters (``section``, ``source``).

        Returns:
            The passages, best first.
        """
        if self._matrix is None:
            msg = "TfidfModel has no index: call fit(documents) or load an artefact first"
            raise RuntimeError(msg)
        scores = self._vectorizer.score(question, self._matrix)
        mask = self._filter_mask(filters)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[: int(k)]
        tokens = self._preprocessor.tokenize(question)
        retrieved: list[RetrievedChunk] = []
        for position, index in enumerate(order, start=1):
            if not np.isfinite(scores[index]):
                break
            chunk = self._chunks.iloc[int(index)]
            document = self._document_of(str(chunk["doc_id"]))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk["chunk_id"]),
                    doc_id=str(chunk["doc_id"]),
                    text=str(chunk["text"]),
                    score=float(scores[index]),
                    rank=position,
                    title=str(document.get("title", "")),
                    section=str(document.get("section", "")),
                    metadata=self._features.build(tokens, dict(chunk), document),
                )
            )
        return retrieved

    def save(self, path: str | Path) -> Path:
        """Persist the index (passages, vocabulary, IDF, document metadata).

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        destination = self._resolve_path(path)
        payload = {
            "algorithm": self.algorithm,
            "params": self.params,
            "task": self.task,
            "random_state": self.random_state,
            "name": self.name,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
            "chunks": self._chunks,
            "documents": self._documents,
            "llm": _llm_node(self.config),
            "fit_result": None if self.fit_result_ is None else self.fit_result_.to_dict(),
            **self._state_to_save(),
        }
        return save_pickle(payload, destination)

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> TfidfModel:
        """Reload a persisted index.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded model, ready to retrieve.
        """
        payload = load_pickle(path)
        model = cls(
            algorithm=str(payload.get("algorithm", "bm25")),
            params=dict(payload.get("params", {})),
            task=str(payload.get("task", "retrieval")),
            random_state=int(payload.get("random_state", 42)),
            config=_merge_config(_artifact_config(payload), config),
            name=str(payload.get("name", "bm25")),
        )
        model._vectorizer = payload["vectorizer"]
        model._matrix = payload["matrix"]
        model._chunks = payload["chunks"]
        model._documents = payload["documents"]
        model._features = ChunkFeatureBuilder(idf=model._idf_mapping())
        model._state_from_payload(payload)
        if payload.get("fit_result"):
            from src.models.base import FitResult

            model.fit_result_ = FitResult.from_dict(payload["fit_result"])
        model._is_fitted = True
        logger.info(
            "Lexical index reloaded | {} passages | vocabulary={}",
            len(model._chunks),
            model._vectorizer.vocabulary_size,
        )
        return model

    # ------------------------------------------------------------------ interne ----------
    def _calibrate_threshold(self, queries: pd.DataFrame | None) -> float:
        """Choose the abstention threshold on the **calibration** split.

        The rule is deliberately simple and stated: the threshold is the value that maximises the
        balanced accuracy of the answer / abstain decision on the calibration questions. A question
        is "answerable" when the corpus holds its answer; the score used is the best passage score
        the retriever produced. Tuning this on the test split would be a leak, and tuning it on the
        validation split would consume the split used for monitoring — hence the dedicated split.

        Args:
            queries: Calibration questions (``None`` keeps the configured threshold).

        Returns:
            The threshold to apply (``0.0`` when calibration is disabled or impossible).
        """
        configured = float(self.params.get("abstention_threshold", 0.0))
        if (
            queries is None
            or queries.empty
            or not bool(self.params.get("calibrate_abstention", True))
        ):
            return configured
        if "answer_type" not in queries.columns:
            return configured
        answerable: list[float] = []
        unanswerable: list[float] = []
        for record in queries.to_dict(orient="records"):
            scores = self._scores_for(str(record["question"]))
            if not len(scores):
                continue
            top = float(np.max(scores))
            target = unanswerable if record.get("answer_type") == "unanswerable" else answerable
            target.append(top)
        if not answerable or not unanswerable:
            logger.warning(
                "Abstention threshold not calibrated ({} answerable, {} hors corpus): "
                "keeping {:.4f}",
                len(answerable),
                len(unanswerable),
                configured,
            )
            return configured
        positive = np.asarray(answerable)
        negative = np.asarray(unanswerable)
        candidates = np.unique(np.concatenate([positive, negative]))
        best_threshold = float(configured)
        best_score = _balanced_accuracy(positive >= configured, negative < configured)
        for candidate in candidates:
            balanced = _balanced_accuracy(positive >= candidate, negative < candidate)
            if balanced > best_score + 1e-12:
                best_score = balanced
                best_threshold = float(candidate)
        gain = best_score - _balanced_accuracy(positive >= configured, negative < configured)
        if gain < self.calibration_min_gain:
            logger.warning(
                "Abstention threshold kept at {:.4f}: calibration on {} questions brings only "
                "{:+.3f} of balanced accuracy, below the {:.2f} required to change the rule",
                configured,
                len(queries),
                gain,
                self.calibration_min_gain,
            )
            return configured
        logger.info(
            "Abstention threshold calibrated | {:.4f} (balanced accuracy {:.3f}, gain {:+.3f} "
            "sur {} questions)",
            best_threshold,
            best_score,
            gain,
            len(queries),
        )
        return best_threshold

    @property
    def calibration_min_gain(self) -> float:
        """Minimum balanced-accuracy gain required to replace the configured threshold.

        Calibrating on a signal that does not separate the classes produces a threshold that is
        worse than the configured one on the other splits: the calibration is therefore accepted
        only when it buys a real improvement, and the refusal is logged with its measured gain.
        """
        return float(self.params.get("calibration_min_gain", 0.05))

    def _scores_for(self, question: str) -> np.ndarray:
        """Return the retrieval scores of every indexed passage for a question."""
        if self._matrix is None:
            return np.zeros(0, dtype="float32")
        return self._vectorizer.score(question, self._matrix)

    def _idf_mapping(self) -> dict[str, float]:
        """Return the IDF weight of every term of the vocabulary.

        The weights feed the explicability features (``src.features.build_features``).
        """
        vectorizer = self._vectorizer
        idf = getattr(vectorizer, "_idf", None)
        if idf is None:
            return {}
        return {
            str(term): float(idf[index])
            for term, index in vectorizer._vectorizer.vocabulary_.items()
        }

    def _document_of(self, doc_id: str) -> Mapping[str, Any]:
        """Return the metadata of a document (empty mapping when unknown)."""
        if self._documents.empty or "doc_id" not in self._documents.columns:
            return {}
        match = self._documents.loc[self._documents["doc_id"] == doc_id]
        return {} if match.empty else match.iloc[0].to_dict()

    def _filter_mask(self, filters: Mapping[str, Any]) -> np.ndarray:
        """Build the boolean mask of the passages allowed by the filters."""
        mask = np.ones(len(self._chunks), dtype=bool)
        if not filters:
            return mask
        if self._documents.empty:
            logger.warning(
                "Filters {} ignored: no document metadata in the artefact", dict(filters)
            )
            return mask
        allowed = set(self._documents["doc_id"].astype(str))
        for column, value in filters.items():
            if column not in self._documents.columns:
                logger.warning("Unknown filter '{}' ignored", column)
                continue
            allowed &= set(
                self._documents.loc[
                    self._documents[column].astype(str) == str(value), "doc_id"
                ].astype(str)
            )
        mask = self._chunks["doc_id"].astype(str).isin(allowed).to_numpy()
        return mask

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the index statistics archived in the fit result."""
        return {
            "vocabulary_size": float(self._vectorizer.vocabulary_size),
            "mean_passage_tokens": float(self._chunks["n_tokens"].mean())
            if len(self._chunks)
            else 0.0,
            "n_documents": float(len(self._documents)),
            "scoring": self._vectorizer.mode,
        }

    def _effective_params(self) -> dict[str, Any]:
        """Return the resolved hyper-parameters (vocabulary size included)."""
        return {
            **dict(self.params),
            # The calibrated threshold is *learned*: the card must publish the value the
            # model actually applies, not the placeholder the configuration declared.
            "abstention_threshold": float(self.abstention_threshold),
            "lexical": {
                "mode": self._vectorizer.mode,
                "ngram_range": list(self._vectorizer.ngram_range),
                "k1": self._vectorizer.k1,
                "b": self._vectorizer.b,
                "vocabulary_size": self._vectorizer.vocabulary_size,
            },
        }

    def benchmark_chunking(
        self, documents: pd.DataFrame, queries: pd.DataFrame, sizes: Sequence[int]
    ) -> pd.DataFrame:
        """Measure recall@5 for several passage sizes (used by notebook 03).

        Re-chunking and re-indexing the corpus for each size is exactly the experiment an
        engineer must run before choosing a chunk size — and it is cheap here, which is the
        advantage of a lexical index.

        Args:
            documents: Reference corpus.
            queries: Annotated questions (validation split).
            sizes: Passage sizes to compare, in tokens.

        Returns:
            One row per size with the vocabulary, the number of passages and recall@5.
        """
        from src.training.losses_metrics import recall_at_k

        rows: list[dict[str, Any]] = []
        original = self._preprocessor.chunker.max_tokens
        try:
            for size in sizes:
                self._preprocessor.chunker.max_tokens = int(size)
                self._preprocessor.chunker.overlap_tokens = max(int(size * 0.25), 10)
                self._chunks = self._preprocessor.prepare_corpus(documents)
                passages = self._chunks["text"].astype(str).tolist()
                self._vectorizer = LexicalVectorizer.from_config(self.params.get("lexical", {}))
                self._vectorizer.fit(passages)
                self._matrix = self._vectorizer.transform(passages)
                recalls: list[float] = []
                for record in queries.to_dict(orient="records"):
                    if record["answer_type"] == "unanswerable":
                        continue
                    gold = {item for item in str(record["gold_doc_ids"]).split(",") if item}
                    passages_found = self._retrieve(str(record["question"]), 5, {})
                    recalls.append(recall_at_k([item.doc_id for item in passages_found], gold, 5))
                rows.append(
                    {
                        "max_tokens": int(size),
                        "n_chunks": len(self._chunks),
                        "vocabulary_size": int(self._vectorizer.vocabulary_size),
                        "recall_at_5": float(np.mean(recalls)) if recalls else float("nan"),
                    }
                )
        finally:
            self._preprocessor.chunker.max_tokens = original
        return pd.DataFrame(rows)


def chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    """Return a :class:`Chunk` as a flat row (kept for the notebooks and the tests)."""
    return chunk.to_row()


__all__ = ["DOCUMENT_COLUMNS", "TfidfModel", "chunk_to_row"]
