"""RAG model of the LangChain stack: an LCEL chain wired to a hybrid-ready index.

The model is the composition of two halves that are usually conflated:

* **the index** — the corpus is chunked, the passages are scored by BM25 (or by dense hashing
  embeddings, or by the fusion of both), and the learned state is persisted in joblib;
* **the chain** — an LCEL graph (``src/models/chain.py``) that stuffs the retrieved passages into
  a prompt, calls an LLM adapter and grounds the answer back on the passages it came from.

The framework is therefore *used*, not merely imported: retrieval goes through a
``langchain_core`` retriever, the prompt is a ``PromptTemplate`` read from the configuration, and
the answer travels through a composed ``Runnable``. What the stack does *not* delegate is the
ranking statistics and the grounding rules — those are the parts the evaluation actually measures.

The dense arm is deliberately cheap and offline: hashed n-grams compressed by a truncated SVD
learned on the passages (``HashingEmbedder``). It is the honest way to show what a semantic
representation buys over BM25 without downloading a pretrained encoder, and the comparison
between the three algorithms of the stack is computed by notebook 04 on the validation split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from src.features.build_features import ChunkFeatureBuilder
from src.models.base import BaseModel, RetrievedChunk
from src.models.chain import (
    ChunkRetriever,
    GroundedAnswer,
    LangChainGroundedLLM,
    build_answer_chain,
    build_prompt,
    build_retrieval_chain,
)
from src.models.llm import BaseLLM
from src.preprocessing.pipelines import TextPreprocessor
from src.preprocessing.transformers import Chunk, HashingEmbedder, LexicalVectorizer
from src.utils.io import load_pickle, save_pickle
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Columns of the document metadata kept in the artefact.
DOCUMENT_COLUMNS: tuple[str, ...] = ("doc_id", "title", "section", "source", "published_at")


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


class LangChainModel(BaseModel):
    """RAG assistant built on an LCEL chain over a lexical/dense index.

    Attributes:
        framework: Stack identifier archived in the model card.
    """

    framework = "langchain"

    def __init__(self, **kwargs: Any) -> None:
        """Build the model (see :class:`~src.models.base.BaseModel` for the arguments)."""
        super().__init__(**kwargs)
        self._preprocessor = TextPreprocessor.from_config(self.config.get("preprocessing", {}))
        self._lexical = LexicalVectorizer.from_config(self.params.get("lexical", {}))
        self._embedder = HashingEmbedder.from_config(self.params.get("embedding", {}))
        self._chunks: pd.DataFrame = self.chunks
        self._documents: pd.DataFrame = pd.DataFrame(columns=list(DOCUMENT_COLUMNS))
        self._lexical_matrix: np.ndarray | None = None
        self._embedding_matrix: np.ndarray | None = None
        self._features = ChunkFeatureBuilder()
        self._prompt, self._max_context_chars = build_prompt(self.config)
        # The adapter built by the base class is the *inner* generator: the chain owns it, and the
        # model-level generator becomes the chain itself (same contract, one more layer).
        adapter: BaseLLM = self.llm
        self._adapter = adapter
        self._retriever = self._build_retriever()
        self.llm = LangChainGroundedLLM(
            adapter=adapter,
            chain=self._build_chain(),
            prompt_chars=len(self._prompt.template),
        )

    # ------------------------------------------------------------------ contrat ----------
    @property
    def chunks(self) -> pd.DataFrame:
        """The indexed passages."""
        stored = getattr(self, "_chunks", None)
        if stored is None:
            return super().chunks
        return stored

    @property
    def chain(self) -> Runnable[str, GroundedAnswer]:
        """The end-to-end chain ``question -> answer`` (retriever then prompt then adapter).

        This is the object a service exposes: it takes a question and returns the answer with its
        citations, its prompt and the passages that support it.
        """
        return build_retrieval_chain(
            retriever=self._retriever,
            answer_chain=self._build_chain(),
        )

    @property
    def retriever(self) -> ChunkRetriever:
        """The ``BaseRetriever`` used by the chain (fitted once the model is indexed)."""
        return self._retriever

    def _fit(
        self,
        documents: pd.DataFrame,
        queries: pd.DataFrame | None,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Chunk the corpus, learn both retrieval arms and rebuild the chain.

        Args:
            documents: Reference corpus.
            queries: Annotated questions (calibration split only).
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
        self._lexical.fit(passages)
        self._lexical_matrix = self._lexical.transform(passages)
        self._embedder.fit(passages)
        self._embedding_matrix = self._embedder.transform(passages)
        self._features = ChunkFeatureBuilder(idf=self._idf_mapping())
        self._retriever = self._build_retriever()
        self.llm = LangChainGroundedLLM(
            adapter=self._adapter,
            chain=self._build_chain(),
            prompt_chars=len(self._prompt.template),
        )
        self.abstention_threshold = self._calibrate_threshold(queries)
        return {}

    def _retrieve(
        self, question: str, k: int, filters: Mapping[str, Any]
    ) -> Sequence[RetrievedChunk]:
        """Rank the passages through the LangChain retriever.

        Args:
            question: User question.
            k: Number of passages to return.
            filters: Document-level filters (``section``, ``source``).

        Returns:
            The passages, best first.

        Raises:
            RuntimeError: When the model has not been indexed.
        """
        if self._lexical_matrix is None:
            msg = "LangChainModel has no index: call fit(documents) or load an artefact first"
            raise RuntimeError(msg)
        documents = self._retriever.documents_for(question, k=int(k), filters=filters)
        tokens = self._preprocessor.tokenize(question)
        retrieved: list[RetrievedChunk] = []
        for document in documents:
            metadata = dict(document.metadata)
            chunk_id = str(metadata["chunk_id"])
            row = self._chunk_row(chunk_id)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(metadata["doc_id"]),
                    text=document.page_content,
                    score=float(metadata.get("score", 0.0)),
                    rank=int(metadata.get("rank", len(retrieved) + 1)),
                    title=str(metadata.get("title", "")),
                    section=str(metadata.get("section", "")),
                    metadata=self._features.build(tokens, row, self._document_of(chunk_id)),
                )
            )
        return retrieved

    def save(self, path: str | Path) -> Path:
        """Persist the index and the learned state (passages, vocabulary, embeddings).

        The LCEL chain is **not** serialised: it holds closures and is rebuilt at load time from
        the configuration. Persisting state instead of objects is what keeps a reloaded artefact
        independent from the library version that wrote it.

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
            "prompt_template": self._prompt.template,
            "max_context_chars": int(self._max_context_chars),
            "lexical": self._lexical,
            "lexical_matrix": self._lexical_matrix,
            "embedder": self._embedder,
            "embedding_matrix": self._embedding_matrix,
            "chunks": self._chunks,
            "documents": self._documents,
            "fit_result": None if self.fit_result_ is None else self.fit_result_.to_dict(),
            **self._state_to_save(),
        }
        return save_pickle(payload, destination)

    @classmethod
    def load(
        cls, path: str | Path, *, config: Mapping[str, Any] | None = None
    ) -> LangChainModel:
        """Reload an artefact written by :meth:`save` and rebuild its chain.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded model, ready to retrieve and answer.
        """
        payload = load_pickle(path)
        model = cls(
            algorithm=str(payload.get("algorithm", "langchain_lexical")),
            params=dict(payload.get("params", {})),
            task=str(payload.get("task", "retrieval")),
            random_state=int(payload.get("random_state", 42)),
            config=dict(config or {}),
            name=str(payload.get("name", "langchain_lexical")),
        )
        if "prompt_template" in payload and not _configured_prompt(config):
            model._prompt = PromptTemplate.from_template(str(payload["prompt_template"]))
        model._max_context_chars = int(
            payload.get("max_context_chars", model._max_context_chars)
        )
        model._lexical = payload["lexical"]
        model._lexical_matrix = payload["lexical_matrix"]
        model._embedder = payload["embedder"]
        model._embedding_matrix = payload["embedding_matrix"]
        model._chunks = payload["chunks"]
        model._documents = payload["documents"]
        model._features = ChunkFeatureBuilder(idf=model._idf_mapping())
        model._retriever = model._build_retriever()
        model.llm = LangChainGroundedLLM(
            adapter=model._adapter,
            chain=model._build_chain(),
            prompt_chars=len(model._prompt.template),
        )
        model._state_from_payload(payload)
        if payload.get("fit_result"):
            from src.models.base import FitResult

            model.fit_result_ = FitResult.from_dict(payload["fit_result"])
        model._is_fitted = True
        logger.info(
            "LangChain index reloaded | {} passages | lexical vocabulary={} | dense dimension={}",
            len(model._chunks),
            model._lexical.vocabulary_size,
            model._dense_width(),
        )
        return model

    # ------------------------------------------------------------------ interne ----------
    def _build_chain(self) -> Runnable[Any, GroundedAnswer]:
        """Build the answer chain (context, prompt, adapter, grounding)."""
        return build_answer_chain(
            prompt=self._prompt,
            adapter=self._adapter,
            max_context_chars=self._max_context_chars,
        )

    def _build_retriever(self) -> ChunkRetriever:
        """Build the ``BaseRetriever`` of the index (empty before the fit)."""
        rows = self._index_rows()
        return ChunkRetriever(
            algorithm=self.algorithm,
            chunks=rows,
            lexical=self._lexical,
            lexical_matrix=self._lexical_matrix,
            embedder=self._embedder,
            embedding_matrix=self._embedding_matrix,
            k=int(self.params.get("retrieval_k", 5)),
            rrf_constant=int(self.params.get("rrf_constant", 60)),
        )

    def _index_rows(self) -> list[dict[str, Any]]:
        """Join the passages with their document metadata (the retriever's index)."""
        if self._chunks.empty:
            return []
        frame = self._chunks.copy()
        for column in ("title", "section", "source", "published_at"):
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        if not self._documents.empty and "doc_id" in self._documents.columns:
            frame = frame.merge(self._documents, on="doc_id", how="left")
        frame["text"] = frame["text"].astype(str)
        return [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]

    def _chunk_row(self, chunk_id: str) -> dict[str, Any]:
        """Return the passage row of an identifier (empty when unknown)."""
        if self._chunks.empty or "chunk_id" not in self._chunks.columns:
            return {}
        match = self._chunks.loc[self._chunks["chunk_id"].astype(str) == chunk_id]
        return {} if match.empty else match.iloc[0].to_dict()

    def _document_of(self, chunk_id: str) -> Mapping[str, Any]:
        """Return the metadata of the document a passage belongs to (empty when unknown)."""
        row = self._chunk_row(chunk_id)
        if not row or self._documents.empty:
            return {}
        match = self._documents.loc[self._documents["doc_id"] == row.get("doc_id")]
        return {} if match.empty else match.iloc[0].to_dict()

    def _calibrate_threshold(self, queries: pd.DataFrame | None) -> float:
        """Choose the abstention threshold on the **calibration** split.

        The rule is the one of the family: the threshold maximises the balanced accuracy of the
        answer / abstain decision, and it is kept only when it buys a real improvement over the
        configured value — the gain is measured and logged, refusals included.

        Args:
            queries: Calibration questions (``None`` keeps the configured threshold).

        Returns:
            The threshold to apply.
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
        """Minimum balanced-accuracy gain required to replace the configured threshold."""
        return float(self.params.get("calibration_min_gain", 0.05))

    def _scores_for(self, question: str) -> np.ndarray:
        """Return the retrieval scores of every indexed passage for a question."""
        if self._lexical_matrix is None:
            return np.zeros(0, dtype="float64")
        if self.algorithm == "langchain_dense":
            return self._retriever._dense_scores(question)
        if self.algorithm == "langchain_hybrid":
            return self._retriever._fused_scores(question)
        return np.asarray(self._lexical.score(question, self._lexical_matrix), dtype="float64")

    def _idf_mapping(self) -> dict[str, float]:
        """Return the IDF weight of every term of the vocabulary (explicability features)."""
        vectorizer = self._lexical
        idf = getattr(vectorizer, "_idf", None)
        if idf is None:
            return {}
        return {
            str(term): float(idf[index])
            for term, index in vectorizer._vectorizer.vocabulary_.items()
        }

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the index statistics archived in the fit result (published as ``index_*``).

        The keys follow the convention of the family (``n_documents``, ``vocabulary_size``,
        ``mean_passage_tokens``, ``scoring``) so that the monitoring callback and the notebooks
        read the same names whatever the stack. Two entries are specific to this stack: the real
        width of the dense matrix (the SVD is shrunk on small corpora) and the rank constant of
        the fusion.
        """
        return {
            "n_documents": float(len(self._documents)),
            "vocabulary_size": float(self._lexical.vocabulary_size),
            "dense_dimension": float(self._dense_width()),
            "mean_passage_tokens": float(self._chunks["n_tokens"].mean())
            if len(self._chunks)
            else 0.0,
            "scoring": self.algorithm,
            "rrf_constant": float(self.params.get("rrf_constant", 60)),
        }

    def _dense_width(self) -> int:
        """Real number of columns of the dense matrix (0 before the fit)."""
        if self._embedding_matrix is None:
            return 0
        return int(np.asarray(self._embedding_matrix).shape[1])

    def _effective_params(self) -> dict[str, Any]:
        """Return the resolved hyper-parameters (both retrieval arms included)."""
        return {
            **dict(self.params),
            "lexical": {
                "mode": self._lexical.mode,
                "ngram_range": list(self._lexical.ngram_range),
                "k1": self._lexical.k1,
                "b": self._lexical.b,
                "vocabulary_size": self._lexical.vocabulary_size,
            },
            "embedding": {
                "n_features": self._embedder.n_features,
                "n_components": self._embedder.n_components,
                "ngram_range": list(self._embedder.ngram_range),
                "dimension": self._dense_width(),
            },
            "prompt_chars": len(self._prompt.template),
        }


def _configured_prompt(config: Mapping[str, Any] | None) -> bool:
    """Whether a configuration carries an explicit prompt template."""
    if not config:
        return False
    model_node = config.get("model")
    if not isinstance(model_node, Mapping):
        return False
    prompt_node = model_node.get("prompt")
    return isinstance(prompt_node, Mapping) and bool(prompt_node.get("template"))


def _json_scalar(value: Any) -> Any:
    """Convert a pandas/NumPy scalar to a JSON-friendly value (``Document`` metadata rules)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    """Return a :class:`Chunk` as a flat row (kept for the notebooks and the tests)."""
    return chunk.to_row()


__all__ = ["DOCUMENT_COLUMNS", "LangChainModel", "chunk_to_row"]
