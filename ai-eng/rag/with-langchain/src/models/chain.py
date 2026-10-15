"""LCEL composition of the RAG assistant: retrieval, prompt, LLM adapter and grounding.

The project composes four LangChain components, and nothing else:

* :class:`ChunkRetriever` — a ``BaseRetriever`` over the passages of the corpus. The *ranking* is
  the project's own (BM25, dense hashing embeddings or the fusion of both): the framework is used
  for what it is good at, composing, not for the retrieval statistics;
* a ``PromptTemplate`` read from ``conf/model/default.yaml`` (``model.prompt.template``) — the
  prompt is configuration, never a string buried in the code;
* an adapter that exposes any :class:`~src.models.llm.BaseLLM` (the deterministic extractive
  generator, or the optional OpenAI-compatible client) as a ``Runnable``;
* a grounding node that turns the raw :class:`~src.models.llm.Generation` into a
  :class:`GroundedAnswer` keeping the citations, the rendered prompt and the passages.

Why a grounding node rather than ``... | llm | StrOutputParser()``? Because ``StrOutputParser``
returns a ``str`` and would drop the identifiers of the passages the answer came from. Citation
precision is one of the metrics of this family, and an answer that cannot be traced back to a
dated document is not acceptable in the use case: the grounding node is a ``Runnable`` like any
other, so the chain stays composable and observable.

Everything here is offline and deterministic: no tracing, no API key, no download. The retriever
and the chain are **rebuilt** from the persisted state at load time, because an LCEL graph is not
part of the artefact — the state is (see :class:`~src.models.model.LangChainModel`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableAssign,
    RunnableLambda,
    RunnableParallel,
)
from pydantic import Field

from src.models.llm import BaseLLM, Generation, Passage
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Prompt used when the configuration declares none. It states the two rules the metrics measure:
#: answer **only** from the context, and cite the passages used. The instruction is written to be
#: readable by a remote model (``ABSTENTION``, ``SOURCES:``) and is passed to every adapter — the
#: extractive one ignores it by design, a chat model needs it.
DEFAULT_PROMPT_TEMPLATE = """Tu réponds uniquement à partir du contexte fourni, en français.
Si le contexte ne contient pas la réponse, réponds exactement 'ABSTENTION'.
Termine ta réponse par 'SOURCES: ' suivi des identifiants des passages utilisés.

Contexte:
{context}

Question: {question}
Réponse:"""

#: Default budget for the stuffed context, in characters.
DEFAULT_MAX_CONTEXT_CHARS = 2400

#: Metadata keys copied from a passage into a ``Document``.
DOCUMENT_METADATA: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "rank",
    "score",
    "title",
    "section",
    "n_chars",
)


@dataclass(slots=True)
class GroundedAnswer:
    """Answer produced by the chain, with everything needed to audit it.

    Attributes:
        question: Question asked.
        text: Generated answer (empty when the chain abstained).
        citations: Identifiers of the passages the answer is built from.
        rationale: Why the answer looks the way it does (abstention, kept sentences, ...).
        documents: Passages handed to the model, best first.
        context: Exact context string stuffed into the prompt.
        prompt: Exact prompt sent to the model.
    """

    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    rationale: str = ""
    documents: list[Document] = field(default_factory=list)
    context: str = ""
    prompt: str = ""

    @property
    def abstained(self) -> bool:
        """Whether the chain produced no answer text."""
        return not self.text.strip()

    def to_generation(self) -> Generation:
        """Return the answer as the model-level :class:`Generation` object."""
        return Generation(
            text=self.text,
            cited_chunk_ids=list(self.citations),
            rationale=self.rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (the prompt and the texts are kept out of it)."""
        return {
            "question": self.question,
            "text": self.text,
            "citations": list(self.citations),
            "n_documents": len(self.documents),
            "abstained": self.abstained,
            "prompt_chars": len(self.prompt),
            "context_chars": len(self.context),
            "rationale": self.rationale,
        }


class ChunkRetriever(BaseRetriever):
    """LangChain retriever over the passages of the corpus.

    The class implements the framework extension point (``BaseRetriever``) and keeps the project's
    scoring behind it: a caller gets ``retriever.invoke(question)``, batching and callbacks for
    free, and the ranking stays the one measured by the evaluation.

    Attributes:
        algorithm: Algorithm of the stack (``langchain_lexical``, ``langchain_dense``,
            ``langchain_hybrid``).
        chunks: One row per passage, document metadata already joined in.
        lexical: Fitted :class:`~src.preprocessing.transformers.LexicalVectorizer` (BM25 or
            TF-IDF), or ``None`` before the fit.
        lexical_matrix: Scores-ready ``(n_chunks, n_terms)`` matrix of the lexical arm.
        embedder: Fitted :class:`~src.preprocessing.transformers.HashingEmbedder` (dense arm).
        embedding_matrix: L2-normalised ``(n_chunks, dimension)`` matrix of the dense arm.
        k: Number of passages returned by default.
        filters: Default metadata filters (``{"section": "procedure"}``).
        rrf_constant: Rank constant of the reciprocal rank fusion (hybrid algorithm).
    """

    # ``Any`` on purpose: the fitted estimators and the score matrices are carried **by
    # reference**. pydantic validates annotated dataclass fields by rebuilding them, which would
    # silently drop the learned vocabulary of the lexical vectoriser.
    algorithm: str = "langchain_lexical"
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    lexical: Any = None
    lexical_matrix: Any = None
    embedder: Any = None
    embedding_matrix: Any = None
    k: int = 5
    filters: dict[str, str] = Field(default_factory=dict)
    rrf_constant: int = 60

    def score_query(self, query: str) -> np.ndarray:
        """Score every indexed passage for a query.

        Args:
            query: User question.

        Returns:
            One score per passage, in the order of :attr:`chunks`.

        Raises:
            RuntimeError: When the retriever has not been fitted (empty index).
        """
        if not self.chunks:
            msg = "ChunkRetriever has no index: build it with a fitted LangChainModel first"
            raise RuntimeError(msg)
        if self.algorithm == "langchain_dense":
            return self._dense_scores(query)
        if self.algorithm == "langchain_hybrid":
            return self._fused_scores(query)
        return np.asarray(self.lexical.score(query, self.lexical_matrix), dtype="float64")

    def documents_for(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Rank the passages and return them as ``Document`` objects.

        Args:
            query: User question.
            k: Number of passages to return (defaults to :attr:`k`).
            filters: Metadata filters overriding :attr:`filters` for this call.

        Returns:
            The passages, best first, each one carrying its identifiers and score.
        """
        scores = self.score_query(query)
        mask = self._filter_mask(self.filters if filters is None else filters)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[: int(k or self.k)]
        documents: list[Document] = []
        for rank, index in enumerate(order, start=1):
            score = float(scores[int(index)])
            if not np.isfinite(score):
                break
            row = self.chunks[int(index)]
            documents.append(
                Document(
                    page_content=str(row["text"]),
                    metadata={
                        "chunk_id": str(row["chunk_id"]),
                        "doc_id": str(row["doc_id"]),
                        "rank": rank,
                        "score": score,
                        "title": str(row.get("title", "")),
                        "section": str(row.get("section", "")),
                        "n_chars": len(str(row["text"])),
                    },
                )
            )
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Framework hook: retrieve the passages of a query (``retriever.invoke``)."""
        del run_manager
        return self.documents_for(query)

    # ------------------------------------------------------------------ interne ----------
    def _dense_scores(self, query: str) -> np.ndarray:
        """Cosine similarity between the query embedding and the passage embeddings."""
        vector = np.asarray(self.embedder.transform([query])[0], dtype="float64")
        return np.asarray(self.embedding_matrix @ vector, dtype="float64")

    def _fused_scores(self, query: str) -> np.ndarray:
        """Reciprocal rank fusion of the lexical and the dense rankings.

        The two arms do not produce comparable scores (BM25 is unbounded, a cosine lives in
        ``[-1, 1]``): fusing the *ranks* instead of the scores is the only way to combine them
        without inventing a normalisation. This is the standard RRF formula, and the scores it
        returns are only meaningful for ranking — which is all the retriever uses them for.
        """
        lexical = np.asarray(self.lexical.score(query, self.lexical_matrix), dtype="float64")
        dense = self._dense_scores(query)
        fused = np.zeros(len(self.chunks), dtype="float64")
        for scores in (lexical, dense):
            for rank, index in enumerate(np.argsort(-scores, kind="stable"), start=1):
                fused[int(index)] += 1.0 / (self.rrf_constant + rank)
        return fused

    def _filter_mask(self, filters: Mapping[str, Any]) -> np.ndarray:
        """Build the boolean mask of the passages allowed by the filters.

        Args:
            filters: Metadata filters (``{"section": "procedure"}``).

        Returns:
            One boolean per passage; unknown filter keys are ignored with a warning rather than
            silently returning nothing.
        """
        mask = np.ones(len(self.chunks), dtype=bool)
        for key, value in dict(filters or {}).items():
            if not any(key in row for row in self.chunks):
                logger.warning("Unknown filter '{}' ignored", key)
                continue
            mask &= np.asarray(
                [str(row.get(key, "")) == str(value) for row in self.chunks], dtype=bool
            )
        return mask


@dataclass
class LangChainGroundedLLM(BaseLLM):
    """Expose a :class:`BaseLLM` adapter through the project's LCEL chain.

    The model layer calls ``generate(question, passages)`` (the contract of the modality); this
    adapter runs the whole composed chain — context stuffing, prompt template, adapter call,
    grounding — and returns the grounded :class:`Generation`. The chain owner is therefore the
    stack, while the generator behind it stays interchangeable.

    Attributes:
        adapter: Generator actually producing the text (extractive or remote).
        chain: LCEL graph ``dict -> GroundedAnswer`` built by :func:`build_answer_chain`.
        prompt_chars: Size of the configured prompt template (model card traceability).
    """

    adapter: BaseLLM
    chain: Runnable[Any, GroundedAnswer]
    prompt_chars: int = 0
    provider = "langchain"

    def generate(
        self, question: str, passages: Sequence[Passage], *, prompt: str | None = None
    ) -> Generation:
        """Run the chain and return the grounded answer.

        Args:
            question: User question.
            passages: Passages retrieved by the retriever.
            prompt: Prompt already rendered by the caller (unused: the chain owns the prompt).

        Returns:
            The generated answer with its citations.
        """
        del prompt
        payload = {"question": str(question), "documents": documents_of(passages)}
        answer: GroundedAnswer = self.chain.invoke(payload)
        return answer.to_generation()

    def describe(self) -> dict[str, Any]:
        """Describe the chain for the model card (never a prompt body, only its size)."""
        return {
            "provider": f"langchain+{self.adapter.provider}",
            "framework": "langchain-core",
            "stages": ["context", "prompt", "llm", "grounding"],
            "prompt_chars": self.prompt_chars,
            "adapter": self.adapter.describe(),
        }


def build_prompt(config: Mapping[str, Any] | Any) -> tuple[PromptTemplate, int]:
    """Build the grounded prompt declared in ``model.prompt``.

    Args:
        config: Application configuration (mapping, pydantic model or OmegaConf node).

    Returns:
        The prompt template and the character budget of the stuffed context.

    Raises:
        ValueError: When the configured template does not expose the ``context`` and ``question``
            variables — a prompt that cannot receive the retrieved passages would silently answer
            from the model's memory, which is exactly what this project measures.
    """
    root = _as_mapping(config)
    node = root.get("model")
    prompt_node = dict(node.get("prompt") or {}) if isinstance(node, Mapping) else {}
    template = str(prompt_node.get("template") or DEFAULT_PROMPT_TEMPLATE)
    max_context_chars = int(prompt_node.get("max_context_chars") or DEFAULT_MAX_CONTEXT_CHARS)
    prompt = PromptTemplate.from_template(template)
    missing = {"context", "question"} - set(prompt.input_variables)
    if missing:
        msg = (
            f"The prompt template of model.prompt must expose {sorted({'context', 'question'})} "
            f"(missing: {sorted(missing)})"
        )
        raise ValueError(msg)
    return prompt, max_context_chars


def documents_of(passages: Sequence[Passage]) -> list[Document]:
    """Convert retrieved passages into LangChain ``Document`` objects.

    Args:
        passages: Passages returned by the model layer.

    Returns:
        The same passages, with their identifiers and score in the metadata.
    """
    return [
        Document(
            page_content=passage.text,
            metadata={
                "chunk_id": passage.chunk_id,
                "doc_id": passage.doc_id,
                "rank": passage.rank,
                "score": passage.score,
                "title": passage.title,
                "section": passage.section,
                "n_chars": len(passage.text),
            },
        )
        for passage in passages
    ]


def passages_of(documents: Sequence[Document]) -> list[Passage]:
    """Convert LangChain ``Document`` objects back into model-level passages.

    Args:
        documents: Documents produced by :class:`ChunkRetriever`.

    Returns:
        The passages handed to the LLM adapter.
    """
    passages: list[Passage] = []
    for document in documents:
        metadata = document.metadata
        passages.append(
            Passage(
                chunk_id=str(metadata.get("chunk_id", "")),
                doc_id=str(metadata.get("doc_id", "")),
                text=document.page_content,
                score=float(metadata.get("score", 0.0) or 0.0),
                rank=int(metadata.get("rank", len(passages) + 1) or 0),
                title=str(metadata.get("title", "")),
                section=str(metadata.get("section", "")),
            )
        )
    return passages


def format_documents(documents: Sequence[Document], *, max_chars: int) -> str:
    """Stuff the retrieved passages into a bounded context string.

    Args:
        documents: Passages, best first.
        max_chars: Maximum number of characters of the context.

    Returns:
        One numbered line per passage (``[rank] (chunk_id) text``), truncated when the budget is
        exceeded. Passages that do not fit are dropped rather than cut in the middle: a truncated
        passage can make a false statement look grounded.
    """
    lines: list[str] = []
    used = 0
    for document in documents:
        metadata = document.metadata
        line = f"[{metadata.get('rank', len(lines) + 1)}] ({metadata.get('chunk_id', '?')}) "
        line += document.page_content.strip()
        if lines and used + len(line) > max_chars:
            logger.debug(
                "Context budget reached: {} passage(s) dropped", len(documents) - len(lines)
            )
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_answer_chain(
    *,
    prompt: PromptTemplate,
    adapter: BaseLLM,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> Runnable[Any, GroundedAnswer]:
    """Build the LCEL chain ``payload -> GroundedAnswer``.

    The graph has four stages, each one a ``Runnable``:

    1. **context** — stuff the retrieved documents into a bounded string;
    2. **prompt** — render the ``PromptTemplate`` (a real runnable node, so the prompt can be
       swapped, traced or evaluated on its own);
    3. **llm** — call the adapter with the rendered prompt;
    4. **grounding** — keep the text, the citations, the rationale and the documents together.

    Args:
        prompt: Prompt template declaring the ``context`` and ``question`` variables.
        adapter: Generator producing the text from the prompt and the passages.
        max_context_chars: Character budget of the stuffed context.

    Returns:
        The composed chain.
    """

    def with_context(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Stage 1: assemble the context of the retrieved documents."""
        documents = list(payload.get("documents") or [])
        return {
            **dict(payload),
            "documents": documents,
            "context": format_documents(documents, max_chars=max_context_chars),
        }

    def prompt_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Stage 2 (input): expose exactly the variables the template declares."""
        return {"context": str(payload.get("context", "")), "question": str(payload["question"])}

    def with_generation(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Stage 3: call the adapter once, with the prompt the chain rendered."""
        documents = list(payload.get("documents") or [])
        question = str(payload["question"])
        rendered = payload["prompt_value"].to_string()
        generation = adapter.generate(question, passages_of(documents), prompt=rendered)
        return {**dict(payload), "generation": generation, "prompt": rendered}

    def grounded_answer(payload: Mapping[str, Any]) -> GroundedAnswer:
        """Stage 4: keep the answer, its citations and the passages that support them."""
        documents = list(payload.get("documents") or [])
        generation: Generation = payload["generation"]
        known = {str(document.metadata.get("chunk_id", "")) for document in documents}
        citations = [identifier for identifier in generation.cited_chunk_ids if identifier in known]
        return GroundedAnswer(
            question=str(payload["question"]),
            text=str(generation.text),
            citations=citations,
            rationale=str(generation.rationale),
            documents=documents,
            context=str(payload.get("context", "")),
            prompt=str(payload.get("prompt", "")),
        )

    return (
        RunnableLambda(with_context)
        | RunnableAssign(RunnableParallel(prompt_value=RunnableLambda(prompt_inputs) | prompt))
        | RunnableLambda(with_generation)
        | RunnableLambda(grounded_answer)
    )


def build_retrieval_chain(
    *,
    retriever: ChunkRetriever,
    answer_chain: Runnable[Any, GroundedAnswer],
) -> Runnable[str, GroundedAnswer]:
    """Build the end-to-end chain ``question -> GroundedAnswer`` (retriever then answer chain).

    Args:
        retriever: Fitted retriever of the corpus.
        answer_chain: Chain produced by :func:`build_answer_chain`.

    Returns:
        The composed chain used by the tests and the notebook to inspect the full path.
    """

    def retrieve(question: str) -> dict[str, Any]:
        """First stage: retrieve the passages of the question."""
        text = str(question)
        return {"question": text, "documents": retriever.documents_for(text)}

    return RunnableLambda(retrieve) | answer_chain


def _as_mapping(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalise a configuration object (pydantic model, OmegaConf or mapping) to a dict."""
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "to_container"):  # OmegaConf DictConfig
        return dict(config.to_container(resolve=True))
    return dict(config or {})


__all__ = [
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_PROMPT_TEMPLATE",
    "DOCUMENT_METADATA",
    "ChunkRetriever",
    "GroundedAnswer",
    "LangChainGroundedLLM",
    "build_answer_chain",
    "build_prompt",
    "build_retrieval_chain",
    "documents_of",
    "format_documents",
    "passages_of",
]
