"""Framework-agnostic model contract of the text modality.

Everything in the project depends on :class:`BaseModel` and on nothing else: the trainer, the
evaluator, the predictor and the notebooks manipulate this abstraction, never a framework class.
Swapping TF-IDF for a LangChain chain or a transformer embedding is therefore a configuration
change (``model.algorithm``), not a refactor.

The contract differs from the tabular one on purpose. A retrieval system does not predict a
label from a feature matrix; it *indexes* a corpus, *ranks* passages for a question and
*grounds* an answer in the passages it selected. Hence three verbs:

``fit(documents, queries=None)``  build the index (and tune whatever must be tuned),
``retrieve(question, k)``         rank passages, best first,
``answer(question, k)``           produce a grounded, cited answer or abstain.

Two behaviours are contractual rather than optional:

* a model that has not been fitted refuses to retrieve (no silently wrong ranking);
* an answer always carries its citations, and an answer whose best passage scores below the
  configured abstention threshold is **empty** — abstention is a first-class outcome, because a
  support assistant that invents an answer costs more than one that says "je ne sais pas".
"""

from __future__ import annotations

import platform
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from src.models.llm import BaseLLM, ExtractiveLLM, Passage, build_llm
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Libraries whose version is archived in the model card when they are already imported.
TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "scipy",
    "joblib",
    "langchain_core",
    "langchain",
    "transformers",
    "torch",
    "spacy",
    "hydra",
    "pandera",
    "pydantic",
)


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 (second precision)."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def library_versions() -> dict[str, str]:
    """Collect the versions of the libraries already loaded in this process.

    Returns:
        Mapping of library name to version, always containing ``python`` and ``platform``.
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in TRACKED_LIBRARIES:
        module = sys.modules.get(name)
        if module is None:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _sanitise(value: Any) -> Any:
    """Convert NumPy/pandas scalars to JSON-serialisable Python objects.

    Args:
        value: Arbitrary payload (metrics, parameters, ...).

    Returns:
        A JSON-safe copy of ``value``.
    """
    import numpy as np

    if isinstance(value, Mapping):
        return {str(key): _sanitise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitise(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_sanitise(item) for item in value.tolist()]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@dataclass(slots=True)
class RetrievedChunk:
    """One passage returned by the retriever.

    Attributes:
        chunk_id: Identifier of the passage.
        doc_id: Identifier of the source document.
        text: Passage text.
        score: Retrieval score (higher is better). Its scale depends on the stack, which is why
            the report always states which scorer produced it.
        rank: Rank in the returned list (1 = best).
        title: Title of the source document, when known.
        section: Editorial section of the source document, when known.
        metadata: Extra payload (per-feature explanation, document date, ...).
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int = 1
    title: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_passage(self) -> Passage:
        """Convert to the lightweight object handed to a generator."""
        return Passage(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            text=self.text,
            score=float(self.score),
            rank=int(self.rank),
            title=self.title,
        )

    def to_row(self) -> dict[str, Any]:
        """Return a flat, JSON-friendly row (used by the reports and notebooks)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "rank": int(self.rank),
            "score": round(float(self.score), 6),
            "title": self.title,
            "section": self.section,
            "n_chars": len(self.text),
        }


@dataclass(slots=True)
class Answer:
    """A grounded answer, or an explicit abstention.

    Attributes:
        query_id: Identifier of the question.
        question: Question asked.
        text: Answer text (empty when the system abstained).
        passages: Passages retrieved for the question, best first.
        cited_chunk_ids: Passages the answer actually cites.
        abstained: Whether the system refused to answer.
        top_score: Score of the best retrieved passage.
        latency_ms: End-to-end latency of the answer, in milliseconds.
        rationale: Why the generator produced this answer (traceability).
    """

    query_id: str
    question: str
    text: str
    passages: list[RetrievedChunk] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False
    top_score: float = 0.0
    latency_ms: float = 0.0
    rationale: str = ""

    @property
    def citations(self) -> list[str]:
        """Return the citations in ``chunk_id|doc_id`` form (as written to the CSV artefact)."""
        index = {passage.chunk_id: passage.doc_id for passage in self.passages}
        return [f"{chunk_id}|{index.get(chunk_id, '')}" for chunk_id in self.cited_chunk_ids]

    def to_row(self, *, n_chunks_indexed: int) -> dict[str, Any]:
        """Return the prediction row persisted by the inference pipeline."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "answer": self.text,
            "citations": ",".join(self.citations),
            "n_citations": len(self.cited_chunk_ids),
            "abstained": int(self.abstained),
            "top_score": round(float(self.top_score), 6),
            "n_chunks_indexed": int(n_chunks_indexed),
            "latency_ms": round(float(self.latency_ms), 3),
        }


@dataclass
class FitResult:
    """What one index build produced.

    Attributes:
        model_name: Class name of the model (``TfidfModel``, ``LangChainModel``, ...).
        algorithm: Algorithm identifier resolved by the factory.
        metrics: Metrics computed on the validation questions (empty when none were provided).
        n_documents: Number of indexed documents.
        n_chunks: Number of indexed passages.
        duration_seconds: Wall-clock duration of the fit.
        started_at: ISO timestamp of the beginning of the fit.
        finished_at: ISO timestamp of the end of the fit.
        params: Effective hyper-parameters.
        extra: Framework specific payload (vocabulary size, embedding dimension, ...).
    """

    model_name: str
    algorithm: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    n_documents: int = 0
    n_chunks: int = 0
    duration_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return _sanitise(asdict(self))

    def to_json(self, path: str | Path) -> Path:
        """Write the result as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FitResult:
        """Rebuild a result from its JSON payload (unknown keys are ignored).

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed result.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{str(key): value for key, value in payload.items() if key in known})


@dataclass
class ModelCard:
    """Traceability document archived next to the model artefact.

    Attributes:
        model_name: Class name of the model.
        framework: Framework identifier (``tfidf``, ``langchain``, ...).
        algorithm: Algorithm identifier.
        task: Learning task (``retrieval``, ``generation``, ...).
        params: Effective hyper-parameters.
        metrics: Metrics of the run.
        library_versions: Versions of the libraries used, for reproducibility.
        created_at: ISO timestamp of the card creation.
        n_documents: Number of indexed documents.
        n_chunks: Number of indexed passages.
        artifact: File name of the serialised artefact, when known.
        llm: Description of the generator used to answer.
        notes: Human readable remarks (known limitations, guards, ...).
    """

    model_name: str
    framework: str
    algorithm: str = ""
    task: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=library_versions)
    created_at: str = field(default_factory=_utc_now)
    n_documents: int = 0
    n_chunks: int = 0
    artifact: str | None = None
    llm: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return _sanitise(asdict(self))

    def to_json(self, path: str | Path) -> Path:
        """Write the card as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return write_json(path, self.to_dict())


class BaseModel(ABC):
    """Contract implemented by every text model of the repository."""

    #: Framework identifier, overridden by each stack.
    framework: ClassVar[str] = "base"

    #: Whether :meth:`answer` produces text (a pure retriever answers ``False``).
    supports_generation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        algorithm: str,
        params: Mapping[str, Any] | None = None,
        task: str = "retrieval",
        random_state: int = 42,
        config: Mapping[str, Any] | None = None,
        name: str = "",
        llm: BaseLLM | None = None,
    ) -> None:
        """Store the configuration of the model.

        Args:
            algorithm: Algorithm identifier resolved by the factory.
            params: Hyper-parameters (everything the stack needs, and nothing hard-coded).
            task: Learning task served by the model.
            random_state: Seed used by the stack.
            config: Full application configuration (read for the LLM node and the abstention
                threshold), kept for traceability.
            name: Human readable name used in logs and reports.
            llm: Generator to use; when omitted, one is built from the configuration.
        """
        self.algorithm = str(algorithm)
        self.params: dict[str, Any] = dict(params or {})
        self.task = str(task)
        self.random_state = int(random_state)
        self.config: dict[str, Any] = dict(config or {})
        self.name = str(name or self.algorithm)
        self.fit_result_: FitResult | None = None
        self._is_fitted = False
        self.llm: BaseLLM = llm or self._build_llm()
        self.abstention_threshold = float(self.params.get("abstention_threshold", 0.0))

    # ------------------------------------------------------------------ propriétés --------
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted (indexed)."""
        return self._is_fitted

    @property
    def default_model_file(self) -> str:
        """Default artefact file name of this stack."""
        return str(self.params.get("model_file", "model.joblib"))

    @property
    def chunks(self) -> pd.DataFrame:
        """The indexed passages, one row per passage (empty before the fit).

        The frame is exposed so that the pipelines can persist it
        (``data/processed/chunks.parquet``)
        and the notebooks can inspect what was actually indexed — a retriever is only as good as
        its chunking, and that is not visible from a score alone.
        """
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "doc_id",
                "chunk_index",
                "start_char",
                "end_char",
                "n_tokens",
                "text",
            ]
        )

    # ------------------------------------------------------------------ cycle de vie ------
    def fit(
        self,
        documents: pd.DataFrame,
        queries: pd.DataFrame | None = None,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> FitResult:
        """Build the index from a corpus, timing the operation.

        Args:
            documents: Reference corpus (``doc_id``, ``text``, metadata columns).
            queries: Optional annotated questions, used to tune what must be tuned (never the
                test split).
            callbacks: Training callbacks, fired once per documented stage.
            context: Mutable callback context.

        Returns:
            The :class:`FitResult` of the run.
        """
        started_at = _utc_now()
        started = time.perf_counter()
        self._seed_everything()
        metrics = self._fit(documents, queries, callbacks=callbacks, context=context)
        duration = time.perf_counter() - started
        self._is_fitted = True
        result = FitResult(
            model_name=type(self).__name__,
            algorithm=self.algorithm,
            metrics={str(key): float(value) for key, value in (metrics or {}).items()},
            n_documents=len(documents),
            n_chunks=int(self._n_chunks()),
            duration_seconds=duration,
            started_at=started_at,
            finished_at=_utc_now(),
            params=self._effective_params(),
            extra=self._extra_metadata(),
        )
        self.fit_result_ = result
        logger.info(
            "Model fitted | {} | documents={} | chunks={} | {:.2f}s",
            self.summary(),
            result.n_documents,
            result.n_chunks,
            duration,
        )
        return result

    def retrieve(
        self,
        question: str,
        k: int = 5,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Rank the indexed passages for a question.

        Args:
            question: User question.
            k: Number of passages to return.
            filters: Optional metadata filters (``{"section": "procedure"}``), applied *before*
                ranking, which is how a real assistant restricts a search space.

        Returns:
            The passages, best first; shorter than ``k`` when the index is smaller.

        Raises:
            RuntimeError: When the model has not been fitted.
            ValueError: When ``k`` is not strictly positive or the question is empty.
        """
        self.check_is_fitted()
        if k <= 0:
            msg = f"k must be > 0, got {k}"
            raise ValueError(msg)
        if not str(question).strip():
            msg = "question must not be empty"
            raise ValueError(msg)
        retrieved = list(self._retrieve(str(question), int(k), dict(filters or {})))
        for rank, chunk in enumerate(retrieved, start=1):
            chunk.rank = rank
            chunk.score = float(chunk.score)
        return retrieved

    def answer(self, question: str, k: int = 5, *, query_id: str = "") -> Answer:
        """Retrieve, then answer from the retrieved passages (or abstain).

        Args:
            question: User question.
            k: Number of passages retrieved before generating.
            query_id: Identifier of the question (defaults to a generated one).

        Returns:
            The grounded :class:`Answer`.
        """
        started = time.perf_counter()
        identifier = query_id or f"QRY-{abs(hash(question)) % 10000:04d}"
        passages = self.retrieve(question, k)
        top_score = max((passage.score for passage in passages), default=0.0)
        if not passages or self._must_abstain(top_score):
            return Answer(
                query_id=identifier,
                question=question,
                text="",
                passages=passages,
                abstained=True,
                top_score=top_score,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                rationale=(
                    "no passage above the abstention threshold "
                    f"({top_score:.4f} < {self.abstention_threshold:.4f})"
                ),
            )
        generation = self.llm.generate(question, [passage.to_passage() for passage in passages])
        abstained = not generation.text.strip()
        return Answer(
            query_id=identifier,
            question=question,
            text=generation.text,
            passages=passages,
            cited_chunk_ids=list(generation.cited_chunk_ids),
            abstained=abstained,
            top_score=top_score,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            rationale=generation.rationale,
        )

    # ------------------------------------------------------------------ traçabilité -------
    def model_card(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact: str | None = None,
        notes: Sequence[str] | None = None,
    ) -> ModelCard:
        """Build the traceability document of this model.

        Args:
            metrics: Metrics to archive (the caller decides which ones).
            artifact: Serialised artefact file name, when known.
            notes: Extra human readable remarks.

        Returns:
            The :class:`ModelCard`.
        """
        result = self.fit_result_
        return ModelCard(
            model_name=type(self).__name__,
            framework=self.framework,
            algorithm=self.algorithm,
            task=self.task,
            params=self._effective_params(),
            metrics={str(key): float(value) for key, value in dict(metrics or {}).items()},
            library_versions=library_versions(),
            created_at=_utc_now(),
            n_documents=int(result.n_documents) if result else 0,
            n_chunks=int(result.n_chunks) if result else self._n_chunks(),
            artifact=artifact,
            llm=self.llm.describe(),
            notes=[str(note) for note in (notes or [])],
        )

    def summary(self) -> str:
        """One-line description used in logs and reports."""
        return f"{self.name} ({self.framework}/{self.algorithm}, task={self.task})"

    def check_is_fitted(self) -> None:
        """Raise when the model has not been fitted yet.

        Raises:
            RuntimeError: Always when the model is not fitted.
        """
        if not self._is_fitted:
            msg = (
                f"{type(self).__name__} is not fitted: call fit(documents) (or load an artefact) "
                "before retrieving."
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ à implémenter -----
    @abstractmethod
    def _fit(
        self,
        documents: pd.DataFrame,
        queries: pd.DataFrame | None,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Build the index (implemented by each stack).

        Args:
            documents: Reference corpus.
            queries: Optional annotated questions.
            callbacks: Training callbacks.
            context: Mutable callback context.

        Returns:
            Validation metrics (possibly empty).
        """

    @abstractmethod
    def _retrieve(
        self, question: str, k: int, filters: Mapping[str, Any]
    ) -> Sequence[RetrievedChunk]:
        """Rank passages for a question (implemented by each stack).

        Args:
            question: User question.
            k: Number of passages to return.
            filters: Metadata filters.

        Returns:
            The passages, best first.
        """

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the fitted model.

        Args:
            path: Destination file (a directory receives :attr:`default_model_file`).

        Returns:
            The written path.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Reload a persisted model.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration used to restore runtime options.

        Returns:
            The reloaded model, ready to retrieve.
        """

    # ------------------------------------------------------------------ helpers -----------
    def _build_llm(self) -> BaseLLM:
        """Instantiate the configured generator."""
        node: Mapping[str, Any] = {}
        model_node = self.config.get("model")
        if isinstance(model_node, Mapping):
            candidate = model_node.get("llm")
            if isinstance(candidate, Mapping):
                node = candidate
        if not node:
            return ExtractiveLLM()
        return build_llm(node)

    def _state_to_save(self) -> dict[str, Any]:
        """Learned state that every stack artefact must carry.

        The abstention threshold is *learned* during ``fit`` (calibration split); an artefact that
        forgets it silently answers every question after a reload, and the evaluation report then
        measures a different decision rule from the one that was trained.

        Returns:
            The key/value pairs to merge into the artefact payload.
        """
        return {"abstention_threshold": float(self.abstention_threshold)}

    def _state_from_payload(self, payload: Mapping[str, Any]) -> None:
        """Restore the state written by :meth:`_state_to_save`.

        Args:
            payload: Artefact payload read from disk.
        """
        if "abstention_threshold" in payload:
            self.abstention_threshold = float(payload["abstention_threshold"])

    def _must_abstain(self, top_score: float) -> bool:
        """Whether the best score is below the abstention threshold."""
        return bool(self.abstention_threshold > 0.0 and top_score < self.abstention_threshold)

    def _n_chunks(self) -> int:
        """Number of indexed passages (0 before the fit)."""
        return len(self.chunks)

    def _extra_metadata(self) -> dict[str, Any]:
        """Framework specific payload archived in the fit result."""
        return {}

    def _effective_params(self) -> dict[str, Any]:
        """Copy of the resolved hyper-parameters."""
        return dict(self.params)

    def _resolve_path(self, path: str | Path) -> Path:
        """Normalise a persistence path (a directory receives the default file name).

        Args:
            path: File or directory.

        Returns:
            The destination file, with its parent directory created.
        """
        destination = Path(path)
        if destination.is_dir() or not destination.suffix:
            destination = destination / self.default_model_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _seed_everything(self) -> None:
        """Seed Python, NumPy and the project's global state before fitting."""
        try:
            from src.utils.utils import set_seed

            set_seed(self.random_state)
        except ImportError:  # pragma: no cover - project without the utils layer
            import numpy as np

            np.random.seed(self.random_state)


__all__ = [
    "Answer",
    "BaseModel",
    "FitResult",
    "ModelCard",
    "RetrievedChunk",
    "library_versions",
]
