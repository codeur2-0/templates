"""Loading and persistence of the text datasets.

The loader is the *only* place that knows where a dataset lives and in which format it is
stored. Everything else (pipelines, evaluator, notebooks) asks for a validated
``DataFrame``: a corpus path that does not exist, a split that does not exist or a frame
that violates its Pandera contract fails here, with a message naming the file and the column.

Nothing is cached implicitly. Text corpora are small enough (a few megabytes) that an
explicit read is cheaper than a cache invalidation bug.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schemas import (
    ChunksSchema,
    DocumentsSchema,
    QueriesSchema,
    validate_chunks,
    validate_documents,
    validate_queries,
)
from src.utils.io import read_json, read_table, write_json, write_table, write_table_multiple
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Name of the metadata file written by the generator next to the corpus.
METADATA_FILE = "generation_metadata.json"


class TextCorpusLoader:
    """Read and write the documents, chunks and questions of a text project.

    Attributes:
        paths: Project filesystem layout.
        formats: Formats written for every table (``parquet`` for machines, ``csv`` for humans).
        validation_enabled: Whether the Pandera contracts are enforced on load and save.
        lazy_validation: Collect every violation instead of stopping at the first one.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        *,
        formats: Sequence[str] = ("parquet", "csv"),
        validation_enabled: bool = True,
        lazy_validation: bool = False,
    ) -> None:
        """Configure the loader.

        Args:
            paths: Filesystem layout (defaults to the project layout detected from ``__file__``).
            formats: Formats written by :meth:`save_documents` and friends.
            validation_enabled: Enforce the Pandera contracts.
            lazy_validation: Collect every contract violation before raising.
        """
        self.paths = paths or ProjectPaths.from_root()
        self.formats = tuple(formats)
        self.validation_enabled = validation_enabled
        self.lazy_validation = lazy_validation

    # ------------------------------------------------------------------ chemins -----------
    @property
    def documents_path(self) -> Path:
        """Expected path of the corpus table (Parquet)."""
        return self.paths.raw_dir / "documents.parquet"

    @property
    def queries_path(self) -> Path:
        """Expected path of the annotated questions table (Parquet)."""
        return self.paths.raw_dir / "queries.parquet"

    @property
    def chunks_path(self) -> Path:
        """Expected path of the chunked corpus (Parquet)."""
        return self.paths.processed_dir / "chunks.parquet"

    @property
    def metadata_path(self) -> Path:
        """Expected path of the generation metadata (JSON)."""
        return self.paths.raw_dir / METADATA_FILE

    # ------------------------------------------------------------------ corpus ------------
    def load_documents(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the reference corpus.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The corpus, one row per document.

        Raises:
            FileNotFoundError: When the corpus has not been generated yet.
        """
        frame = self._read(self.documents_path, "corpus")
        if self._should_validate(validate):
            frame = validate_documents(frame, lazy=self.lazy_validation)
        logger.debug("Loaded {} documents from {}", len(frame), self.documents_path)
        return frame

    def save_documents(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the corpus in every configured format.

        Args:
            frame: Corpus to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = validate_documents(frame, lazy=self.lazy_validation) if self.validation_enabled else frame
        written = write_table_multiple(payload, self.documents_path, self.formats)
        logger.info("Corpus written: {} documents -> {}", len(payload), sorted(written))
        return written

    # ------------------------------------------------------------------ passages ----------
    def load_chunks(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the chunked corpus (the unit actually ranked by a retriever).

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The chunks, one row per indexed passage.
        """
        frame = self._read(self.chunks_path, "chunks")
        if self._should_validate(validate):
            frame = validate_chunks(frame, lazy=self.lazy_validation)
        logger.debug("Loaded {} chunks from {}", len(frame), self.chunks_path)
        return frame

    def save_chunks(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the chunked corpus.

        Args:
            frame: Chunks to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = validate_chunks(frame, lazy=self.lazy_validation) if self.validation_enabled else frame
        written = write_table_multiple(payload, self.chunks_path, self.formats)
        logger.info("Chunks written: {} passages -> {}", len(payload), sorted(written))
        return written

    # ------------------------------------------------------------------ questions ---------
    def load_queries(self, split: str | None = None, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the annotated questions, optionally restricted to one split.

        Args:
            split: ``calibration``, ``val``, ``test`` or ``None`` for every split.
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The questions, one row per question.

        Raises:
            ValueError: When the requested split does not exist in the file.
        """
        frame = self._read(self.queries_path, "questions")
        if self._should_validate(validate):
            frame = validate_queries(frame, lazy=self.lazy_validation)
        if split is None:
            return frame
        available = sorted(frame["split"].unique().tolist())
        if split not in available:
            msg = f"Unknown split '{split}'. Available in {self.queries_path}: {available}"
            raise ValueError(msg)
        selected = frame.loc[frame["split"] == split].reset_index(drop=True)
        logger.debug("Split '{}' holds {} questions", split, len(selected))
        return selected

    def save_queries(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the annotated questions.

        Args:
            frame: Questions to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = validate_queries(frame, lazy=self.lazy_validation) if self.validation_enabled else frame
        written = write_table_multiple(payload, self.queries_path, self.formats)
        logger.info("Questions written: {} rows -> {}", len(payload), sorted(written))
        return written

    def load_inference_questions(self, path: str | Path) -> pd.DataFrame:
        """Load an arbitrary question file provided by the user (``predict`` mode).

        Only the ``question`` column is required: an inference payload is a question, not an
        annotated example. Missing identifiers are generated on the fly.

        Args:
            path: CSV, Parquet or JSON file holding at least a ``question`` column.

        Returns:
            A frame with ``query_id`` and ``question`` columns.

        Raises:
            ValueError: When the file carries no ``question`` column.
        """
        file_path = Path(path)
        if file_path.suffix.lower() == ".json":
            payload = read_json(file_path)
            frame = pd.DataFrame(payload if isinstance(payload, list) else [payload])
        else:
            frame = read_table(file_path)
        if "question" not in frame.columns:
            msg = (
                f"Inference file {file_path} must contain a 'question' column "
                f"(found: {sorted(frame.columns)})"
            )
            raise ValueError(msg)
        if "query_id" not in frame.columns:
            frame = frame.assign(
                query_id=[f"QRY-{index:04d}" for index in range(9000, 9000 + len(frame))]
            )
        return frame.loc[:, ["query_id", "question"]]

    # ------------------------------------------------------------------ métadonnées -------
    def load_metadata(self) -> dict[str, Any]:
        """Load the generation metadata written by the synthetic generator.

        Returns:
            The metadata mapping, or an empty mapping when the file is absent (older run).
        """
        if not self.metadata_path.exists():
            logger.warning("No generation metadata found at {}", self.metadata_path)
            return {}
        payload = read_json(self.metadata_path)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def save_metadata(self, payload: Mapping[str, Any]) -> Path:
        """Persist the generation metadata next to the corpus.

        Args:
            payload: Metadata mapping.

        Returns:
            The written path.
        """
        return write_json(self.metadata_path, dict(payload))

    # ------------------------------------------------------------------ internes ----------
    def _read(self, path: Path, label: str) -> pd.DataFrame:
        """Read a table, falling back on CSV when only the human format is present."""
        if path.exists():
            return read_table(path)
        csv_candidate = path.with_suffix(".csv")
        if csv_candidate.exists():
            logger.warning("{} not found, falling back on {}", path.name, csv_candidate.name)
            return read_table(csv_candidate)
        msg = (
            f"{label.capitalize()} not found: {path}. "
            "Run `make data` (python -m src.main mode=generate-data) first."
        )
        raise FileNotFoundError(msg)

    def _should_validate(self, override: bool | None) -> bool:
        """Resolve the validation flag for one call."""
        return self.validation_enabled if override is None else override

    def write_chunks_csv_only(self, frame: pd.DataFrame) -> Path:
        """Write a single CSV copy of the chunks (used by the inference report).

        Args:
            frame: Chunks to persist.

        Returns:
            The written path.
        """
        return write_table(frame, self.chunks_path.with_suffix(".csv"))


__all__ = ["METADATA_FILE", "TextCorpusLoader", "ChunksSchema", "DocumentsSchema", "QueriesSchema"]
