"""The preprocessing pipeline of a text project, assembled from the configuration.

The pipeline is the object the rest of the code talks to: it knows how to turn a corpus of
documents into the passage table that will be indexed, and how to tokenise a query exactly like
the corpus was tokenised. Keeping both in one place is what prevents the classic retrieval bug
where the query is normalised differently from the documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.transformers import (
    DocumentChunker,
    TextPreprocessingPipeline,
    tokenize,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Chunking and tokenisation, configured by Hydra and persisted with the model.

    Attributes:
        pipeline: The configured :class:`TextPreprocessingPipeline`.
    """

    def __init__(self, pipeline: TextPreprocessingPipeline | None = None) -> None:
        """Store the underlying pipeline.

        Args:
            pipeline: Configured pipeline (defaults to the template defaults).
        """
        self.pipeline = pipeline or TextPreprocessingPipeline()

    @classmethod
    def from_config(cls, config: Any) -> TextPreprocessor:
        """Build the preprocessor from the ``preprocessing`` node of the configuration.

        Args:
            config: ``preprocessing`` node (``chunking`` sub-node is used).

        Returns:
            The configured preprocessor.
        """
        return cls(TextPreprocessingPipeline.from_config(config))

    @property
    def chunker(self) -> DocumentChunker:
        """The configured chunker."""
        return self.pipeline.chunker

    def prepare_corpus(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Turn a corpus into the passage table that will be indexed.

        Args:
            documents: Corpus with at least the chunker's identifier column and ``text``.

        Returns:
            The passage table (``chunk_id``, ``doc_id``, ``chunk_index``, offsets, tokens, text).
        """
        chunks = self.pipeline.transform_documents(documents)
        logger.info(
            "Corpus prepared: {} documents -> {} passages (mean {:.1f} passages/document)",
            len(documents),
            len(chunks),
            len(chunks) / max(len(documents), 1),
        )
        return chunks

    def tokenize(self, text: str) -> list[str]:
        """Tokenise one text with the corpus tokeniser.

        Args:
            text: Text to tokenise.

        Returns:
            The filtered token list.
        """
        return self.pipeline.stop_words.transform(tokenize(text))

    def tokenize_many(self, texts: list[str]) -> list[list[str]]:
        """Tokenise several texts.

        Args:
            texts: Texts to tokenise.

        Returns:
            One token list per text.
        """
        return self.pipeline.transform(texts)

    def save(self, path: str | Path) -> Path:
        """Persist the preprocessing configuration.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        return self.pipeline.save(path)

    @classmethod
    def load(cls, path: str | Path) -> TextPreprocessor:
        """Reload a persisted preprocessor.

        Args:
            path: Artefact written by :meth:`save`.

        Returns:
            The restored preprocessor.
        """
        return cls(TextPreprocessingPipeline.load(path))


__all__ = ["TextPreprocessor"]
