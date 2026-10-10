"""Contract of the family-specific synthetic corpus generator.

The repository is built on synthetic data: no download, no confidential content, and a generator
that *knows the ground truth* — which documents are relevant to which question, and which
questions have no answer at all. That knowledge is what makes the evaluation of a retrieval
system meaningful instead of circular.

Each family ships its own ``src/data/generators.py`` implementing this contract; the pipelines
only know the contract, so a new family never touches the modality layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class GeneratedCorpus:
    """The three tables a text project needs, plus the metadata of the generation.

    Attributes:
        documents: Reference corpus (one row per document).
        queries: Annotated questions (one row per question, with its split).
        metadata: Generation metadata (seeds, sizes, latent parameters, known ceilings). It is
            archived next to the corpus and reused by the report, which can therefore state the
            *achievable* ceiling of a metric instead of implying that 1.0 is reachable.
    """

    documents: pd.DataFrame
    queries: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCorpusGenerator(ABC):
    """Interface implemented by every family generator."""

    #: Name of the generated dataset (used in logs and artefacts).
    dataset_name: str = "corpus"

    @classmethod
    @abstractmethod
    def from_config(cls, config: Any, *, seed: int | None = None) -> BaseCorpusGenerator:
        """Build a generator from the ``data`` node of the configuration.

        Args:
            config: ``data`` configuration node.
            seed: Optional seed override (defaults to the configuration seed).

        Returns:
            The configured generator.
        """

    @abstractmethod
    def generate(self) -> GeneratedCorpus:
        """Generate the corpus, the questions and the metadata.

        Returns:
            The :class:`GeneratedCorpus`.
        """


__all__ = ["BaseCorpusGenerator", "GeneratedCorpus"]
