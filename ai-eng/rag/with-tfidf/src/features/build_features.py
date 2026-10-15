"""Explicability features of a retrieved passage.

A retrieval score is a single number: "this passage scored 0.42". That number answers *what*
but never *why*, and an assistant that cannot explain its citations is unusable in a support
desk. This module computes, for a ``(question, passage)`` pair, the handful of quantities a
human reviewer actually inspects:

``token_overlap``    share of the question's content words present in the passage,
``idf_overlap``      same overlap, but weighted by term rarity (the discriminating one),
``exact_phrase``     whether the passage contains the longest question n-gram,
``title_match``      share of question words present in the document title,
``recency_days``     age of the source document,
``position_ratio``   where the passage sits inside its document (procedures answer early).

The features are computed post-hoc, from the retrieved passages only, so they never influence
the ranking: they explain it. The notebook 06 uses them to compare the two stacks' error
profiles, and the report prints them for the questions that failed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_date(value: Any) -> datetime | None:
    """Parse an arbitrary date-like value into a timezone-aware datetime.

    Args:
        value: Value coming from a pandas frame (Timestamp, string, None).

    Returns:
        The parsed datetime (UTC), or ``None`` when the value is missing or invalid.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return (
        None if pd.isna(parsed) else parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed
    )


def _ngrams(tokens: Sequence[str], size: int) -> set[str]:
    """Return the set of ``size``-grams of a token sequence."""
    if size <= 0 or len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


@dataclass
class ChunkFeatureBuilder:
    """Compute explicability features for ``(question, passage)`` pairs.

    Attributes:
        idf: Inverse document frequency per term, as learned by the lexical index. When it is
            missing the rarity-weighted overlap falls back on the plain overlap, which is
            documented behaviour rather than a silent default.
        reference_date: Date used to compute the age of a document (defaults to "now", which
            makes the feature non-deterministic across days; pipelines pass the dataset's
            reference date instead).
        max_phrase: Longest question n-gram searched verbatim in the passage.
    """

    idf: Mapping[str, float] = field(default_factory=dict)
    reference_date: datetime | None = None
    max_phrase: int = 3

    def build(
        self,
        question_tokens: Sequence[str],
        chunk: Mapping[str, Any],
        document: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute the features of one ``(question, passage)`` pair.

        Args:
            question_tokens: Tokenised question (content words only).
            chunk: Passage row (``text``, ``chunk_index``, ``n_tokens`` at least).
            document: Source document row (``title``, ``published_at``, ``n_tokens``), optional.

        Returns:
            A mapping of feature name to value, JSON friendly.
        """
        text = str(chunk.get("text", ""))
        passage_tokens = text.lower().split()
        passage_set = set(passage_tokens)
        question_set = set(question_tokens)
        shared = question_set & passage_set

        token_overlap = len(shared) / len(question_set) if question_set else 0.0
        question_mass = sum(self.idf.get(token, 1.0) for token in question_set)
        shared_mass = sum(self.idf.get(token, 1.0) for token in shared)
        idf_overlap = shared_mass / question_mass if question_mass > 0 else token_overlap

        exact_phrase = 0.0
        normalised_text = " ".join(passage_tokens)
        for size in range(min(self.max_phrase, len(question_tokens)), 0, -1):
            grams = _ngrams([token.lower() for token in question_tokens], size)
            if grams and any(gram in normalised_text for gram in grams):
                exact_phrase = float(size)
                break

        title_match = 0.0
        recency_days = float("nan")
        if document is not None:
            title_tokens = set(str(document.get("title", "")).lower().split())
            title_match = (
                len(question_set & title_tokens) / len(question_set) if question_set else 0.0
            )
            published = _parse_date(document.get("published_at"))
            reference = self.reference_date or datetime.now(tz=timezone.utc)
            if published is not None:
                recency_days = max((reference - published).days, 0)

        chunk_index = float(chunk.get("chunk_index", 0) or 0)
        document_tokens = float((document or {}).get("n_tokens", 0) or 0)
        position_ratio = (
            min(chunk_index * float(chunk.get("n_tokens", 0) or 0) / document_tokens, 1.0)
            if document_tokens > 0
            else 0.0
        )
        return {
            "token_overlap": round(float(token_overlap), 4),
            "idf_overlap": round(float(idf_overlap), 4),
            "exact_phrase_len": exact_phrase,
            "title_match": round(float(title_match), 4),
            "recency_days": (
                float("nan") if recency_days != recency_days else round(recency_days, 1)
            ),
            "position_ratio": round(float(position_ratio), 4),
            "n_tokens": int(chunk.get("n_tokens", 0) or 0),
        }

    def build_frame(
        self,
        question_tokens: Sequence[str],
        retrieved: Sequence[Mapping[str, Any]],
        documents: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute the features of every passage retrieved for one question.

        Args:
            question_tokens: Tokenised question.
            retrieved: Retrieved passage rows (must carry ``doc_id`` and ``chunk_index``).
            documents: Corpus frame, used to attach the source document metadata.

        Returns:
            One row per passage, features first, with the retrieval rank as index.
        """
        index_by_doc: dict[str, Mapping[str, Any]] = {}
        if documents is not None and not documents.empty:
            index_by_doc = {
                str(record["doc_id"]): record for record in documents.to_dict(orient="records")
            }
        rows = [
            self.build(question_tokens, chunk, index_by_doc.get(str(chunk.get("doc_id"))))
            for chunk in retrieved
        ]
        frame = pd.DataFrame(rows)
        frame.insert(0, "rank", range(1, len(frame) + 1))
        return frame


__all__ = ["ChunkFeatureBuilder"]
