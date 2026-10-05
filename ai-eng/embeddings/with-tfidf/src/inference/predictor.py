"""Serving path of a retrieval-augmented system.

The predictor is what a service would wrap: it loads nothing at call time, answers a batch of
questions, and writes two artefacts — the answers and, separately, the passages they rest on.
Keeping the passages in their own file is a deliberate choice: when a user disputes an answer,
the first question is *which passage produced it*, and that must be answerable without
re-running the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import write_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class Predictor:
    """Answer questions with a fitted model and persist the result.

    Attributes:
        model: Fitted model implementing the text contract.
        k: Number of passages retrieved before generating.
        paths: Project filesystem layout.
        written_artifacts: Files written by the last :meth:`predict` call.
    """

    def __init__(
        self,
        model: Any,
        *,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        k: int = 5,
    ) -> None:
        """Configure the predictor.

        Args:
            model: Fitted model.
            config: Application configuration (kept for traceability and future options).
            paths: Project filesystem layout.
            k: Number of passages retrieved before generating.
        """
        self.model = model
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.k = int(k)
        self.written_artifacts: list[str] = []

    def predict(self, questions: pd.DataFrame) -> pd.DataFrame:
        """Answer a batch of questions.

        Args:
            questions: Frame with at least a ``question`` column (``query_id`` is used when
                present, generated otherwise).

        Returns:
            One row per question, ready for
            :func:`src.data.schemas.validate_predictions`.

        Raises:
            ValueError: When the frame carries no ``question`` column.
        """
        if "question" not in questions.columns:
            msg = f"questions must contain a 'question' column (found: {sorted(questions.columns)})"
            raise ValueError(msg)
        n_chunks = len(getattr(self.model, "chunks", []))
        rows: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []
        for index, record in enumerate(questions.to_dict(orient="records")):
            query_id = str(record.get("query_id") or f"QRY-{9000 + index:04d}")
            answer = self.model.answer(str(record["question"]), self.k, query_id=query_id)
            rows.append(answer.to_row(n_chunks_indexed=max(n_chunks, 1)))
            for passage in answer.passages:
                passages.append(
                    {
                        "query_id": query_id,
                        "cited": int(passage.chunk_id in answer.cited_chunk_ids),
                        **passage.to_row(),
                    }
                )
        self.written_artifacts = []
        if passages:
            passages_path = write_table(
                pd.DataFrame(passages), self.paths.reports_dir / "retrieved_passages.csv"
            )
            self.written_artifacts.append(str(passages_path))
        logger.info(
            "Answered {} questions ({} passages retrieved, abstention {:.1%})",
            len(rows),
            len(passages),
            float(pd.DataFrame(rows)["abstained"].mean()) if rows else 0.0,
        )
        return pd.DataFrame(rows)

    def answer_one(self, question: str, *, query_id: str = "QRY-0000") -> dict[str, Any]:
        """Answer a single question (used by the notebooks and the tests).

        Args:
            question: Question to answer.
            query_id: Identifier attached to the answer.

        Returns:
            The prediction row.
        """
        frame = self.predict(pd.DataFrame([{"query_id": query_id, "question": question}]))
        return dict(frame.iloc[0].to_dict())

    def explain(self, question: str, *, k: int | None = None) -> pd.DataFrame:
        """Return the retrieved passages of a question with their scores.

        Args:
            question: Question to answer.
            k: Number of passages (defaults to the predictor's ``k``).

        Returns:
            One row per retrieved passage.
        """
        retrieved: Sequence[Any] = self.model.retrieve(question, int(k or self.k))
        return pd.DataFrame([passage.to_row() for passage in retrieved])


__all__ = ["Predictor", "Path"]
