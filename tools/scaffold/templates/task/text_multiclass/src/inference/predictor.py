"""Service de classification : un texte entre, un libellé et sa justification sortent.

C'est le chemin de *serving* du projet, et il est court par construction : recharger l'artefact,
classer, écrire. Deux choses valent d'être dites parce qu'elles sont mesurées plutôt que
supposées :

* la **latence publiée est une latence chaude** — l'artefact est déjà chargé — et les textes sont
  classés **un par un**, comme un service les reçoit ; un lot entier mesurerait un débit, pas un
  temps de réponse ;
* chaque prédiction peut être **expliquée** (:meth:`explain`) : les termes qui ont pesé, avec leur
  poids. Un classifieur linéaire qui refuserait de dire pourquoi serait un oracle, et un service
  qui ne peut pas expliquer une décision ne peut pas être discuté par le métier qui la reçoit.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schemas import probability_columns
from src.models.contract import BaseTextClassifier
from src.utils.io import read_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class TextClassificationPredictor:
    """Classify tickets with a trained artefact, one row at a time."""

    def __init__(
        self,
        model: BaseTextClassifier,
        *,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        text_column: str = "text",
        explain_top: int = 3,
    ) -> None:
        """Configure the predictor.

        Args:
            model: Trained classifier.
            config: Full application configuration (predict node included).
            paths: Project filesystem layout.
            text_column: Column holding the text to classify.
            explain_top: Number of terms kept in the explanation of each prediction.
        """
        self.model = model
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.text_column = str(text_column)
        self.explain_top = max(int(explain_top), 0)
        self.latencies_ms: list[float] = []

    # ------------------------------------------------------------------ entrée ------------
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Classify every row of a frame.

        Args:
            frame: Frame holding the text column (an optional ``doc_id`` and an optional
                reference label are carried through when present).

        Returns:
            A validated-ready prediction frame: identity, prediction, confidence, per-label
            probabilities, correctness when a reference label exists, and latency.

        Raises:
            ValueError: When the text column is missing or the frame is empty.
        """
        if frame.empty:
            msg = "Nothing to classify: the frame is empty"
            raise ValueError(msg)
        if self.text_column not in frame.columns:
            msg = (
                f"Column '{self.text_column}' missing from the input frame: "
                f"{sorted(frame.columns)}"
            )
            raise ValueError(msg)
        self.latencies_ms = []
        rows: list[dict[str, Any]] = []
        labels = self.model.labels
        for record in frame.to_dict(orient="records"):
            text = str(record[self.text_column])
            clock = time.perf_counter()
            probabilities = self.model.predict_proba([text])[0]
            latency = (time.perf_counter() - clock) * 1000.0
            self.latencies_ms.append(latency)
            position = int(np.argmax(probabilities))
            prediction = labels[position]
            expected = record.get("label")
            rows.append(
                {
                    "doc_id": str(record.get("doc_id", f"INF-{len(rows) + 1:04d}")),
                    "text": text,
                    "expected_label": None if expected is None else str(expected),
                    "prediction": prediction,
                    "confidence": float(probabilities[position]),
                    "correct": int(expected is not None and str(expected) == prediction),
                    "latency_ms": float(latency),
                    **{
                        name: float(probabilities[index])
                        for index, name in enumerate(probability_columns(tuple(labels)))
                    },
                }
            )
        logger.info(
            "Classified {} texts | mean latency {:.2f} ms | p95 {:.2f} ms",
            len(rows),
            float(np.mean(self.latencies_ms)),
            float(np.quantile(self.latencies_ms, 0.95)),
        )
        return pd.DataFrame(rows)

    def explain(self, texts: Sequence[str]) -> list[list[tuple[str, float]]]:
        """Explain the decisions of raw texts.

        Args:
            texts: Texts to explain.

        Returns:
            One list of ``(term, weight)`` pairs per text.
        """
        return self.model.explain(texts, k=self.explain_top)

    def load_input(self, path: str | Path | None = None) -> pd.DataFrame:
        """Read the inference payload (a file, or a sample of the corpus).

        Args:
            path: Explicit file to read (``predict.input`` when ``None``).

        Returns:
            The frame to classify.

        Raises:
            FileNotFoundError: When neither the file nor the corpus exists.
        """
        configured = path if path is not None else self._predict_node().get("input")
        if configured:
            source = Path(str(configured))
            if not source.is_absolute():
                source = self.paths.root / source
            frame = read_table(source)
            return frame

        from src.data.loaders import TextLabelLoader

        loader = TextLabelLoader(self.paths)
        for split in ("test", "val"):
            try:
                sample = loader.split(split)
            except (ValueError, FileNotFoundError):
                continue
            limit = int(self._predict_node().get("n_samples", 5))
            return sample.head(limit).reset_index(drop=True)
        msg = "No input available: generate the corpus first (mode=generate-data)"
        raise FileNotFoundError(msg)

    def _predict_node(self) -> dict[str, Any]:
        """Return the ``predict`` node of the configuration."""
        node = self.config.get("predict")
        return dict(node) if isinstance(node, Mapping) else {}


__all__ = ["TextClassificationPredictor"]
