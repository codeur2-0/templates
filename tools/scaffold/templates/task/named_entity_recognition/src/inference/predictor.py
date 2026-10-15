"""Service d'extraction : un lot de messages entre, une table de mentions sort.

C'est le chemin de *serving* du projet, et il est court par construction : recharger l'artefact,
extraire, écrire. Trois choses valent d'être dites parce qu'elles sont mesurées plutôt que
supposées :

* la **latence publiée est une latence chaude** — l'artefact est déjà chargé — et les messages sont
  traités par la boucle partagée de :mod:`src.inference.extraction`, un par un, comme un service les
  reçoit : un lot entier mesurerait un débit, pas un temps de réponse ;
* chaque mention publiée porte sa **provenance** (``regle`` ou ``modele``) et sa **confiance** : le
  service ne rend pas un score opaque, il rend une décision que le conseiller peut relire ;
* les mentions ne se **chevauchent jamais** et sont triées par position de lecture : le back-office
  remplit un dossier avec une liste, pas avec un graphe d'ambiguïtés.

Le service écrit une table de mentions (une ligne par entité) et un résumé par message : les
longueurs et latences servent tous les deux, la première pour remplir un dossier, la seconde pour
piloter la qualité du service.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.extraction import MENTION_COLUMNS, extract_one_by_one
from src.models.contract import BaseEntityTagger
from src.training.metrics import latency_stats
from src.utils.io import read_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class EntityPredictor:
    """Extract the entities of incoming messages with a trained artefact."""

    def __init__(
        self,
        model: BaseEntityTagger,
        *,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        text_column: str = "text",
        id_column: str = "msg_id",
    ) -> None:
        """Configure the predictor.

        Args:
            model: Trained extractor.
            config: Full application configuration (the ``predict`` node is read when present).
            paths: Project filesystem layout.
            text_column: Column holding the text to annotate.
            id_column: Column holding the message identifier.
        """
        self.model = model
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.text_column = str(text_column)
        self.id_column = str(id_column)
        self.latencies_ms: list[float] = []

    def predict(self, frame: pd.DataFrame, *, spans: pd.DataFrame | None = None) -> pd.DataFrame:
        """Extract the entities of every message of a frame.

        Args:
            frame: Frame holding the text column.
            spans: Optional reference annotations, used to fill ``expected_label`` and ``correct``.

        Returns:
            The mention table: identity, offsets, type, surface, provenance, confidence, latency,
            and the verdict when the reference is known.

        Raises:
            ValueError: When the frame is empty or a required column is missing.
        """
        if frame.empty:
            msg = "Nothing to annotate: the frame is empty"
            raise ValueError(msg)
        mentions, self.latencies_ms = extract_one_by_one(
            self.model,
            frame,
            text_column=self.text_column,
            id_column=self.id_column,
        )
        if spans is not None and not mentions.empty:
            mentions = self._with_reference(mentions, spans)
        if not mentions.empty:
            identifiers = [str(value) for value in frame[self.id_column]]
            mentions["latency_ms"] = _expand_latencies(mentions, identifiers, self.latencies_ms)
            extra = [name for name in mentions.columns if name not in MENTION_COLUMNS]
            mentions = mentions.loc[:, [*MENTION_COLUMNS, *extra]]
        logger.info(
            "Annotated {} messages | {} mentions | mean latency {:.2f} ms | p95 {:.2f} ms",
            len(frame),
            len(mentions),
            float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0,
            float(np.quantile(self.latencies_ms, 0.95)) if self.latencies_ms else 0.0,
        )
        return mentions

    def summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return one row per message: how long it is, and how long the extraction took.

        Args:
            frame: The frame that was annotated.

        Returns:
            A summary table with the identifier, the number of whitespace-separated tokens and the
            latency of the message, in milliseconds.
        """
        identifiers = [str(value) for value in frame[self.id_column]]
        texts = [str(value) for value in frame[self.text_column]]
        rows: list[dict[str, Any]] = []
        for position, identifier in enumerate(identifiers):
            latency = (
                float(self.latencies_ms[position]) if position < len(self.latencies_ms) else 0.0
            )
            rows.append(
                {
                    "msg_id": identifier,
                    "n_tokens": len(texts[position].split()),
                    "latency_ms": latency,
                }
            )
        summary = pd.DataFrame(rows)
        if summary.empty:
            return summary
        summary.attrs["latency"] = latency_stats(self.latencies_ms)
        return summary

    def load_input(self, path: str | Path | None = None) -> pd.DataFrame:
        """Read the inference payload (a file, or a sample of the corpus).

        Args:
            path: Explicit file to read (``predict.input`` when ``None``).

        Returns:
            The frame to annotate.

        Raises:
            FileNotFoundError: When neither the file nor the corpus exists.
        """
        configured = path if path is not None else self._predict_node().get("input")
        if configured:
            source = Path(str(configured))
            if not source.is_absolute():
                source = self.paths.root / source
            return read_table(source)

        from src.data.loaders import EntityCorpusLoader  # noqa: PLC0415 - évite un cycle

        loader = EntityCorpusLoader(self.paths)
        for split in ("test", "val"):
            try:
                sample = loader.split(split)
            except (ValueError, FileNotFoundError):
                continue
            limit = int(self._predict_node().get("n_samples", 5))
            return sample.head(limit).reset_index(drop=True)
        msg = "No input available: generate the corpus first (mode=generate-data)"
        raise FileNotFoundError(msg)

    def _with_reference(self, mentions: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
        """Add ``expected_label`` and ``correct`` to a mention table.

        Args:
            mentions: Predicted mentions.
            spans: Reference annotations.

        Returns:
            The mention table with two extra columns.
        """
        truth = {
            (str(row.msg_id), int(row.start), int(row.end), str(row.label))
            for row in spans.itertuples(index=False)
        }
        keys = [
            (str(row.msg_id), int(row.start), int(row.end), str(row.label))
            for row in mentions.itertuples(index=False)
        ]
        expected = {
            (str(row.msg_id), int(row.start), int(row.end)): str(row.label)
            for row in spans.itertuples(index=False)
        }
        frame = mentions.copy()
        frame["expected_label"] = [expected.get((key[0], key[1], key[2])) for key in keys]
        frame["correct"] = [int(key in truth) for key in keys]
        return frame

    def _predict_node(self) -> dict[str, Any]:
        """Return the ``predict`` node of the configuration."""
        node = self.config.get("predict")
        return dict(node) if isinstance(node, Mapping) else {}


def _expand_latencies(
    mentions: pd.DataFrame, identifiers: Sequence[str], latencies: Sequence[float]
) -> list[float]:
    """Repeat the per-message latency on every mention of that message.

    La latence est mesurée **par message** ; la table publiée a une ligne par **mention**. La
    répétition est donc faite ici, sur l'identifiant de chaque mention : c'est ce qui permet de lire
    « cette entité a coûté 3,1 ms » sans confondre le coût du message et celui de la mention.

    Args:
        mentions: Mention table (``msg_id`` column).
        identifiers: Message identifiers, in the order the extraction loop used.
        latencies: Latency of every message, in milliseconds.

    Returns:
        One latency per mention, in publication order.

    Raises:
        KeyError: When a mention references a message outside the annotated frame.
    """
    position_of = {str(identifier): index for index, identifier in enumerate(identifiers)}
    return [float(latencies[position_of[str(value)]]) for value in mentions["msg_id"]]


__all__ = ["EntityPredictor"]
