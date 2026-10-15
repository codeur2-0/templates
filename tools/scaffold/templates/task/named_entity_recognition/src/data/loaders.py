"""Lecture et écriture du corpus annoté : les messages **et** leurs annotations.

Le loader est le seul endroit qui sait **où** vit un jeu de données et **dans quel format** il est
stocké. Tout le reste (pipelines, évaluateur, notebooks) demande des ``DataFrame`` validés : un
corpus absent, une table d'annotations manquante ou un cadre qui viole son contrat Pandera échoue
ici, avec un message qui nomme le fichier et la colonne.

Deux tables, deux chemins, et une règle : elles se lisent **ensemble**. Un message sans annotation
n'est pas « vide », c'est un message dont le corpus dit qu'il ne contient aucune entité — et le
loader refuse de rendre l'un sans l'autre, parce que toutes les métriques du projet sont calculées au
niveau entité, donc sur les deux tables jointes (:func:`~src.data.schemas.validate_corpus` vérifie la
jointure, l'exactitude des surfaces et l'absence de chevauchement).

Le découpage est lu depuis la colonne ``split`` des messages, jamais retiré au hasard : deux
exécutions à graine fixée voient exactement les mêmes lignes d'entraînement et de test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schemas import (
    ENTITY_LABELS,
    SPLITS,
    holdout_share,
    label_counts,
    validate_corpus,
    validate_messages,
    validate_predictions,
    validate_spans,
)
from src.utils.io import read_json, read_table, write_json, write_table, write_table_multiple
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Name of the metadata file written by the generator next to the corpus.
METADATA_FILE = "generation_metadata.json"

#: Name of the annotation table (without extension).
ANNOTATIONS_STEM = "spans"


class EntityCorpusLoader:
    """Read and write the annotated corpus, its splits, its metadata and its predictions.

    Attributes:
        paths: Project filesystem layout.
        dataset_name: Name of the message table on disk (``data.dataset_name``).
        formats: Formats written for the corpus (``parquet`` for machines, ``csv`` for humans).
        validation_enabled: Whether the Pandera contracts are enforced on load and save.
        lazy_validation: Collect every violation instead of stopping at the first one.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        *,
        dataset_name: str = "sav_messages",
        formats: Sequence[str] = ("parquet", "csv"),
        validation_enabled: bool = True,
        lazy_validation: bool = False,
    ) -> None:
        """Configure the loader.

        Args:
            paths: Filesystem layout (defaults to the project layout detected from ``__file__``).
            dataset_name: Stem of the message table written by the generator.
            formats: Formats written by :meth:`save_documents`.
            validation_enabled: Enforce the Pandera contracts.
            lazy_validation: Collect every contract violation before raising.
        """
        self.paths = paths or ProjectPaths.from_root()
        self.dataset_name = str(dataset_name)
        self.formats = tuple(formats)
        self.validation_enabled = bool(validation_enabled)
        self.lazy_validation = bool(lazy_validation)

    # ------------------------------------------------------------------ chemins -----------
    @property
    def documents_path(self) -> Path:
        """Expected path of the message table (Parquet)."""
        return self.paths.raw_dir / f"{self.dataset_name}.parquet"

    @property
    def annotations_path(self) -> Path:
        """Expected path of the annotation table (Parquet)."""
        return self.paths.raw_dir / f"{ANNOTATIONS_STEM}.parquet"

    @property
    def metadata_path(self) -> Path:
        """Expected path of the generation metadata (JSON)."""
        return self.paths.raw_dir / METADATA_FILE

    @property
    def predictions_path(self) -> Path:
        """Expected path of the prediction table (CSV)."""
        return self.paths.reports_dir / "predictions.csv"

    # ------------------------------------------------------------------ corpus ------------
    def load_documents(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the message table alone (metadata inspection, notebooks).

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The messages, one row per message.

        Raises:
            FileNotFoundError: When the corpus has not been generated yet.
            pandera.errors.SchemaError: When the table breaks its contract.
        """
        frame = self._read(self.documents_path, "corpus")
        if self._should_validate(validate):
            frame = validate_messages(frame, lazy=self.lazy_validation)
        logger.debug("Loaded {} messages from {}", len(frame), self.documents_path)
        return frame

    def load_annotations(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the annotation table alone.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The annotations, one row per entity.

        Raises:
            FileNotFoundError: When the annotations have not been generated yet.
            pandera.errors.SchemaError: When the table breaks its contract.
        """
        frame = self._read(self.annotations_path, "annotations")
        if self._should_validate(validate):
            frame = validate_spans(frame, lazy=self.lazy_validation)
        return frame

    def load_corpus(self, *, validate: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both tables and validate the links between them.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The messages and their annotations.

        Raises:
            FileNotFoundError: When one of the two tables is missing.
            ValueError: When a cross-table constraint is violated (join key, surface, overlap).
        """
        documents = self._read(self.documents_path, "corpus")
        spans = self._read(self.annotations_path, "annotations")
        if self._should_validate(validate):
            documents, spans = validate_corpus(documents, spans, lazy=self.lazy_validation)
        logger.debug(
            "Loaded {} messages and {} annotations from {}",
            len(documents),
            len(spans),
            self.paths.raw_dir,
        )
        return documents, spans

    def save_documents(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the message table in every configured format.

        Args:
            frame: Messages to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = validate_messages(frame, lazy=self.lazy_validation) if self.validation_enabled else frame
        written = write_table_multiple(payload, self.documents_path, self.formats)
        logger.info("Messages written: {} rows -> {}", len(payload), sorted(written))
        return written

    def save_annotations(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the annotation table in every configured format.

        Args:
            frame: Annotations to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = validate_spans(frame, lazy=self.lazy_validation) if self.validation_enabled else frame
        written = write_table_multiple(payload, self.annotations_path, self.formats)
        logger.info("Annotations written: {} rows -> {}", len(payload), sorted(written))
        return written

    def split(self, name: str, *, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the messages of one split, validated.

        Args:
            name: Split name (:data:`src.data.schemas.SPLITS`).
            frame: Message table to slice (defaults to the one on disk).

        Returns:
            The messages of the split.

        Raises:
            ValueError: When the split name is unknown or empty.
        """
        if name not in SPLITS:
            msg = f"Unknown split '{name}': expected one of {sorted(SPLITS)}"
            raise ValueError(msg)
        documents = self.load_documents() if frame is None else frame
        if "split" not in documents.columns:
            msg = "The corpus carries no 'split' column: the generator did not write the splits"
            raise ValueError(msg)
        selected = documents[documents["split"].astype(str) == name].reset_index(drop=True)
        if selected.empty:
            msg = f"Split '{name}' is empty: check data.n_samples and the generator shares"
            raise ValueError(msg)
        return selected

    def split_annotations(
        self,
        name: str,
        *,
        documents: pd.DataFrame | None = None,
        spans: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the annotations belonging to one split.

        Args:
            name: Split name (``train``, ``val``, ``calibration``, ``test``).
            documents: Message table (defaults to the one on disk).
            spans: Annotation table (defaults to the one on disk).

        Returns:
            The annotations of the split, sorted by message and offset.
        """
        if documents is None or spans is None:
            loaded_documents, loaded_spans = self.load_corpus()
            documents = loaded_documents if documents is None else documents
            spans = loaded_spans if spans is None else spans
        selected = self.split(name, frame=documents)
        keep = set(selected["msg_id"].astype(str))
        subset = spans[spans["msg_id"].astype(str).isin(keep)]
        return subset.sort_values(["msg_id", "start", "end"]).reset_index(drop=True)

    def entity_distribution(
        self,
        *,
        documents: pd.DataFrame | None = None,
        spans: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Describe the annotation table, split by split and by entity type.

        Args:
            documents: Message table (defaults to the one on disk).
            spans: Annotation table (defaults to the one on disk).

        Returns:
            One row per (split, label) with the mention count, the share inside the split and the
            share of reserved surfaces — trois colonnes qui disent quel type est difficile et
            pourquoi.
        """
        if documents is None or spans is None:
            loaded_documents, loaded_spans = self.load_corpus()
            documents = loaded_documents if documents is None else documents
            spans = loaded_spans if spans is None else spans
        split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
        annotated = spans.assign(split=spans["msg_id"].map(split_of))
        rows: list[dict[str, Any]] = []
        for split in SPLITS:
            subset = annotated[annotated["split"] == split]
            counts = label_counts(subset)
            shares = holdout_share(subset)
            total = int(sum(counts.values()))
            for label in ENTITY_LABELS:
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "n_mentions": counts[label],
                        "share": round(counts[label] / total, 4) if total else 0.0,
                        "holdout_share": shares[label],
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ métadonnées --------
    def save_metadata(self, payload: Mapping[str, Any]) -> Path:
        """Persist the generation metadata next to the corpus.

        Args:
            payload: Metadata mapping produced by the family generator.

        Returns:
            The written path.
        """
        return write_json(self.metadata_path, dict(payload))

    def load_metadata(self) -> dict[str, Any]:
        """Read the generation metadata.

        Returns:
            The metadata mapping (empty when the file does not exist yet).

        Raises:
            FileNotFoundError: When the metadata is missing while the corpus exists: an artefact
                without its recipe cannot be explained.
        """
        if not self.metadata_path.exists():
            if self.documents_path.exists():
                raise FileNotFoundError(
                    f"Corpus present but metadata missing: {self.metadata_path} "
                    "(regenerate the data with mode=generate-data)"
                )
            return {}
        return dict(read_json(self.metadata_path))

    # ------------------------------------------------------------------ prédictions -------
    def save_predictions(self, frame: pd.DataFrame) -> Path:
        """Validate then persist the prediction table.

        Args:
            frame: Predicted mentions.

        Returns:
            The written path.
        """
        payload = (
            validate_predictions(frame, lazy=self.lazy_validation)
            if self.validation_enabled
            else frame
        )
        written = write_table(payload, self.predictions_path)
        logger.info("Predictions written: {} mentions -> {}", len(payload), written)
        return written

    # ------------------------------------------------------------------ interne -----------
    def _read(self, path: Path, label: str) -> pd.DataFrame:
        """Read a table, with a message naming the missing file."""
        if not path.exists():
            msg = (
                f"{label.capitalize()} not found at {path}: generate the dataset first "
                "(python -m src.main mode=generate-data, or make data)"
            )
            raise FileNotFoundError(msg)
        return read_table(path)

    def _should_validate(self, override: bool | None) -> bool:
        """Resolve the validation flag for one call."""
        return self.validation_enabled if override is None else bool(override)


__all__ = ["ANNOTATIONS_STEM", "METADATA_FILE", "EntityCorpusLoader"]
