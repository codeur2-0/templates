"""Lecture et écriture du corpus de tickets étiquetés.

Le loader est le seul endroit qui sait **où** vit un jeu de données et **dans quel format** il est
stocké. Tout le reste (pipelines, évaluateur, notebooks) demande un ``DataFrame`` validé : un
corpus absent, un découpage vide ou un cadre qui viole son contrat Pandera échoue ici, avec un
message qui nomme le fichier et la colonne.

Le découpage est lu depuis la colonne ``split`` du corpus, jamais retiré au hasard : deux
exécutions à graine fixée voient donc exactement les mêmes lignes d'entraînement et de test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schemas import SPLITS, validate_predictions, validate_tickets
from src.utils.io import read_json, read_table, write_json, write_table, write_table_multiple
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Name of the metadata file written by the generator next to the corpus.
METADATA_FILE = "generation_metadata.json"


class TextLabelLoader:
    """Read and write the labelled corpus, its splits and the generation metadata.

    Attributes:
        paths: Project filesystem layout.
        formats: Formats written for the corpus (``parquet`` for machines, ``csv`` for humans).
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
            formats: Formats written by :meth:`save_documents`.
            validation_enabled: Enforce the Pandera contracts.
            lazy_validation: Collect every contract violation before raising.
        """
        self.paths = paths or ProjectPaths.from_root()
        self.formats = tuple(formats)
        self.validation_enabled = bool(validation_enabled)
        self.lazy_validation = bool(lazy_validation)

    # ------------------------------------------------------------------ chemins -----------
    @property
    def documents_path(self) -> Path:
        """Expected path of the corpus table (Parquet)."""
        return self.paths.raw_dir / "support_tickets.parquet"

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
        """Load the labelled corpus.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The corpus, one row per ticket.

        Raises:
            FileNotFoundError: When the corpus has not been generated yet.
        """
        frame = self._read(self.documents_path, "corpus")
        if self._should_validate(validate):
            frame = validate_tickets(frame, lazy=self.lazy_validation)
        logger.debug("Loaded {} tickets from {}", len(frame), self.documents_path)
        return frame

    def save_documents(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the corpus in every configured format.

        Args:
            frame: Corpus to persist.

        Returns:
            Mapping of format to written path.
        """
        payload = (
            validate_tickets(frame, lazy=self.lazy_validation)
            if self.validation_enabled
            else frame
        )
        written = write_table_multiple(payload, self.documents_path, self.formats)
        logger.info("Corpus written: {} tickets -> {}", len(payload), sorted(written))
        return written

    def split(self, name: str, *, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the rows of one split, validated.

        Args:
            name: Split name (:data:`src.data.schemas.SPLITS`).
            frame: Corpus to slice (defaults to the one on disk).

        Returns:
            The rows of the split.

        Raises:
            ValueError: When the split name is unknown or empty.
        """
        if name not in SPLITS:
            msg = f"Unknown split '{name}': expected one of {sorted(SPLITS)}"
            raise ValueError(msg)
        corpus = self.load_documents() if frame is None else frame
        if "split" not in corpus.columns:
            msg = "The corpus carries no 'split' column: it is not a labelled classification set"
            raise ValueError(msg)
        selected = corpus[corpus["split"].astype(str) == name].reset_index(drop=True)
        if selected.empty:
            msg = f"Split '{name}' is empty: check data.n_samples and the generator shares"
            raise ValueError(msg)
        return selected

    def label_distribution(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the class distribution of a corpus, split by split.

        Args:
            frame: Corpus to describe (defaults to the one on disk).

        Returns:
            One row per (split, label) with the row count and the share inside the split.
        """
        corpus = self.load_documents() if frame is None else frame
        counts = (
            corpus.groupby(["split", "label"], observed=True)
            .size()
            .rename("n_documents")
            .reset_index()
        )
        totals = counts.groupby("split", observed=True)["n_documents"].transform("sum")
        counts["share"] = counts["n_documents"] / totals
        return counts.sort_values(["split", "n_documents"], ascending=[True, False]).reset_index(
            drop=True
        )

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
            frame: Prediction rows.

        Returns:
            The written path.
        """
        payload = (
            validate_predictions(frame, lazy=self.lazy_validation)
            if self.validation_enabled
            else frame
        )
        written = write_table(payload, self.predictions_path)
        logger.info("Predictions written: {} rows -> {}", len(payload), written)
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


__all__ = ["METADATA_FILE", "TextLabelLoader"]
