"""Lecture et écriture du corpus à résumer : documents, références et faits.

Le loader est le seul endroit qui sait **où** vit un jeu de données et **dans quel format** il est
stocké. Tout le reste (pipelines, évaluateur, notebooks) demande des ``DataFrame`` validés : un
corpus absent, une table de faits manquante ou un cadre qui viole son contrat Pandera échoue ici,
avec un message qui nomme le fichier et la colonne.

Trois tables, trois chemins, et une règle : elles se lisent **ensemble**. Un résumé de référence
sans son document n'est pas évaluable, et un fait sans la table des documents ne peut pas être
confronté au texte produit. Le loader refuse donc de rendre les documents sans leurs références,
et la validation des liens (document inconnu, résumé manquant, fait hors des phrases) est faite
par :func:`~src.data.schemas.validate_corpus` **avant** tout calcul de ROUGE.

Le découpage est lu depuis la colonne ``split`` des documents, jamais retiré au hasard : deux
exécutions à graine fixée voient exactement les mêmes lignes d'entraînement et de test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schemas import (
    FACT_TYPES,
    SPLITS,
    salient_fact_counts,
    split_sizes,
    validate_corpus,
    validate_documents,
    validate_facts,
    validate_predictions,
    validate_references,
)
from src.utils.io import read_json, read_table, write_json, write_table, write_table_multiple
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Name of the metadata file written by the generator next to the corpus.
METADATA_FILE = "generation_metadata.json"

#: Stem of the reference summary table (without extension).
REFERENCES_STEM = "reference_summaries"

#: Stem of the fact table (without extension).
FACTS_STEM = "salient_facts"


class SummaryCorpusLoader:
    """Read and write the corpus, its splits, its metadata and its predictions.

    Attributes:
        paths: Project filesystem layout.
        dataset_name: Name of the document table on disk (``data.dataset_name``).
        formats: Formats written for the corpus (``parquet`` for machines, ``csv`` for humans).
        validation_enabled: Whether the Pandera contracts are enforced on load and save.
        lazy_validation: Collect every violation instead of stopping at the first one.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        *,
        dataset_name: str = "intervention_reports",
        formats: Sequence[str] = ("parquet", "csv"),
        validation_enabled: bool = True,
        lazy_validation: bool = False,
    ) -> None:
        """Configure the loader.

        Args:
            paths: Filesystem layout (defaults to the project layout detected from ``__file__``).
            dataset_name: Stem of the document table written by the generator.
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
        """Expected path of the document table (Parquet)."""
        return self.paths.raw_dir / f"{self.dataset_name}.parquet"

    @property
    def references_path(self) -> Path:
        """Expected path of the reference summary table (Parquet)."""
        return self.paths.raw_dir / f"{REFERENCES_STEM}.parquet"

    @property
    def facts_path(self) -> Path:
        """Expected path of the fact table (Parquet)."""
        return self.paths.raw_dir / f"{FACTS_STEM}.parquet"

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
        """Load the document table alone (metadata inspection, notebooks).

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The documents, one row per compte-rendu.

        Raises:
            FileNotFoundError: When the corpus has not been generated yet.
                pandera.errors.SchemaError: When the table breaks its contract.
        """
        frame = self._read(self.documents_path, "corpus")
        if self._should_validate(validate):
            frame = validate_documents(frame, lazy=self.lazy_validation)
        return frame

    def load_references(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the reference summary table alone.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The reference summaries, one row per document.
        """
        frame = self._read(self.references_path, "références")
        if self._should_validate(validate):
            frame = validate_references(frame, lazy=self.lazy_validation)
        return frame

    def load_facts(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the fact table alone.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The facts, one row per reportable fact.
        """
        frame = self._read(self.facts_path, "faits")
        if self._should_validate(validate):
            frame = validate_facts(frame, lazy=self.lazy_validation)
        return frame

    def load_corpus(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate the three tables **together**.

        Returns:
            ``(documents, references, facts)``, validated and linked.

        Raises:
            FileNotFoundError: When one of the three tables is missing.
            ValueError: When the tables do not agree on the same documents.
        """
        documents = self.load_documents()
        references = self.load_references()
        facts = self.load_facts()
        documents, references, facts = validate_corpus(
            documents,
            references,
            facts,
            lazy=self.lazy_validation,
        )
        logger.info(
            "Corpus loaded: {} documents ({}), {} références, {} faits ({} saillants)",
            len(documents),
            ", ".join(f"{name}={count}" for name, count in split_sizes(documents).items()),
            len(references),
            len(facts),
            int(facts["salient"].sum()),
        )
        return documents, references, facts

    def load_split(self, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load one split of the corpus, references and facts included.

        Args:
            split: Split name (``train``, ``val``, ``calibration`` or ``test``).

        Returns:
            The three tables restricted to the documents of that split.

        Raises:
            ValueError: When the split name is unknown.
        """
        if split not in SPLITS:
            msg = f"Unknown split '{split}': expected one of {list(SPLITS)}"
            raise ValueError(msg)
        documents, references, facts = self.load_corpus()
        selected = documents.loc[documents["split"] == split]
        keys = set(selected["doc_id"])
        return (
            selected.reset_index(drop=True),
            references.loc[references["doc_id"].isin(keys)].reset_index(drop=True),
            facts.loc[facts["doc_id"].isin(keys)].reset_index(drop=True),
        )

    def load_metadata(self) -> dict[str, Any]:
        """Load the generation metadata (recipe of the corpus).

        Returns:
            The metadata mapping, empty when the file has not been written.
        """
        if not self.metadata_path.is_file():
            return {}
        payload = read_json(self.metadata_path)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def load_references_frame(self, split: str | None = None) -> pd.DataFrame:
        """Load the reference summaries, optionally restricted to one split.

        Args:
            split: Optional split filter.

        Returns:
            The reference summary table, indexed by ``doc_id``.
        """
        _, references, _ = self.load_corpus()
        if split is not None:
            documents, _, _ = self.load_corpus()
            keys = set(documents.loc[documents["split"] == split, "doc_id"])
            references = references.loc[references["doc_id"].isin(keys)]
        return references.reset_index(drop=True)

    def load_predictions(self, *, validate: bool | None = None) -> pd.DataFrame:
        """Load the prediction table.

        Args:
            validate: Override :attr:`validation_enabled` for this call.

        Returns:
            The predictions, one row per (document, strategy).
        """
        frame = self._read(self.predictions_path, "prédictions", formats=("csv", "parquet"))
        if self._should_validate(validate):
            frame = validate_predictions(frame, lazy=self.lazy_validation)
        return frame

    # ------------------------------------------------------------------ écriture ----------
    def save_documents(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the document table in every configured format.

        Args:
            frame: Generated document table.

        Returns:
            Mapping ``format -> path``.
        """
        if self.validation_enabled:
            frame = validate_documents(frame, lazy=self.lazy_validation)
        return write_table_multiple(frame, self.paths.raw_dir / self.dataset_name, self.formats)

    def save_references(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the reference summary table.

        Args:
            frame: Generated reference summaries.

        Returns:
            Mapping ``format -> path``.
        """
        if self.validation_enabled:
            frame = validate_references(frame, lazy=self.lazy_validation)
        return write_table_multiple(frame, self.paths.raw_dir / REFERENCES_STEM, self.formats)

    def save_facts(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Validate then persist the fact table.

        Args:
            frame: Generated facts.

        Returns:
            Mapping ``format -> path``.
        """
        if self.validation_enabled:
            frame = validate_facts(frame, lazy=self.lazy_validation)
        return write_table_multiple(frame, self.paths.raw_dir / FACTS_STEM, self.formats)

    def save_metadata(self, payload: Mapping[str, Any]) -> Path:
        """Persist the generation metadata (the recipe of the corpus).

        Args:
            payload: Metadata written by the generator.

        Returns:
            The written path.
        """
        return write_json(self.metadata_path, dict(payload), indent=2, sort_keys=True)

    def save_predictions(self, frame: pd.DataFrame) -> Path:
        """Validate then persist the prediction table.

        Args:
            frame: Prediction rows.

        Returns:
            The written path.
        """
        if self.validation_enabled:
            frame = validate_predictions(frame, lazy=self.lazy_validation)
        return write_table(frame, self.predictions_path)

    # ------------------------------------------------------------------ helpers -----------
    def split_sizes(self) -> dict[str, int]:
        """Count the documents of each split without loading the annotated tables.

        Returns:
            Mapping ``split -> number of documents``.
        """
        return split_sizes(self.load_documents())

    def salient_facts(self) -> dict[str, int]:
        """Count the salient facts of each type.

        Returns:
            Mapping ``fact_type -> number of salient facts``, :data:`FACT_TYPES` order.
        """
        return salient_fact_counts(self.load_facts())

    def _should_validate(self, override: bool | None) -> bool:
        """Resolve the validation flag of a call.

        Args:
            override: Per-call override (``None`` keeps the loader default).

        Returns:
            Whether the Pandera contracts must be enforced.
        """
        return self.validation_enabled if override is None else bool(override)

    @staticmethod
    def _read(path: Path, label: str, *, formats: Sequence[str] = ("parquet", "csv")) -> pd.DataFrame:
        """Read a table, raising a message that names the missing artefact.

        Args:
            path: Expected path (its extension is used as the primary format).
            label: Human label used in the error message.
            formats: Fallback formats, tried in order.

        Returns:
            The table.

        Raises:
            FileNotFoundError: When no candidate file exists.
        """
        candidates = [path, *(path.with_suffix(f".{name}") for name in formats)]
        for candidate in candidates:
            if candidate.is_file():
                return read_table(candidate)
        expected = path.parent / f"{path.stem}.{formats[0]}"
        msg = (
            f"{label.capitalize()} introuvable : {expected} n'existe pas. "
            "Exécuter `make data` (ou `python -m src.main mode=generate-data`) pour le générer."
        )
        raise FileNotFoundError(msg)


__all__ = [
    "FACTS_STEM",
    "METADATA_FILE",
    "REFERENCES_STEM",
    "SummaryCorpusLoader",
]
