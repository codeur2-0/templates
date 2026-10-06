"""Contrat d'extraction d'entités : une sortie n'est pas une étiquette mais une liste de spans.

Un projet de reconnaissance d'entités ne rend pas un libellé par document : il rend, pour chaque
texte, une liste de mentions avec leurs **décalages de caractères**. Ce contrat est énoncé ici,
une
fois, parce qu'il ne ressemble à aucun autre :

* :meth:`BaseEntityTagger.fit` lit **deux tables** — les messages et leurs annotations — et
apprend à
  reproduire les spans du split d'entraînement ;
* :meth:`BaseEntityTagger.predict` prend des textes bruts et rend une liste de
  :class:`EntityMention` par texte, triée par décalage de début, **sans chevauchement** : une
  mention
  ambiguë est une erreur de lecture pour le métier, pas une information ;
* chaque mention porte sa **provenance** (``regle`` ou ``modele``) et sa **confiance**. Un tagger
à
  transitions n'expose pas de probabilité par mention : la confiance publiée est le *taux de
  recouvrement* entre le span prédit et la couche de règles déclarées, et sa précision est mesurée
  par niveau dans le rapport d'évaluation. Publier une fausse probabilité serait pire que publier
  une mesure de corroboration documentée ;
* :meth:`save` / :meth:`load` persistent un artefact qui prédit exactement comme le modèle ajusté.
  L'artefact est un **répertoire** : un pipeline spaCy s'écrit avec ``nlp.to_disk`` (poids,
  vocabulaire, configuration), et le projet y ajoute l'index de règles appris sur le train ainsi
  qu'un manifeste — un fichier unique ne saurait pas porter ces trois choses sans les confondre.

Ce qui est réellement commun aux modalités est partagé :
:class:`~src.models.base.FitResult`, :class:`~src.models.base.ModelCard` et les versions de
bibliothèques viennent du socle.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from src.models.base import FitResult, ModelCard, library_versions
from src.utils.io import read_json, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Name of the artefact manifest written inside the artefact directory.
MANIFEST_FILE = "manifest.json"

#: Provenance of a predicted mention: the declared rule layer, or the learned tagger.
SOURCES: tuple[str, ...] = ("regle", "modele")


def _utc_now() -> str:
    """Return the current UTC timestamp, ISO formatted."""
    return datetime.now(timezone.utc).isoformat()


def tracked_versions() -> dict[str, str]:
    """Return the versions of the libraries loaded in this process, spaCy included.

    ``spacy`` et ``thinc`` ne font pas partie des bibliothèques suivies par le socle : ce projet
    les
    ajoute lui-même, parce qu'un artefact spaCy ne se relit pas avec la même version de tokenizer
    sans le dire.

    Returns:
        Mapping of library name to version.
    """
    versions = library_versions()
    for name in ("spacy", "thinc", "srsly"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - spaCy est une dépendance déclarée de la stack
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


@dataclass(frozen=True, slots=True)
class EntityMention:
    """One extracted mention: where it is, what type it is, where it comes from.

    Attributes:
        msg_id: Identifier of the document the mention belongs to.
        start: Index of the first character of the mention (inclusive).
        end: Index of the first character **after** the mention (exclusive, Python convention).
        label: Entity type of the mention.
        surface: Exact text of the mention (``text[start:end]``).
        source: Provenance — ``regle`` (declared rule layer) or ``modele`` (learned tagger).
        confidence: Corroboration level of the span (0.0 to 1.0), documented and measured.
        holdout: Whether the annotation used a surface reserved to the evaluation splits (``True``
            only when the reference annotations are known: it is a property of the corpus, not a
            prediction).
    """

    msg_id: str
    start: int
    end: int
    label: str
    surface: str
    source: str = "modele"
    confidence: float = 0.0
    holdout: bool = False

    def as_tuple(self) -> tuple[int, int, str]:
        """Return the ``(start, end, label)`` tuple used by the metrics.

        Returns:
            The span coordinates and its type — the exact-match key of an entity-level evaluation.
        """
        return (int(self.start), int(self.end), str(self.label))

    def to_row(self, *, expected: str | None = None) -> dict[str, Any]:
        """Return a flat, tabular representation of the mention.

        Args:
            expected: Reference label when the annotation is known (``None`` for a prediction).

        Returns:
            A mapping ready for :class:`pandas.DataFrame` construction or Parquet writing.
        """
        row: dict[str, Any] = {
            "msg_id": str(self.msg_id),
            "start": int(self.start),
            "end": int(self.end),
            "label": str(self.label),
            "surface": str(self.surface),
            "length": int(self.end - self.start),
            "source": str(self.source),
            "confidence": float(self.confidence),
        }
        if expected is not None:
            row["expected_label"] = str(expected)
            row["correct"] = int(str(expected) == str(self.label))
        return row


def sort_mentions(mentions: Sequence[EntityMention]) -> list[EntityMention]:
    """Sort mentions by start offset, then by end offset, then by type.

    Args:
        mentions: Mentions to sort.

    Returns:
        A new list, in reading order.
    """
    return sorted(mentions, key=lambda item: (item.start, item.end, item.label))


class BaseEntityTagger(ABC):
    """Common behaviour of every entity extractor of the project.

    Subclasses implement three things: the training (:meth:`_fit`), the extraction
    (:meth:`_predict`) and the artefact pair (:meth:`_export` / :meth:`_restore`). Everything else
    —
    contract checks, timing, model card, prediction frame, save/load plumbing — is implemented
    once,
    here.
    """

    #: Framework identifier, archived in the model card and displayed by the reports.
    framework: ClassVar[str] = "spacy"

    #: Default artefact directory name used when the path carries no suffix.
    default_model_file: ClassVar[str] = "tagger"

    def __init__(
        self,
        *,
        algorithm: str = "",
        params: Mapping[str, Any] | None = None,
        task: str = "multiclass",
        text_column: str = "text",
        target_name: str | None = "label",
        id_column: str = "msg_id",
        labels: Sequence[str] | None = None,
        random_state: int = 42,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the contract of the extractor.

        Args:
            algorithm: Algorithm identifier resolved by the factory (``gazetteer``, ``tagger``,
                ``hybrid``).
            params: Hyper-parameters of the architecture and of the training loop.
            task: Learning task as declared by the manifest (``multiclass``: entity typing is a
                per-token classification).
            text_column: Name of the column holding the text to annotate.
            target_name: Name of the entity-type column of the annotation table.
            id_column: Name of the join key between the two tables.
            labels: Known entity types, in display order (filled at fit time when unknown).
            random_state: Reproducibility seed.
            config: Full application configuration (train options, preprocessing, metrics).
            name: Human readable model name (defaults to the algorithm identifier).
        """
        self.algorithm = str(algorithm)
        self.params: dict[str, Any] = dict(params or {})
        self.task = str(task)
        self.text_column = str(text_column)
        self.target_name = str(target_name or "label")
        self.id_column = str(id_column)
        self.random_state = int(random_state)
        self.config: dict[str, Any] = dict(config or {})
        self.name = str(name or algorithm or type(self).__name__)
        self._labels: list[str] = [str(label) for label in (labels or [])]
        self._is_fitted = False
        self.fit_result_: FitResult | None = None

    # ------------------------------------------------------------------ identité ----------
    @property
    def is_fitted(self) -> bool:
        """Whether the extractor has been trained at least once."""
        return bool(self._is_fitted)

    @property
    def state(self) -> str:
        """Human readable state of the extractor (``fitted`` / ``untrained``)."""
        return "fitted" if self.is_fitted else "untrained"

    @property
    def labels(self) -> list[str]:
        """Entity types the extractor knows, in display order."""
        return list(self._labels)

    def summary(self) -> str:
        """Return a one-line description of the model (used in logs and reports)."""
        return (
            f"{type(self).__name__}(algorithm={self.algorithm or 'n/a'}, "
            f"labels={len(self._labels)}, state={self.state})"
        )

    def __repr__(self) -> str:
        """Return the unambiguous representation of the model."""
        return f"<{self.summary()}>"

    def check_is_fitted(self) -> None:
        """Raise when the extractor is used before being fitted.

        Raises:
            RuntimeError: When the model has not been trained (or reloaded) yet.
        """
        if not self.is_fitted:
            msg = (
                f"{type(self).__name__} is not fitted yet: call fit(documents, spans) or load an "
                "artefact before predicting."
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ entraînement ------
    def fit(
        self,
        documents: pd.DataFrame,
        spans: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> FitResult:
        """Fit the extractor on the annotated training messages.

        Args:
            documents: Message frame (at least the text column and the join key).
            spans: Annotation frame (join key, ``start``, ``end``, ``label``).
            callbacks: Project callbacks, fired once per epoch by models that have epochs.
            context: Mutable callback context (an early stop request is honoured).

        Returns:
            The :class:`~src.models.base.FitResult` of the fit.

        Raises:
            ValueError: When a required column is missing or the annotations are empty.
        """
        text_column = self.text_column
        if text_column not in documents.columns:
            msg = (
                f"Column '{text_column}' missing from the training corpus: "
                f"{sorted(documents.columns)}"
            )
            raise ValueError(msg)
        for column in (self.id_column, "start", "end", self.target_name):
            if column not in spans.columns:
                msg = (
                    f"Column '{column}' missing from the annotation table: {sorted(spans.columns)}"
                )
                raise ValueError(msg)
        if spans.empty:
            msg = "The annotation table is empty: a NER model cannot be fitted without entities"
            raise ValueError(msg)

        self._seed_everything()
        observed = sorted({str(label) for label in spans[self.target_name]})
        if not self._labels:
            self._labels = observed
        started_at = _utc_now()
        started = time.perf_counter()
        result = self._fit(documents, spans, callbacks=callbacks, context=context)
        result.duration_seconds = round(time.perf_counter() - started, 3)
        result.started_at = started_at
        result.finished_at = _utc_now()
        result.model_name = type(self).__name__
        result.algorithm = self.algorithm
        result.params = self._effective_params()
        result.n_samples = len(documents)
        result.n_features = len(self._labels)
        result.history = dict(getattr(result, "history", {}) or {})
        self.fit_result_ = result
        self._is_fitted = True
        logger.info(
            "Model fitted | {} | {} messages | {} mentions",
            self.summary(),
            len(documents),
            len(spans),
        )
        return result

    # ------------------------------------------------------------------ prédiction --------
    def predict(
        self, texts: Sequence[str], *, ids: Sequence[str] | None = None
    ) -> list[list[EntityMention]]:
        """Extract the entities of raw texts.

        Args:
            texts: Texts to annotate.
            ids: Optional identifiers, one per text (used in the returned mentions).

        Returns:
            One list of mentions per text, in reading order and without overlap.

        Raises:
            RuntimeError: When the model is not fitted.
            ValueError: When ``ids`` does not match ``texts``.
        """
        self.check_is_fitted()
        if ids is not None and len(ids) != len(texts):
            msg = f"ids has {len(ids)} entries for {len(texts)} texts"
            raise ValueError(msg)
        identifiers = [str(value) for value in (ids or [])] or [
            f"DOC-{index + 1:04d}" for index in range(len(texts))
        ]
        return self._predict(list(texts), identifiers)

    def predict_frame(
        self, documents: pd.DataFrame, *, spans: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Extract the entities of a message frame and return them as a table.

        Args:
            documents: Message frame (text column and join key).
            spans: Optional reference annotations, used to fill ``expected_label`` and
            ``correct``.

        Returns:
            A frame with one row per predicted mention, ready to be compared to the reference.

        Raises:
            ValueError: When the text column or the join key is missing.
        """
        if self.text_column not in documents.columns:
            msg = f"Column '{self.text_column}' missing from the frame: {sorted(documents.columns)}"
            raise ValueError(msg)
        if self.id_column not in documents.columns:
            msg = f"Column '{self.id_column}' missing from the frame: {sorted(documents.columns)}"
            raise ValueError(msg)
        identifiers = [str(value) for value in documents[self.id_column]]
        texts = [str(value) for value in documents[self.text_column]]
        expected = self._expected_labels(spans) if spans is not None else {}
        rows: list[dict[str, Any]] = []
        predictions = self.predict(texts, ids=identifiers)
        for identifier, mentions in zip(identifiers, predictions, strict=True):
            for mention in mentions:
                key = (str(identifier), int(mention.start), int(mention.end))
                rows.append(mention.to_row(expected=expected.get(key)))
        return pd.DataFrame(rows)

    @staticmethod
    def _expected_labels(spans: pd.DataFrame | None) -> dict[tuple[str, int, int], str]:
        """Index the reference labels by ``(msg_id, start, end)``.

        Args:
            spans: Reference annotation frame (or ``None``).

        Returns:
            Mapping of span key to reference entity type.
        """
        if spans is None or spans.empty:
            return {}
        return {
            (str(row.msg_id), int(row.start), int(row.end)): str(row.label)
            for row in spans.itertuples(index=False)
        }

    # ------------------------------------------------------------------ persistance -------
    def model_card(
        self,
        *,
        metrics: Mapping[str, float] | None = None,
        artifact: str | None = None,
        notes: Sequence[str] | None = None,
    ) -> ModelCard:
        """Build the traceability card archived next to the artefact.

        Args:
            metrics: Metrics of the run (training and validation).
            artifact: Name of the artefact written on disk.
            notes: Extra remarks (limitations, guards, measured references).

        Returns:
            The populated :class:`~src.models.base.ModelCard`.
        """
        return ModelCard(
            model_name=type(self).__name__,
            framework=self.framework,
            algorithm=self.algorithm,
            task=self.task,
            target_name=self.target_name,
            feature_names=[self.text_column],
            params=self._effective_params(),
            metrics={str(key): float(value) for key, value in (metrics or {}).items()},
            library_versions=tracked_versions(),
            created_at=_utc_now(),
            n_samples=int(self.fit_result_.n_samples) if self.fit_result_ else 0,
            n_features=len(self._labels),
            artifact=artifact,
            notes=list(notes or []),
        )

    def save(self, path: str | Path) -> Path:
        """Persist the extractor as a **directory** artefact.

        Args:
            path: Destination directory (a file path is turned into a sibling directory).

        Returns:
            The directory actually written.

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.check_is_fitted()
        destination = self._resolve_path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self._export(destination)
        write_json(
            destination / MANIFEST_FILE,
            {
                "model_name": type(self).__name__,
                "framework": self.framework,
                "algorithm": self.algorithm,
                "task": self.task,
                "text_column": self.text_column,
                "target_name": self.target_name,
                "id_column": self.id_column,
                "labels": self._labels,
                "params": self._effective_params(),
                "library_versions": tracked_versions(),
                "saved_at": _utc_now(),
            },
        )
        logger.info("Model saved | {} | {}", self.summary(), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseEntityTagger:
        """Reload an artefact written by :meth:`save`.

        Args:
            path: Artefact directory (a file path is turned into a sibling directory).
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded extractor, ready to predict.

        Raises:
            FileNotFoundError: When the manifest is missing from the artefact directory.
        """
        directory = cls._resolve_directory(path)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            msg = f"Not a model artefact directory (missing {MANIFEST_FILE}): {directory}"
            raise FileNotFoundError(msg)
        manifest = dict(read_json(manifest_path))
        model = cls(
            algorithm=str(manifest.get("algorithm", "")),
            params=dict(manifest.get("params") or {}),
            task=str(manifest.get("task", "multiclass")),
            text_column=str(manifest.get("text_column", "text")),
            target_name=str(manifest.get("target_name", "label")),
            id_column=str(manifest.get("id_column", "msg_id")),
            labels=[str(label) for label in manifest.get("labels", [])],
            random_state=int((config or {}).get("seed", 42)),
            config=dict(config or {}),
        )
        model._restore(directory)
        model._is_fitted = True
        return model

    @classmethod
    def _resolve_directory(cls, path: str | Path) -> Path:
        """Return the artefact directory of a path (creating the convention if needed).

        Args:
            path: Artefact path or directory.

        Returns:
            The directory that holds the artefact.
        """
        candidate = Path(path)
        if candidate.suffix:
            return candidate.with_suffix("")
        return candidate

    def _resolve_path(self, path: str | Path) -> Path:
        """Return the destination directory of :meth:`save`.

        Args:
            path: Destination path passed by the caller.

        Returns:
            The directory to write (``default_model_file`` is appended when the path is a bare
            directory or when it carries a file suffix that is not a directory name).
        """
        candidate = Path(path)
        if candidate.suffix:
            # `models/tagger.joblib` est un réflexe d'autres stacks : ici l'artefact est un
            # répertoire, donc le suffixe devient le nom du répertoire plutôt qu'un fichier.
            return candidate.with_suffix("")
        return candidate

    # ------------------------------------------------------------------ interne -----------
    @abstractmethod
    def _fit(
        self,
        documents: pd.DataFrame,
        spans: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> FitResult:
        """Train the extractor (implemented by the concrete stack)."""

    @abstractmethod
    def _predict(self, texts: list[str], ids: list[str]) -> list[list[EntityMention]]:
        """Extract the entities of raw texts (implemented by the concrete stack)."""

    @abstractmethod
    def _export(self, directory: Path) -> None:
        """Write the framework specific payload into the artefact directory."""

    @abstractmethod
    def _restore(self, directory: Path) -> None:
        """Reload the framework specific payload from the artefact directory."""

    def _effective_params(self) -> dict[str, Any]:
        """Return the effective hyper-parameters of the model."""
        return dict(self.params)

    def _seed_everything(self) -> None:
        """Seed the random generators used by the framework."""
        import random

        import numpy as np

        random.seed(self.random_state)
        np.random.seed(self.random_state)


__all__ = [
    "MANIFEST_FILE",
    "SOURCES",
    "BaseEntityTagger",
    "EntityMention",
    "sort_mentions",
    "tracked_versions",
]
