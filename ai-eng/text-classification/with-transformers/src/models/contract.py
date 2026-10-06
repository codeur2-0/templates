"""Contrat texte du classifieur : une ligne porte son texte, son libellé et son découpage.

Le contrat tabulaire du socle (:mod:`src.models.base`) est bâti autour d'une **matrice de
features** : sélection de colonnes, imputation, alignement des features à l'inférence. Un
classifieur de texte n'a rien de tout cela : il reçoit un texte brut et fabrique lui-même sa
représentation, exactement comme un ``ColumnTransformer`` la fabrique pour un tableau. Réutiliser le
contrat tabulaire traînerait donc des concepts qui ne veulent rien dire ici, et c'est pourquoi le
contrat de classification de texte est énoncé une fois, dans ce module :

* :meth:`BaseTextClassifier.fit` lit un *DataFrame* (une ligne = un document), apprend sur les
  lignes de son découpage d'entraînement et rend les métriques qu'il sait calculer ;
* :meth:`BaseTextClassifier.predict` / :meth:`predict_proba` prennent des textes bruts et
  renvoient les libellés / les probabilités, dans l'ordre de :attr:`classes` ;
* :meth:`BaseTextClassifier.explain` montre les termes qui ont décidé une prédiction : un
  classifieur linéaire *peut* dire pourquoi, et un projet qui ne l'affiche pas s'interdit la
  seule lecture d'erreur qui distingue « mauvaise représentation » de « mauvaise décision » ;
* :meth:`save` / :meth:`load` persistent un artefact qui classe exactement comme le modèle ajusté —
  vocabulaire, poids, classes et configuration de vectorisation compris.

Ce qui est réellement commun aux modalités est partagé : :class:`~src.models.base.FitResult`,
:class:`~src.models.base.ModelCard` et les versions de bibliothèques viennent du socle.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.models.base import FitResult, ModelCard, library_versions
from src.utils.io import save_pickle
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    """Return the current UTC timestamp, ISO formatted."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TextPrediction:
    """One classification decision, with what decided it.

    Attributes:
        text: The classified text.
        label: Predicted label.
        confidence: Probability of the predicted label.
        probabilities: Probability of every known label (may be empty for a margin-based model).
        explanation: Most influential terms, most influential first (may be empty).
    """

    text: str
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    explanation: list[tuple[str, float]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Return a flat, tabular representation of the decision.

        Returns:
            A mapping ready for :class:`pandas.DataFrame` construction or CSV writing.
        """
        row: dict[str, Any] = {
            "text": self.text,
            "prediction": self.label,
            "confidence": self.confidence,
        }
        for label, probability in self.probabilities.items():
            row[f"p_{label}"] = probability
        row["explanation"] = ", ".join(
            f"{term} ({weight:+.2f})" for term, weight in self.explanation
        )
        return row


class BaseTextClassifier(ABC):
    """Common behaviour of every text classifier of the project.

    Subclasses implement four things: the training (:meth:`_fit`), the scoring
    (:meth:`_predict_proba`), the local explanation (:meth:`_explain`) and the persistence pair
    (:meth:`_payload` / :meth:`_restore`). Everything else — contract checks, timing, model card,
    prediction frame — is implemented once, here.
    """

    #: Framework identifier, archived in the model card and displayed by the reports.
    framework: ClassVar[str] = "text-classifier"

    #: Default file name used when the artefact path is a directory.
    default_model_file: ClassVar[str] = "classifier.joblib"

    def __init__(
        self,
        *,
        algorithm: str = "",
        params: Mapping[str, Any] | None = None,
        task: str = "multiclass",
        text_column: str = "text",
        target_name: str | None = None,
        labels: Sequence[str] | None = None,
        random_state: int = 42,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the contract of the classifier.

        Args:
            algorithm: Algorithm identifier resolved by the factory.
            params: Hyper-parameters of the vectorisation and of the estimator.
            task: Learning task (``multiclass``, ``binary``).
            text_column: Name of the column holding the text to classify.
            target_name: Name of the label column.
            labels: Known labels, in display order (filled at fit time when unknown).
            random_state: Reproducibility seed.
            config: Full application configuration (preprocessing, metrics, train options).
            name: Human readable model name (defaults to the algorithm identifier).
        """
        self.algorithm = str(algorithm)
        self.params: dict[str, Any] = dict(params or {})
        self.task = str(task)
        self.text_column = str(text_column)
        self.target_name = target_name
        self.random_state = int(random_state)
        self.config: dict[str, Any] = dict(config or {})
        self.name = str(name or algorithm or type(self).__name__)
        self._labels: list[str] = [str(label) for label in (labels or [])]
        self._is_fitted = False
        self.fit_result_: FitResult | None = None

    # ------------------------------------------------------------------ identité ----------
    @property
    def is_fitted(self) -> bool:
        """Whether the classifier has been trained at least once."""
        return self._is_fitted

    @property
    def state(self) -> str:
        """Training state, used in logs and reports (``fitted`` / ``unfitted``)."""
        return "fitted" if self._is_fitted else "unfitted"

    @property
    def labels(self) -> list[str]:
        """Known labels, in the order the probability columns follow."""
        return list(self._labels)

    @property
    def n_features(self) -> int:
        """Size of the representation actually learned (vocabulary size, hidden size, ...)."""
        return 0

    def summary(self) -> str:
        """Return a one-line, human readable description of the classifier."""
        return (
            f"{type(self).__name__}(algorithm={self.algorithm or '-'}, "
            f"classes={len(self._labels)}, representation={self.n_features}, state={self.state})"
        )

    def __repr__(self) -> str:
        """Return the log-friendly representation."""
        return self.summary()

    # ------------------------------------------------------------------ contrat -----------
    def check_is_fitted(self) -> None:
        """Assert that the classifier is trained.

        Raises:
            RuntimeError: When the classifier has never been fitted (or was replaced by ``load``
                on a broken artefact).
        """
        if not self._is_fitted:
            msg = (
                f"{type(self).__name__} is not fitted: call fit(documents) before predicting "
                "(or reload an artefact with load_model)."
            )
            raise RuntimeError(msg)

    def fit(
        self,
        documents: pd.DataFrame,
        *,
        target: str | None = None,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> FitResult:
        """Train the classifier on a frame of labelled documents.

        Args:
            documents: Training documents (one row per document).
            target: Label column (defaults to ``target_name`` then to ``label``).
            callbacks: Training callbacks, fired by the trainer (the model only forwards them).
            context: Mutable callback context handed to the trainer.

        Returns:
            The :class:`~src.models.base.FitResult` of the run.

        Raises:
            ValueError: When the frame is empty, the label column is missing, or a single class
                is present (a classifier trained on one class cannot be evaluated).
        """
        column = self._target_column(documents, target)
        if documents.empty:
            msg = "Cannot fit on an empty frame: generate the corpus first (mode=generate-data)"
            raise ValueError(msg)
        labels = sorted(str(value) for value in documents[column].astype(str).unique())
        if len(labels) < 2:
            msg = f"Column '{column}' holds a single class ({labels}): nothing to separate"
            raise ValueError(msg)
        self.target_name = column
        self._labels = self._labels or labels
        started = _utc_now()
        clock = time.perf_counter()
        metrics = dict(self._fit(documents, callbacks=callbacks, context=context) or {})
        duration = time.perf_counter() - clock
        self._is_fitted = True
        self.fit_result_ = FitResult(
            model_name=type(self).__name__,
            algorithm=self.algorithm,
            metrics=metrics,
            n_samples=len(documents),
            n_features=int(self.n_features),
            duration_seconds=duration,
            started_at=started,
            finished_at=_utc_now(),
            params=self._effective_params(),
            extra=self._extra_metadata(),
        )
        logger.info(
            "{} fitted | {} documents | {} classes | {} | {:.3f}s",
            type(self).__name__,
            len(documents),
            len(self._labels),
            self.algorithm,
            duration,
        )
        return self.fit_result_

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Classify raw texts.

        Args:
            texts: Texts to classify.

        Returns:
            One label per text, taken from :attr:`labels`.
        """
        probabilities = self.predict_proba(texts)
        indices = np.argmax(probabilities, axis=1)
        return [self._labels[int(index)] for index in indices]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score raw texts.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(n_texts, n_labels)`` matrix whose rows follow :attr:`labels`.

        Raises:
            RuntimeError: When the classifier is not fitted.
        """
        self.check_is_fitted()
        items = [str(text) for text in texts]
        if not items:
            return np.zeros((0, len(self._labels)), dtype="float64")
        probabilities = np.asarray(self._predict_proba(items), dtype="float64")
        if probabilities.shape != (len(items), len(self._labels)):
            msg = (
                f"{type(self).__name__} produced a {probabilities.shape} score matrix for "
                f"{len(items)} texts and {len(self._labels)} labels"
            )
            raise RuntimeError(msg)
        return probabilities

    def predict_frame(
        self, documents: pd.DataFrame, *, text_column: str | None = None
    ) -> pd.DataFrame:
        """Classify a frame and return the decisions as a table.

        Args:
            documents: Frame holding the text column.
            text_column: Column override (defaults to the configured one).

        Returns:
            A frame with ``prediction``, ``confidence`` and one probability column per label.

        Raises:
            ValueError: When the text column is missing.
        """
        column = str(text_column or self.text_column)
        if column not in documents.columns:
            msg = f"Column '{column}' not found in the frame: {sorted(documents.columns)}"
            raise ValueError(msg)
        texts = documents[column].astype(str).tolist()
        probabilities = self.predict_proba(texts)
        frame = pd.DataFrame({"prediction": self.predict(texts)})
        frame["confidence"] = probabilities.max(axis=1)
        for position, label in enumerate(self._labels):
            frame[f"p_{label}"] = probabilities[:, position]
        return frame

    def explain(self, texts: Sequence[str], k: int = 5) -> list[list[tuple[str, float]]]:
        """Show the terms that pushed each prediction.

        Args:
            texts: Texts to explain.
            k: Number of terms to return per text.

        Returns:
            One list of ``(term, weight)`` pairs per text, most influential first.
        """
        self.check_is_fitted()
        items = [str(text) for text in texts]
        if not items:
            return []
        return self._explain(items, max(int(k), 1))

    def model_card(
        self,
        *,
        metrics: Mapping[str, float] | None = None,
        artifact: str | None = None,
        notes: Sequence[str] | None = None,
    ) -> ModelCard:
        """Assemble the traceability card of the fit.

        Args:
            metrics: Metrics to archive (defaults to the fit result's, then to the run's).
            artifact: File name of the persisted artefact.
            notes: Human readable remarks (leakage guards, known limitations).

        Returns:
            The :class:`~src.models.base.ModelCard` of the run.
        """
        fit = self.fit_result_
        return ModelCard(
            model_name=type(self).__name__,
            framework=self.framework,
            algorithm=self.algorithm,
            task=self.task,
            target_name=self.target_name,
            feature_names=[self.text_column],
            params=self._effective_params(),
            metrics=dict(metrics if metrics is not None else (fit.metrics if fit else {})),
            library_versions=library_versions(),
            created_at=_utc_now(),
            n_samples=0 if fit is None else fit.n_samples,
            n_features=int(self.n_features),
            artifact=artifact,
            notes=list(notes or []),
        )

    # ------------------------------------------------------------------ persistance -------
    def save(self, path: str | Path) -> Path:
        """Persist the classifier (representation, weights, labels, configuration).

        Args:
            path: Destination file (a directory receives the default file name).

        Returns:
            The written path.
        """
        destination = self._resolve_path(path)
        payload: dict[str, Any] = {
            "algorithm": self.algorithm,
            "params": self.params,
            "task": self.task,
            "text_column": self.text_column,
            "target_name": self.target_name,
            "labels": self.labels,
            "random_state": self.random_state,
            "name": self.name,
            "fit_result": None if self.fit_result_ is None else self.fit_result_.to_dict(),
            **self._payload(),
        }
        return save_pickle(payload, destination)

    @classmethod
    def load(
        cls, path: str | Path, *, config: Mapping[str, Any] | None = None
    ) -> BaseTextClassifier:
        """Reload a persisted classifier.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded classifier, ready to classify.
        """
        from src.utils.io import load_pickle

        payload = load_pickle(path)
        model = cls(
            algorithm=str(payload.get("algorithm", "")),
            params=dict(payload.get("params", {})),
            task=str(payload.get("task", "multiclass")),
            text_column=str(payload.get("text_column", "text")),
            target_name=payload.get("target_name"),
            labels=[str(label) for label in payload.get("labels", [])],
            random_state=int(payload.get("random_state", 42)),
            config=dict(config or {}),
            name=str(payload.get("name", payload.get("algorithm", ""))),
        )
        model._restore(payload)
        if payload.get("fit_result"):
            model.fit_result_ = FitResult.from_dict(payload["fit_result"])
        model._is_fitted = True
        return model

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve the artefact destination (directory or file)."""
        destination = Path(path)
        if destination.is_dir() or not destination.suffix:
            destination = destination / self.default_model_file
        return destination

    # ------------------------------------------------------------------ à implémenter -----
    @abstractmethod
    def _fit(
        self,
        documents: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Train on labelled documents and return the training metrics.

        Args:
            documents: Training documents.
            callbacks: Training callbacks (only the framework that runs epochs must fire them:
                the project trainer does it for the single-shot estimators).
            context: Mutable callback context.

        Returns:
            Training metrics, keyed with their plain names (the trainer prefixes the validation
            ones with ``val_``).
        """

    @abstractmethod
    def _predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score raw texts, in the order of :attr:`labels`.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(n_texts, n_labels)`` matrix of probabilities.
        """

    @abstractmethod
    def _explain(self, texts: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Return the most influential terms of each prediction.

        Args:
            texts: Texts to explain.
            k: Number of terms per text.

        Returns:
            One list of ``(term, weight)`` pairs per text.
        """

    @abstractmethod
    def _payload(self) -> dict[str, Any]:
        """Return the framework-specific state to persist.

        Returns:
            A picklable mapping restored by :meth:`_restore`.
        """

    @abstractmethod
    def _restore(self, payload: Mapping[str, Any]) -> None:
        """Restore the framework-specific state of an artefact.

        Args:
            payload: Mapping produced by :meth:`_payload`.
        """

    def _effective_params(self) -> dict[str, Any]:
        """Return the hyper-parameters actually applied (published in the card)."""
        return dict(self.params)

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the extra facts archived next to the fit result."""
        return {
            "n_classes": float(len(self._labels)),
            "representation_size": float(self.n_features),
        }

    def _target_column(self, documents: pd.DataFrame, target: str | None) -> str:
        """Resolve the label column, with a readable error when it is missing.

        Args:
            documents: Frame to read.
            target: Explicit column override.

        Returns:
            The name of the label column.

        Raises:
            ValueError: When no candidate column exists.
        """
        candidates = [target, self.target_name, "label"]
        for candidate in candidates:
            if candidate and str(candidate) in documents.columns:
                return str(candidate)
        msg = (
            f"No label column found in {sorted(documents.columns)}: expected "
            f"'{target or self.target_name or 'label'}'"
        )
        raise ValueError(msg)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn a score matrix into probabilities (stable, per row).

    Args:
        scores: ``(n_samples, n_classes)`` scores.

    Returns:
        The row-wise softmax of the scores.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / np.sum(exponentials, axis=1, keepdims=True), dtype="float64")


__all__ = ["BaseTextClassifier", "TextPrediction", "softmax"]
