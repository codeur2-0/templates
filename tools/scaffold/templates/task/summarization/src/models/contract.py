"""Contrat commun des générateurs de résumé, et objet rendu par chacun d'eux.

Un projet de résumé compare **des stratégies qui ne partagent rien** : une baseline extractive qui
choisit des phrases dans le document, un encodeur-décodeur qui les réécrit. Ce qu'elles partagent,
c'est ce contrat — ``fit`` sur les documents et leurs résumés de référence, ``summarize`` qui rend
un :class:`TextSummary` par document, ``model_card`` qui publie ce que le modèle a appris et sur
quoi il a été mesuré. Le reste du projet (évaluateur, rapport, notebooks, API de prédiction) ne
connaît que lui : ajouter une architecture ne touche ni l'évaluation, ni les figures, ni les tests.

Trois décisions sont dans ce fichier plutôt que dans chaque modèle :

* **le budget de longueur est une décision du modèle, pas du décodage** : ``max_output_tokens``
  borne la sortie, et ``compression`` fixe la longueur visée rapportée à celle du document ; un
  modèle qui ignore son budget produirait des résumés incomparables d'un document à l'autre ;
* **``hit_max_length`` est publié** : un décodeur qui bute systématiquement sur sa borne ne résume
  pas, il s'arrête, et le rapport doit pouvoir le dire ;
* **la latence est mesurée mais jamais un critère** : elle dépend de la machine, donc elle est
  enregistrée dans la fiche de modèle et exclue des verdicts.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 (second resolution)."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class TextSummary:
    """One generated summary, with everything the report needs to judge it.

    Attributes:
        doc_id: Document the summary was produced for.
        summary: The generated text.
        sentences: The generated text, split into sentences.
        strategy: Name of the strategy that produced it (``textrank``, ``transformer_tiny``…).
        budget_tokens: Length budget the strategy was given for this document.
        n_tokens: Number of tokens of the generated summary.
        document_tokens: Number of tokens of the source document.
        compression: ``n_tokens / document_tokens``.
        hit_max_length: Whether the generation stopped on its budget rather than on an end marker.
        latency_ms: Generation latency in milliseconds (machine-dependent, never a criterion).
        metadata: Extra fields published by the strategy (beam size, iterations…).
    """

    doc_id: str
    summary: str
    sentences: list[str] = field(default_factory=list)
    strategy: str = "generator"
    budget_tokens: int = 0
    n_tokens: int = 0
    document_tokens: int = 0
    compression: float = 0.0
    hit_max_length: bool = False
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten the summary into a row of the prediction table.

        Returns:
            A mapping with one scalar per column (no list, no nested object).
        """
        return {
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "prediction": self.summary,
            "n_sentences": len(self.sentences),
            "n_tokens": self.n_tokens,
            "compression": round(float(self.compression), 6),
            "hit_max_length": int(bool(self.hit_max_length)),
            "latency_ms": round(float(self.latency_ms), 3),
            **{f"meta_{key}": value for key, value in sorted(self.metadata.items())},
        }


class BaseTextGenerator(ABC):
    """Interface implemented by every summary strategy of the repository.

    Attributes:
        strategy: Short name of the strategy, written in every artefact.
        max_output_tokens: Hard bound on the summary length.
        min_output_tokens: Floor of the length budget (a two-word summary is not a summary).
        compression: Target length of the summary, relative to the document.
        seed: Seed used by the strategy (kept in the model card).
    """

    strategy: str = "generator"

    def __init__(
        self,
        *,
        max_output_tokens: int = 80,
        min_output_tokens: int = 12,
        compression: float = 0.18,
        seed: int = 42,
        **params: Any,
    ) -> None:
        """Store the decoding budget and the seed.

        Args:
            max_output_tokens: Hard bound on the summary length.
            min_output_tokens: Floor of the length budget.
            compression: Target length of the summary, relative to the document.
            seed: Seed of the strategy.
            params: Extra strategy parameters, kept for the model card.
        """
        if max_output_tokens < min_output_tokens:
            msg = (
                f"max_output_tokens ({max_output_tokens}) must be greater than or equal to "
                f"min_output_tokens ({min_output_tokens})"
            )
            raise ValueError(msg)
        if not 0.0 < compression <= 1.0:
            msg = f"compression must lie in (0, 1], got {compression}"
            raise ValueError(msg)
        self.max_output_tokens = int(max_output_tokens)
        self.min_output_tokens = int(min_output_tokens)
        self.compression = float(compression)
        self.seed = int(seed)
        self.params: dict[str, Any] = dict(params)
        self._is_fitted = False
        self._fit_metrics: dict[str, float] = {}
        self._fitted_at: str | None = None

    # ------------------------------------------------------------------ état --------------
    @property
    def is_fitted(self) -> bool:
        """Whether the strategy has been fitted (a baseline is fitted on construction)."""
        return bool(self._is_fitted)

    @property
    def state(self) -> str:
        """``fitted`` or ``untrained``, for logs and model cards."""
        return "fitted" if self.is_fitted else "untrained"

    @property
    def fit_metrics(self) -> dict[str, float]:
        """Metrics recorded by the last :meth:`fit` call."""
        return dict(self._fit_metrics)

    def summary(self) -> str:
        """One-line description of the strategy (used in logs and reports)."""
        return (
            f"{self.strategy} [{self.state}] budget {self.min_output_tokens}-"
            f"{self.max_output_tokens} tokens, compression {self.compression:.0%}"
        )

    def __repr__(self) -> str:
        """Developer representation, deterministic (no memory address)."""
        return f"{type(self).__name__}(strategy={self.strategy!r}, state={self.state!r})"

    def check_is_fitted(self) -> None:
        """Raise when the strategy has not been fitted yet.

        Raises:
            RuntimeError: When :meth:`fit` has not run.
        """
        if not self.is_fitted:
            msg = (
                f"{type(self).__name__} is not fitted: call fit(documents, references) first, or "
                f"build the '{self.strategy}' strategy through the factory of the project."
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ api ---------------
    def fit(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        *,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Fit the strategy on the training split.

        Args:
            documents: Training documents (``doc_id``, ``text``, ``n_tokens``).
            references: Reference summaries of the same documents.
            validation: Optional ``(documents, references)`` pair used to monitor the fit; a
                baseline ignores it, a learned model reports a validation ROUGE.

        Returns:
            The metrics recorded during the fit (possibly empty for a baseline).
        """
        metrics = self._fit(documents, references, validation=validation)
        self._is_fitted = True
        self._fit_metrics = {key: float(value) for key, value in metrics.items()}
        self._fitted_at = _utc_now()
        logger.info("{} fitted: {}", type(self).__name__, self._fit_metrics or "aucune métrique")
        return self.fit_metrics

    def summarize(self, texts: list[str], *, doc_ids: list[str] | None = None) -> list[TextSummary]:
        """Summarise a batch of raw texts.

        Args:
            texts: Documents to summarise.
            doc_ids: Identifiers, in the same order as
                ``texts`` (defaults to ``doc-0``, ``doc-1``…).

        Returns:
            One :class:`TextSummary` per text, in input order.
        """
        self.check_is_fitted()
        identifiers = (
            doc_ids if doc_ids is not None else [f"doc-{index}" for index in range(len(texts))]
        )
        if len(identifiers) != len(texts):
            msg = f"doc_ids has {len(identifiers)} entries for {len(texts)} texts"
            raise ValueError(msg)
        summaries: list[TextSummary] = []
        for doc_id, text in zip(identifiers, texts, strict=True):
            started = time.perf_counter()
            summary = self._summarize(str(text), doc_id=str(doc_id))
            elapsed = (time.perf_counter() - started) * 1000.0
            summary.strategy = self.strategy
            summary.doc_id = str(doc_id)
            summary.latency_ms = round(elapsed, 3)
            if summary.n_tokens == 0:
                summary.n_tokens = len(summary.summary.split())
            if summary.document_tokens == 0:
                summary.document_tokens = len(str(text).split())
            summary.compression = round(summary.n_tokens / max(summary.document_tokens, 1), 6)
            summaries.append(summary)
        return summaries

    def summarize_frame(
        self, documents: pd.DataFrame, *, text_column: str = "text", id_column: str = "doc_id"
    ) -> pd.DataFrame:
        """Summarise every document of a frame.

        Args:
            documents: Corpus with an identifier column and a text column.
            text_column: Name of the text column.
            id_column: Name of the identifier column.

        Returns:
            A frame with one row per document: ``doc_id``, ``strategy``, ``prediction``, budget and
            compression columns.
        """
        identifiers = [str(value) for value in documents[id_column]]
        texts = [str(value) for value in documents[text_column]]
        rows = [item.to_row() for item in self.summarize(texts, doc_ids=identifiers)]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ artefacts ---------
    def model_card(self) -> dict[str, Any]:
        """Describe the strategy: what it is, how it was configured, what it scored.

        Returns:
            A JSON-serialisable mapping (published as ``artifacts/models/model_card.json``).
        """
        payload: dict[str, Any] = {
            "strategy": self.strategy,
            "class": type(self).__name__,
            "state": self.state,
            "fitted_at": self._fitted_at,
            "budget": {
                "min_output_tokens": self.min_output_tokens,
                "max_output_tokens": self.max_output_tokens,
                "compression": self.compression,
            },
            "seed": self.seed,
            "fit_metrics": self.fit_metrics,
            "params": dict(self.params),
        }
        payload.update(self._extra_metadata())
        return payload

    def save(self, path: str | Path) -> Path:
        """Persist the strategy (joblib payload plus a JSON model card).

        Args:
            path: Destination file, or directory when it ends with ``/``.

        Returns:
            The written artefact path.
        """
        self.check_is_fitted()
        target = Path(path)
        if target.suffix == "" or target.name.endswith("/"):
            target = target / "generator"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "class": type(self).__name__,
            "budget": {
                "min_output_tokens": self.min_output_tokens,
                "max_output_tokens": self.max_output_tokens,
                "compression": self.compression,
            },
            "seed": self.seed,
            "fit_metrics": self.fit_metrics,
            "params": dict(self.params),
            "extra": self._extra_metadata(),
            "payload": self._payload(),
        }
        joblib.dump(payload, target)
        card_path = target.with_name("model_card.json")
        card_path.write_text(
            json.dumps(self.model_card(), indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Artefact écrit : {}", target)
        return target

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> BaseTextGenerator:
        """Reload a persisted strategy.

        Args:
            path: Artefact written by :meth:`save`.
            overrides: Values that take precedence over the persisted ones (the project's Hydra
                configuration is authoritative: an artefact moved to another project must obey the
                new configuration).

        Returns:
            The restored strategy.

        Raises:
            FileNotFoundError: When the artefact does not exist.
        """
        target = Path(path)
        if not target.is_file():
            msg = (
                f"Model artefact not found: {target}. Run `make train` (or "
                "`python -m src.main mode=train`) to produce it."
            )
            raise FileNotFoundError(msg)
        payload = joblib.load(target)
        budget = dict(payload.get("budget", {}))
        params = {**payload.get("params", {}), **budget, **overrides}
        instance = cls(**params)
        instance._restore(payload.get("payload", {}))
        instance._is_fitted = True
        instance._fit_metrics = {
            str(key): float(value) for key, value in dict(payload.get("fit_metrics", {})).items()
        }
        instance._fitted_at = payload.get("fitted_at")
        return instance

    # ------------------------------------------------------------------ hooks -------------
    def _budget_tokens(self, document_tokens: int) -> int:
        """Length budget of one document, bounded by the strategy's floor and ceiling.

        Args:
            document_tokens: Number of tokens of the source document.

        Returns:
            The target number of tokens of the summary.
        """
        target = round(self.compression * max(int(document_tokens), 1))
        return int(min(max(target, self.min_output_tokens), self.max_output_tokens))

    @abstractmethod
    def _fit(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        *,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Strategy-specific fitting; called by :meth:`fit`."""

    @abstractmethod
    def _summarize(self, text: str, *, doc_id: str) -> TextSummary:
        """Strategy-specific summarisation of one text; called by :meth:`summarize`."""

    @abstractmethod
    def _payload(self) -> dict[str, Any]:
        """State that must be persisted to reload the strategy."""

    def _restore(self, payload: dict[str, Any]) -> None:
        """Restore the strategy state; a stateless strategy has nothing to do.

        Args:
            payload: Mapping written by :meth:`_payload`.
        """
        return None

    def _extra_metadata(self) -> dict[str, Any]:
        """Strategy-specific entries of the model card.

        Returns:
            A JSON-serialisable mapping (empty by default).
        """
        return {}


__all__ = [
    "BaseTextGenerator",
    "TextSummary",
]
