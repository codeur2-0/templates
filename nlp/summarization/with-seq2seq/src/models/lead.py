"""Baseline triviale : les premières phrases du document, sans aucun apprentissage.

C'est la référence que tout lecteur a en tête quand il lit un score extractif — «
prendre le début du document marche presque toujours » — et c'est pour cela qu'elle est
**mesurée sur les mêmes lignes** que le modèle servi, puis publiée. Sur un compte-rendu
d'intervention, l'ouverture contient le contexte, le symptôme et souvent l'équipement :
la baseline n'est pas ridicule, et un modèle qui ne la bat pas n'apporte rien.

Deux nuances la distinguent d'un simple ``head(3)`` :

* le **budget est celui des autres stratégies** (même ``compression``, mêmes bornes) : sinon la
  comparaison mesurerait la longueur des résumés, pas leur contenu ;
* les phrases sont prises dans l'ordre du document jusqu'à saturation du budget, sans jamais couper
  une phrase : un résumé tronqué au milieu d'un mot serait un artefact, pas une baseline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.contract import BaseTextGenerator, TextSummary
from src.preprocessing.transformers import split_sentences
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LeadSummarizer(BaseTextGenerator):
    """Extractive floor: the opening sentences of the document, up to the length budget.

    Attributes:
        max_sentences: Maximum number of sentences read in the document.
    """

    strategy = "lead"

    def __init__(self, *, max_sentences: int = 40, **params: Any) -> None:
        """Configure the baseline.

        Args:
            max_sentences: Maximum number of sentences read from the document.
            params: Extra parameters forwarded to :class:`BaseTextGenerator`.
        """
        super().__init__(**params)
        self.max_sentences = int(max_sentences)

    def _fit(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        *,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Nothing to learn: the baseline is fitted by construction.

        Args:
            documents: Training documents (used only for the published count).
            references: Training references (unused).
            validation: Optional validation pair (unused).

        Returns:
            The number of documents seen, which documents the absence of any parameter.
        """
        del references, validation
        # Une baseline qui « s'entraîne » sans rien apprendre le dit explicitement : le nombre de
        # documents lus est la seule trace, et elle vaut zéro paramètre.
        return {"fit_documents": float(len(documents)), "n_parameters": 0.0}

    def _summarize(self, text: str, *, doc_id: str) -> TextSummary:
        """Summarise one document by taking its opening sentences.

        Args:
            text: Document text.
            doc_id: Document identifier.

        Returns:
            The extracted summary, in reading order.
        """
        sentences = split_sentences(text)[: self.max_sentences]
        document_tokens = len(text.split())
        budget = self._budget_tokens(document_tokens)
        selected: list[str] = []
        used = 0
        for sentence in sentences:
            length = len(sentence.split())
            if selected and used + length > budget:
                break
            selected.append(sentence)
            used += length
            if used >= budget:
                break
        summary = " ".join(selected).strip()
        return TextSummary(
            doc_id=str(doc_id),
            summary=summary,
            sentences=selected,
            budget_tokens=budget,
            n_tokens=used,
            document_tokens=document_tokens,
            compression=round(used / max(document_tokens, 1), 6),
            hit_max_length=False,
            metadata={"n_candidates": len(sentences)},
        )

    def _payload(self) -> dict[str, Any]:
        """State persisted with the artefact (a single bound, no learned parameter)."""
        return {"max_sentences": self.max_sentences}

    def _restore(self, payload: dict[str, Any]) -> None:
        """Restore the sentence bound from an artefact.

        Args:
            payload: Mapping written by :meth:`_payload`.
        """
        self.max_sentences = int(payload.get("max_sentences", self.max_sentences))

    def _extra_metadata(self) -> dict[str, Any]:
        """Baseline description published in the model card."""
        return {
            "algorithm": "lead",
            "n_parameters": 0,
            "max_sentences": self.max_sentences,
            "note": (
                "Les premières phrases du document, coupées au budget de longueur commun : "
                "le plancher extractif publié à côté des modèles appris."
            ),
        }


__all__ = ["LeadSummarizer"]
