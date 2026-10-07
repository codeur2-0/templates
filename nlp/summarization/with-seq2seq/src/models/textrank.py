"""Baseline extractive : TextRank sur le graphe de phrases, puis sélection MMR.

La baseline sert deux fois dans ce projet : c'est le **plancher honnête**
contre lequel l'encodeur-décodeur est comparé (elle ne réécrit rien, donc elle
ne peut pas inventer de fait), et c'est le modèle de référence qui explique,
sur des cas concrets, ce qu'un résumé extractif peut et ne peut pas faire.

L'algorithme tient en deux étages, tous les deux déterministes :

1. **TextRank** — la similarité entre phrases est une matrice de poids ; on y fait tourner
   une marche aléatoire amortie (``damping``) jusqu'à convergence, ce qui donne à chaque
   phrase un score qui récompense celles qui ressemblent à *beaucoup* d'autres. Un graphe
   sans arête retombe sur une distribution uniforme plutôt que de renvoyer des zéros ;
2. **MMR** — sélectionner les phrases les plus centrales produit un résumé redondant, donc la
   sélection pénalise la ressemblance avec ce qui est déjà choisi (``mmr_lambda`` proche de 1
   favorise la centralité, proche de 0 la diversité). Les phrases retenues sont ensuite **remises
   dans l'ordre du document** : un résumé se lit, et la sortie d'un MMR brut ne se lit pas.

``fit`` ne calibre pas de poids : il choisit ``mmr_lambda`` sur le split de
validation, si on le lui donne, et publie la grille parcourue. C'est un
apprentissage d'un seul hyperparamètre, mesuré par ROUGE-1 F1 sur un échantillon
déterministe — la baseline a donc, elle aussi, une trace de ce qui a été arbitré.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.rouge import rouge_scores
from src.features.build_features import SentenceFeatureBuilder, SentenceFeatures
from src.models.contract import BaseTextGenerator, TextSummary
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Grid of MMR weights explored by :meth:`TextRankSummarizer._fit` when a validation split is given.
MMR_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)

#: Maximum number of documents used to calibrate the weight (the grid is cheap, but a fixed bound
#: keeps the fitting time identical from one machine to the next).
CALIBRATION_DOCUMENTS = 48


class TextRankSummarizer(BaseTextGenerator):
    """Extractive baseline: sentence graph, then diversity-aware selection.

    Attributes:
        damping: Damping factor of the random walk (the 0.85 of the original PageRank).
        iterations: Maximum number of random-walk iterations.
        tolerance: Convergence tolerance on the L1 norm of the score vector.
        mmr_lambda: Weight of centrality against redundancy in the selection.
        max_sentences: Maximum number of sentences kept by the graph builder.
    """

    strategy = "textrank"

    def __init__(
        self,
        *,
        damping: float = 0.85,
        iterations: int = 40,
        tolerance: float = 1e-8,
        mmr_lambda: float = 0.75,
        max_sentences: int = 40,
        builder: SentenceFeatureBuilder | None = None,
        **params: Any,
    ) -> None:
        """Configure the baseline.

        Args:
            damping: Damping factor of the random walk.
            iterations: Maximum number of random-walk iterations.
            tolerance: Convergence tolerance on the score vector.
            mmr_lambda: Weight of centrality in the MMR selection.
            max_sentences: Maximum number of sentences kept per document.
            builder: Optional pre-built feature builder (a configured preprocessor, in practice).
            params: Extra parameters forwarded to :class:`BaseTextGenerator`.
        """
        super().__init__(**params)
        if not 0.0 < damping < 1.0:
            msg = f"damping must lie in (0, 1), got {damping}"
            raise ValueError(msg)
        if iterations < 1:
            msg = f"iterations must be at least 1, got {iterations}"
            raise ValueError(msg)
        if not 0.0 <= mmr_lambda <= 1.0:
            msg = f"mmr_lambda must lie in [0, 1], got {mmr_lambda}"
            raise ValueError(msg)
        self.damping = float(damping)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.mmr_lambda = float(mmr_lambda)
        self.max_sentences = int(max_sentences)
        self.builder = builder or SentenceFeatureBuilder(max_sentences=max_sentences)

    # ------------------------------------------------------------------ entraînement -------
    def _fit(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        *,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Calibrate ``mmr_lambda`` on a deterministic sample of the validation split.

        Args:
            documents: Training documents (unused by the baseline, kept for the contract).
            references: Training references (unused by the baseline).
            validation: ``(documents, references)`` of the validation split, if available.

        Returns:
            The chosen weight and the ROUGE-1 F1 of the grid, plus the size of the sample.
        """
        if validation is None or validation[0].empty:
            logger.info(
                "TextRank: aucune validation fournie, mmr_lambda reste à {}", self.mmr_lambda
            )
            return {"mmr_lambda": self.mmr_lambda, "grid_documents": 0.0}
        val_documents, val_references = validation
        sample = val_documents.head(CALIBRATION_DOCUMENTS)
        golden = dict(zip(val_references["doc_id"], val_references["summary"], strict=False))
        sample = sample.loc[sample["doc_id"].isin(golden)].reset_index(drop=True)
        if sample.empty:
            return {"mmr_lambda": self.mmr_lambda, "grid_documents": 0.0}

        results: dict[float, float] = {}
        for weight in MMR_GRID:
            self.mmr_lambda = float(weight)
            predictions = [
                self._summarize(str(row.text), doc_id=str(row.doc_id)).summary
                for row in sample.itertuples(index=False)
            ]
            scores = [
                rouge_scores(golden[str(row.doc_id)], prediction).rouge1.f1
                for row, prediction in zip(sample.itertuples(index=False), predictions, strict=True)
            ]
            results[float(weight)] = round(float(np.mean(scores)), 4)
        best = max(results.items(), key=lambda item: (item[1], -item[0]))[0]
        self.mmr_lambda = float(best)
        # La grille complète est publiée : un hyperparamètre choisi sans trace n'est pas
        # reproductible, et le notebook 04 la relit telle quelle.
        metrics = {
            "mmr_lambda": self.mmr_lambda,
            "grid_documents": float(len(sample)),
            "fit_rouge1_f": results[float(best)],
        }
        for weight, score in sorted(results.items()):
            metrics[f"grid_lambda_{weight:.2f}"] = score
        return metrics

    # ------------------------------------------------------------------ génération ---------
    def _summarize(self, text: str, *, doc_id: str) -> TextSummary:
        """Summarise one document with TextRank + MMR.

        Args:
            text: Document text.
            doc_id: Document identifier.

        Returns:
            The extracted summary, sentences in reading order.
        """
        features = self.builder.build(doc_id, text)
        budget = self._budget_tokens(features.n_tokens)
        selected = self._select(features, budget)
        sentences = [features.sentences[index] for index in sorted(selected)]
        summary_text = " ".join(sentences).strip()
        n_tokens = int(sum(len(features.tokens[index]) for index in sorted(selected)))
        top = features.most_central_sentences(k=1)
        return TextSummary(
            doc_id=str(doc_id),
            summary=summary_text,
            sentences=sentences,
            budget_tokens=budget,
            n_tokens=n_tokens,
            document_tokens=features.n_tokens,
            compression=round(n_tokens / max(features.n_tokens, 1), 6),
            hit_max_length=False,
            metadata={
                "n_candidates": features.n_sentences,
                "top_sentence_index": top[0][0] if top else -1,
                "top_sentence_score": round(top[0][1], 6) if top else 0.0,
            },
        )

    def textrank_scores(self, features: SentenceFeatures) -> np.ndarray:
        """Run the damped random walk on the sentence graph.

        Args:
            features: Sentence representation of a document.

        Returns:
            The score vector (summing to one when the graph has edges, uniform otherwise).
        """
        n_sentences = features.n_sentences
        if n_sentences == 0:
            return np.zeros(0, dtype="float64")
        if n_sentences == 1:
            return np.ones(1, dtype="float64")
        column_sums = features.similarity.sum(axis=0)
        # Une colonne sans poids entrant ne reçoit rien : on la laisse à zéro plutôt que de diviser
        # par zéro, ce qui ferait diverger la marche sur un document sans terme répété.
        transition = np.divide(
            features.similarity,
            np.where(column_sums > 0.0, column_sums, 1.0),
        )
        scores = np.full(n_sentences, 1.0 / n_sentences, dtype="float64")
        for _ in range(self.iterations):
            updated = (1.0 - self.damping) / n_sentences + self.damping * (transition @ scores)
            delta = float(np.abs(updated - scores).sum())
            scores = updated
            if delta < self.tolerance:
                break
        total = float(scores.sum())
        return scores / total if total > 0.0 else np.full(n_sentences, 1.0 / n_sentences)

    def _select(self, features: SentenceFeatures, budget_tokens: int) -> list[int]:
        """Select the sentences of the summary with a budget-aware MMR.

        Args:
            features: Sentence representation of the document.
            budget_tokens: Maximum number of tokens of the summary.

        Returns:
            The selected sentence indices, in selection order.
        """
        n_sentences = features.n_sentences
        if n_sentences == 0:
            return []
        scores = self.textrank_scores(features)
        best = float(scores.max()) if n_sentences else 0.0
        normalised = scores / best if best > 0.0 else np.zeros(n_sentences, dtype="float64")

        available = set(range(n_sentences))
        selected: list[int] = []
        used_tokens = 0
        while available:
            best_index = -1
            best_value = float("-inf")
            for index in sorted(available):
                redundancy = max(
                    (features.similarity[index, chosen] for chosen in selected),
                    default=0.0,
                )
                value = self.mmr_lambda * normalised[index] - (1.0 - self.mmr_lambda) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index < 0:
                break
            sentence_tokens = int(features.length[best_index])
            # La première phrase est toujours retenue : un résumé vide n'est pas un résumé, même
            # quand le budget est plus court que la phrase la plus informative.
            if selected and used_tokens + sentence_tokens > budget_tokens:
                break
            selected.append(best_index)
            available.discard(best_index)
            used_tokens += sentence_tokens
            if used_tokens >= budget_tokens:
                break
        return selected

    # ------------------------------------------------------------------ persistance --------
    def _payload(self) -> dict[str, Any]:
        """State persisted with the artefact (the calibrated weight and the graph settings)."""
        return {
            "damping": self.damping,
            "iterations": self.iterations,
            "mmr_lambda": self.mmr_lambda,
            "max_sentences": self.max_sentences,
        }

    def _restore(self, payload: dict[str, Any]) -> None:
        """Restore the calibrated weight from an artefact.

        Args:
            payload: Mapping written by :meth:`_payload`.
        """
        self.damping = float(payload.get("damping", self.damping))
        self.iterations = int(payload.get("iterations", self.iterations))
        self.mmr_lambda = float(payload.get("mmr_lambda", self.mmr_lambda))
        self.max_sentences = int(payload.get("max_sentences", self.max_sentences))

    def _extra_metadata(self) -> dict[str, Any]:
        """Graph settings and grid traces published in the model card."""
        return {
            "algorithm": "textrank+mmr",
            "damping": self.damping,
            "iterations": self.iterations,
            "mmr_lambda": self.mmr_lambda,
            "max_sentences": self.max_sentences,
            "grid": list(MMR_GRID),
        }

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> TextRankSummarizer:
        """Build the baseline from the ``model`` node of the configuration.

        Args:
            config: ``model`` node (``params`` is used).
            overrides: Values that take precedence over the configuration.

        Returns:
            The configured baseline.
        """
        params = dict(getattr(config, "params", {}) or {})
        params.update(overrides)
        return cls(**params)

    def save(self, path: str | Path) -> Path:
        """Persist the baseline (see :meth:`BaseTextGenerator.save`).

        Args:
            path: Destination file or directory.

        Returns:
            The written artefact path.
        """
        return super().save(path)


__all__ = ["CALIBRATION_DOCUMENTS", "MMR_GRID", "TextRankSummarizer"]
