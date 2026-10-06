"""Génération du corpus étiqueté et de ses métadonnées.

``mode=generate-data`` (et ``make data``) exécute ce pipeline. C'est le seul endroit où le
générateur de la famille est appelé et le seul qui écrit dans ``data/raw`` :

1. instancier le générateur depuis le nœud ``data`` de la configuration,
2. produire le corpus étiqueté et les métadonnées,
3. valider le tableau contre son contrat Pandera **avant** d'écrire quoi que ce soit,
4. persister (Parquet pour les machines, CSV pour les humains) et archiver la recette.

Les métadonnées ne sont pas décoratives : elles enregistrent la graine, les effectifs par classe,
les parts de styles rédactionnels et — surtout — les **références triviales** mesurées sur le
corpus (classe majoritaire, tirage stratifié). Le rapport peut donc annoncer le niveau à battre
plutôt que de laisser croire qu'une exactitude de 0,30 est un résultat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data.generators import SyntheticTicketGenerator
from src.data.loaders import TextLabelLoader
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataGenerationPipeline(BasePipeline):
    """Générer -> valider -> persister le corpus étiqueté."""

    name = "generate-data"

    def _execute(self) -> PipelineResult:
        """Run the generation flow.

        Returns:
            The pipeline result, whose metrics summarise the generated dataset.
        """
        self.paths.ensure()
        data_config = self.config.data
        generator = SyntheticTicketGenerator.from_config(data_config)
        bundle = generator.generate()

        loader = TextLabelLoader(
            self.paths,
            formats=data_config.formats,
            validation_enabled=data_config.validation.raw,
            lazy_validation=data_config.validation.lazy,
        )
        written = loader.save_documents(bundle.documents)
        metadata_path = loader.save_metadata(bundle.metadata)

        metrics = self._summarise(bundle.documents)
        artifacts = [*[str(path) for path in written.values()], str(metadata_path)]
        logger.info(
            "Corpus '{}' generated: {} tickets, {} classes",
            data_config.dataset_name,
            metrics["n_documents"],
            metrics["n_classes"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=artifacts,
            payload=bundle,
            messages=[
                f"{metrics['n_documents']:.0f} tickets étiquetés en "
                f"{metrics['n_classes']:.0f} classes, {metrics['n_tokens_mean']:.0f} tokens en "
                "moyenne.",
                f"Référence triviale (classe majoritaire) : {metrics['majority_accuracy']:.1%} "
                f"d'exactitude — le niveau à battre, publié avec le corpus.",
            ],
        )

    @staticmethod
    def _summarise(documents: pd.DataFrame) -> dict[str, float]:
        """Compute the summary metrics of a generated corpus.

        Args:
            documents: Generated corpus frame.

        Returns:
            Finite metrics describing the dataset (sizes, balance, trivial references).
        """
        counts = documents["label"].value_counts()
        majority_share = float(counts.iloc[0]) / float(len(documents))
        return {
            "n_documents": float(len(documents)),
            "n_classes": float(counts.size),
            "n_sources": float(documents["source"].nunique()),
            "n_tokens_mean": float(documents["n_tokens"].mean()),
            "n_tokens_median": float(documents["n_tokens"].median()),
            # La classe majoritaire donne l'exactitude d'un modèle constant : c'est le plancher
            # que tout le reste doit dépasser, et il est publié avec le corpus.
            "majority_accuracy": majority_share,
            "class_balance_ratio": float(counts.iloc[-1]) / float(counts.iloc[0]),
            "share_canonique": float((documents["style"] == "canonique").mean()),
            "share_paraphrase": float((documents["style"] == "paraphrase").mean()),
            "share_bruite": float((documents["style"] == "bruite").mean()),
        }


def summarise_metadata(payload: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the numeric entries of the generation metadata (used in the report).

    Args:
        payload: Metadata mapping written by the generator.

    Returns:
        Mapping of ``key`` to finite float values.
    """
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[str(key)] = float(value)
    return flattened


__all__ = ["DataGenerationPipeline", "summarise_metadata"]
