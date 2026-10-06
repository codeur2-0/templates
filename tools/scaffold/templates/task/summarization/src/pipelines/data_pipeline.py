"""Génération du corpus : documents, résumés de référence, faits et métadonnées.

``mode=generate-data`` (et ``make data``) exécute ce pipeline. C'est le seul endroit où le générateur
de la famille est appelé et le seul qui écrit dans ``data/raw`` :

1. instancier le générateur depuis le nœud ``data`` de la configuration,
2. produire les documents, leurs résumés de référence **et** la table des faits,
3. valider les trois tables contre leurs contrats Pandera **et** les liens entre elles (résumé
   manquant, document inconnu, fait hors des phrases) **avant** d'écrire quoi que ce soit,
4. persister (Parquet pour les machines, CSV pour les humains) et archiver la recette.

Les métadonnées ne sont pas décoratives : elles enregistrent la graine, les effectifs par split, la
**compression demandée**, le taux de couverture qu'un résumé recopiant le document obtiendrait, et la
part de faits saillants par type. Le rapport peut donc situer un score au lieu de le présenter seul.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data.generators import SyntheticSummaryCorpusGenerator
from src.data.loaders import SummaryCorpusLoader
from src.data.schemas import FACT_TYPES, INTERVENTION_TYPES, SPLITS, split_sizes
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataGenerationPipeline(BasePipeline):
    """Générer -> valider les trois tables -> persister le corpus et sa recette."""

    name = "generate-data"

    def _execute(self) -> PipelineResult:
        """Run the generation flow.

        Returns:
            The pipeline result, whose payload is the generated corpus.
        """
        self.paths.ensure()
        data_config = self.config.data
        generator = SyntheticSummaryCorpusGenerator.from_config(data_config)
        bundle = generator.generate()

        loader = SummaryCorpusLoader(
            self.paths,
            dataset_name=str(data_config.dataset_name),
            formats=data_config.formats,
            validation_enabled=data_config.validation.raw,
            lazy_validation=data_config.validation.lazy,
        )
        written_documents = loader.save_documents(bundle.documents)
        written_references = loader.save_references(bundle.queries)
        written_facts = loader.save_facts(bundle.facts)
        metadata_path = loader.save_metadata(bundle.metadata)

        metrics = self._summarise(bundle.documents, bundle.queries, bundle.facts)
        artifacts = [
            *[str(path) for path in written_documents.values()],
            *[str(path) for path in written_references.values()],
            *[str(path) for path in written_facts.values()],
            str(metadata_path),
        ]
        logger.info(
            "Corpus '{}' generated: {} comptes-rendus, {} faits ({} saillants) en {} types",
            data_config.dataset_name,
            metrics["n_documents"],
            metrics["n_facts"],
            metrics["n_salient_facts"],
            len(FACT_TYPES),
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=artifacts,
            payload=bundle,
            messages=[
                f"{metrics['n_documents']:.0f} comptes-rendus, {metrics['n_references']:.0f} résumés "
                f"de référence ({metrics['compression_mean']:.1%} de la longueur du document en "
                "moyenne).",
                f"{metrics['n_salient_facts']:.0f} faits saillants annotés en "
                f"{len(FACT_TYPES)} types : la couverture d'un résumé est mesurable, pas supposée.",
                f"Référence publiée avec le corpus : un résumé vide obtient un ROUGE de "
                f"{metrics['trivial_rouge1']:.2f}, et recopier le document entier en obtiendrait "
                f"un proche de 1,0 sans rien résumer.",
            ],
        )

    @staticmethod
    def _summarise(
        documents: pd.DataFrame,
        references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute the summary metrics of a generated corpus.

        Args:
            documents: Generated document table.
            references: Generated reference summary table.
            facts: Generated fact table (reportable facts of every document).

        Returns:
            Finite metrics describing the dataset (sizes, compression, fact types, trivial floor).
        """
        salient = facts.loc[facts["salient"] == 1]
        metrics: dict[str, float] = {
            "n_documents": float(len(documents)),
            "n_references": float(len(references)),
            "n_facts": float(len(facts)),
            "n_salient_facts": float(len(salient)),
            "document_tokens_mean": round(float(documents["n_tokens"].mean()), 2),
            "reference_tokens_mean": round(float(references["n_tokens"].mean()), 2),
            "compression_mean": round(float(references["compression"].mean()), 4),
            "compression_max": round(float(references["compression"].max()), 4),
            "salient_facts_mean": round(float(references["n_salient_facts"].mean()), 2),
            "n_sentences_mean": round(float(documents["n_sentences"].mean()), 2),
            # Le plancher trivial est publié avec le corpus : 0,0 pour un résumé vide, et le plafond
            # par recopie est rappelé pour que personne ne lise un ROUGE élevé comme un résumé.
            "trivial_rouge1": 0.0,
            "copy_rouge1_ceiling": 1.0,
        }
        for name, count in split_sizes(documents).items():
            metrics[f"split_{name}"] = float(count)
        for name in FACT_TYPES:
            metrics[f"share_{name}"] = round(
                float((salient["fact_type"] == name).mean()) if len(salient) else 0.0,
                4,
            )
        for name in INTERVENTION_TYPES:
            metrics[f"share_{name}"] = round(
                float((documents["intervention_type"] == name).mean()), 4
            )
        return metrics


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


def split_names() -> tuple[str, ...]:
    """Splits declared by the family, in reading order.

    Returns:
        The split names (``train``, ``val``, ``calibration``, ``test``).
    """
    return SPLITS


__all__ = ["DataGenerationPipeline", "split_names", "summarise_metadata"]
