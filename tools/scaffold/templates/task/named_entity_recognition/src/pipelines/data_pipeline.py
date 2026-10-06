"""Génération du corpus annoté : messages, annotations et métadonnées.

``mode=generate-data`` (et ``make data``) exécute ce pipeline. C'est le seul endroit où le
générateur
de la famille est appelé et le seul qui écrit dans ``data/raw`` :

1. instancier le générateur depuis le nœud ``data`` de la configuration,
2. produire les messages **et** leurs annotations,
3. valider les deux tables contre leurs contrats Pandera **et** les liens entre elles (surface
exacte,
   absence de chevauchement, jointure) **avant** d'écrire quoi que ce soit,
4. persister (Parquet pour les machines, CSV pour les humains) et archiver la recette.

Les métadonnées ne sont pas décoratives : elles enregistrent la graine, les effectifs par type,
les
parts de styles rédactionnels, la part de surfaces **réservées** aux splits d'évaluation, la
**mémorisation des surfaces** (ce qu'une liste apprise pourrait couvrir, et rien de plus) et le
plancher trivial. Le rapport peut donc annoncer le niveau à battre au lieu de laisser croire
qu'une
F1 de 0,3 est un résultat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data.generators import SyntheticEntityCorpusGenerator
from src.data.loaders import EntityCorpusLoader
from src.data.schemas import ENTITY_LABELS, SPLITS, label_counts
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataGenerationPipeline(BasePipeline):
    """Générer -> valider les deux tables -> persister le corpus et sa recette."""

    name = "generate-data"

    def _execute(self) -> PipelineResult:
        """Run the generation flow.

        Returns:
            The pipeline result, whose payload is the generated corpus.
        """
        self.paths.ensure()
        data_config = self.config.data
        generator = SyntheticEntityCorpusGenerator.from_config(data_config)
        bundle = generator.generate()

        loader = EntityCorpusLoader(
            self.paths,
            dataset_name=str(data_config.dataset_name),
            formats=data_config.formats,
            validation_enabled=data_config.validation.raw,
            lazy_validation=data_config.validation.lazy,
        )
        written_documents = loader.save_documents(bundle.documents)
        written_annotations = loader.save_annotations(bundle.queries)
        metadata_path = loader.save_metadata(bundle.metadata)

        metrics = self._summarise(bundle.documents, bundle.queries)
        artifacts = [
            *[str(path) for path in written_documents.values()],
            *[str(path) for path in written_annotations.values()],
            str(metadata_path),
        ]
        logger.info(
            "Corpus '{}' generated: {} messages, {} entités, {} types",
            data_config.dataset_name,
            metrics["n_documents"],
            metrics["n_entities"],
            metrics["n_labels"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=artifacts,
            payload=bundle,
            messages=[
                f"{metrics['n_documents']:.0f} messages annotés, {metrics['n_entities']:.0f} "
                f"mentions ({metrics['entities_per_document']:.2f} par message) en "
                f"{metrics['n_labels']:.0f} types.",
                f"Référence triviale publiée avec le corpus : un système qui n'annote rien obtient "
                f"une F1 de {metrics['trivial_f1']:.2f}.",
                f"Surfaces réservées aux splits d'évaluation : {metrics['holdout_share']:.1%} des "
                f"mentions, soit {metrics['holdout_entities']:.0f} entités qu'une liste apprise ne "
                "peut pas retrouver.",
            ],
        )

    @staticmethod
    def _summarise(documents: pd.DataFrame, spans: pd.DataFrame) -> dict[str, float]:
        """Compute the summary metrics of a generated corpus.

        Args:
            documents: Generated message table.
            spans: Generated annotation table.

        Returns:
            Finite metrics describing the dataset (sizes, styles, holdout, trivial references).
        """
        counts = label_counts(spans)
        memorisation = _memorisation_reference(documents, spans)
        return {
            "n_documents": float(len(documents)),
            "n_entities": float(len(spans)),
            "n_labels": float(len(counts)),
            "entities_per_document": round(float(len(spans) / max(len(documents), 1)), 3),
            "n_tokens_mean": round(float(documents["n_tokens"].mean()), 2),
            "share_redige": round(float((documents["style"] == "redige").mean()), 4),
            "share_abrege": round(float((documents["style"] == "abrege").mean()), 4),
            "share_formulaire": round(float((documents["canal"] == "formulaire").mean()), 4),
            "share_chat": round(float((documents["canal"] == "chat").mean()), 4),
            "share_commande": round(counts["commande"] / max(len(spans), 1), 4),
            "share_produit": round(counts["produit"] / max(len(spans), 1), 4),
            "share_montant": round(counts["montant"] / max(len(spans), 1), 4),
            "share_date": round(counts["date"] / max(len(spans), 1), 4),
            "share_transporteur": round(counts["transporteur"] / max(len(spans), 1), 4),
            "holdout_share": round(float(spans["holdout"].mean()), 4),
            "holdout_entities": float(int(spans["holdout"].sum())),
            "memorisation_produit": memorisation["produit"],
            "memorisation_transporteur": memorisation["transporteur"],
            "memorisation_commande": memorisation["commande"],
            "memorisation_montant": memorisation["montant"],
            "memorisation_date": memorisation["date"],
            # Le plancher trivial est publié avec le corpus : 0,0 pour un système muet.
            "trivial_f1": 0.0,
        }


def _memorisation_reference(documents: pd.DataFrame, spans: pd.DataFrame) -> dict[str, float]:
    """Measure the share of evaluation surfaces that a training vocabulary already covers.

    Args:
        documents: Message table (``msg_id``, ``split``).
        spans: Annotation table (``msg_id``, ``label``, ``surface``).

    Returns:
        Per entity type, the share of evaluation mentions whose surface was annotated, for the
        same
        type, in the training split. Un montant ou une date inédits restent reconnaissables par
        leur
        forme ; un nom de produit inédit ne l'est pas — c'est la différence que la colonne
        chiffre.
    """
    split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
    annotated = spans.assign(split=spans["msg_id"].map(split_of))
    train = annotated[annotated["split"] == "train"]
    evaluation = annotated[annotated["split"].isin([name for name in SPLITS if name != "train"])]
    reference: dict[str, float] = {}
    for label in ENTITY_LABELS:
        seen = {
            str(surface).casefold() for surface in train.loc[train["label"] == label, "surface"]
        }
        subset = evaluation[evaluation["label"] == label]
        if subset.empty:
            reference[label] = 0.0
            continue
        # ``known`` est lié explicitement : une fermeture sur `seen` capturerait la dernière
        # valeur de la boucle (B023), donc la mémorisation serait mesurée sur le mauvais type.
        covered = subset["surface"].map(
            lambda surface, known=seen: str(surface).casefold() in known
        )
        reference[label] = round(float(covered.mean()), 4)
    return reference


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
