"""Couche données du résumé : contrats Pandera, chargeurs et vérifications croisées.

Trois tables vivent ici et se lisent ensemble : les documents, leurs résumés de référence et les
faits annotés. Les contrats sont stricts (types, bornes, valeurs autorisées) et
:func:`~src.data.schemas.validate_corpus` vérifie **les liens** entre les tables — un résumé de
référence orphelin, un document sans résumé ou un fait qui pointe hors de son document sont les
trois erreurs qui biaisent silencieusement un ROUGE, donc elles sont refusées avant toute mesure.

Le générateur du corpus vit dans la couche *famille* (``src.data.generators``) : ce paquet ne
l'importe pas, ce qui permet de lire un corpus persisté sans embarquer le générateur.
"""

from src.data.loaders import SummaryCorpusLoader
from src.data.schemas import (
    FACT_TYPES,
    DocumentsSchema,
    FactsSchema,
    PredictedSummariesSchema,
    ReferenceSummariesSchema,
    salient_fact_counts,
    split_sizes,
    validate_corpus,
    validate_documents,
    validate_facts,
    validate_predictions,
    validate_references,
)

__all__ = [
    "FACT_TYPES",
    "DocumentsSchema",
    "FactsSchema",
    "PredictedSummariesSchema",
    "ReferenceSummariesSchema",
    "SummaryCorpusLoader",
    "salient_fact_counts",
    "split_sizes",
    "validate_corpus",
    "validate_documents",
    "validate_facts",
    "validate_predictions",
    "validate_references",
]
