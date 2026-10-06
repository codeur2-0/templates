"""Données du résumé automatique : trois tables, un loader, des contrats Pandera.

Un corpus de résumé est **trois tables** : les comptes-rendus, leurs résumés de référence et les
**faits** que ces résumés doivent rapporter. Ce paquet expose la surface publique des trois et le
contrat qui les lie (un résumé par document, un fait rattaché à une phrase existante).
"""

from src.data.loaders import FACTS_STEM, METADATA_FILE, REFERENCES_STEM, SummaryCorpusLoader
from src.data.schemas import (
    DOC_ID_PATTERN,
    FACT_TYPES,
    INTERVENTION_TYPES,
    SITES,
    SPLITS,
    URGENCIES,
    DocumentsSchema,
    FactsSchema,
    PredictedSummariesSchema,
    ReferenceSummariesSchema,
    fact_columns,
    salient_fact_counts,
    split_sizes,
    validate_corpus,
    validate_documents,
    validate_facts,
    validate_predictions,
    validate_references,
)

__all__ = [
    "DOC_ID_PATTERN",
    "FACTS_STEM",
    "FACT_TYPES",
    "INTERVENTION_TYPES",
    "METADATA_FILE",
    "REFERENCES_STEM",
    "SITES",
    "SPLITS",
    "URGENCIES",
    "DocumentsSchema",
    "FactsSchema",
    "PredictedSummariesSchema",
    "ReferenceSummariesSchema",
    "SummaryCorpusLoader",
    "fact_columns",
    "salient_fact_counts",
    "split_sizes",
    "validate_corpus",
    "validate_documents",
    "validate_facts",
    "validate_predictions",
    "validate_references",
]
