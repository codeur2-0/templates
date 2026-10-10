"""Données de la reconnaissance d'entités : deux tables, un loader, des contrats Pandera.

Un corpus d'extraction d'entités est **deux tables** : les messages et leurs annotations. Ce paquet
expose la surface publique des deux, et le contrat qui les lie (jointure par identifiant, surface
égale à ``text[start:end]``, absence de chevauchement).
"""

from src.data.loaders import ANNOTATIONS_STEM, METADATA_FILE, EntityCorpusLoader
from src.data.schemas import (
    CANAUX,
    ENTITY_LABELS,
    MESSAGE_ID_PATTERN,
    NAME_LABELS,
    SOURCES,
    SPLITS,
    STYLES,
    MessagesSchema,
    PredictedMentionsSchema,
    SpansSchema,
    corpus_violations,
    describe_labels,
    holdout_share,
    label_counts,
    validate_corpus,
    validate_messages,
    validate_predictions,
    validate_spans,
)

__all__ = [
    "ANNOTATIONS_STEM",
    "CANAUX",
    "ENTITY_LABELS",
    "MESSAGE_ID_PATTERN",
    "METADATA_FILE",
    "NAME_LABELS",
    "SOURCES",
    "SPLITS",
    "STYLES",
    "EntityCorpusLoader",
    "MessagesSchema",
    "PredictedMentionsSchema",
    "SpansSchema",
    "corpus_violations",
    "describe_labels",
    "holdout_share",
    "label_counts",
    "validate_corpus",
    "validate_messages",
    "validate_predictions",
    "validate_spans",
]
