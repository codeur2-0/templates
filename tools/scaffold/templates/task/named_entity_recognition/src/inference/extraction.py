"""Mécanique d'extraction partagée : un message entre, une liste de mentions sortent.

Ce module existe pour une raison précise : la **latence publiée** doit être mesurée de la même
façon partout. Le trainer, l'évaluateur et le service d'inférence appellent donc la même boucle,
qui traite les messages **un par un** — comme un service les reçoit — et chronomètre chaque
appel. Un lot entier mesurerait un débit, pas un temps de réponse, et deux protocoles de
mesure dans le même projet rendraient les chiffres incomparables.

Les mentions rendues sont triées par position de lecture et ne se **chevauchent jamais** : une liste
de mentions est faite pour remplir un dossier, pas pour décrire un graphe d'ambiguïtés. C'est le
contrat de la stack (``BaseEntityTagger``) ; ce module le consomme, il ne le redéfinit pas.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.models.contract import EntityMention, sort_mentions

if TYPE_CHECKING:  # pragma: no cover - import de typage uniquement
    from src.models.contract import BaseEntityTagger

#: Columns of the mention table, in publication order.
MENTION_COLUMNS: tuple[str, ...] = (
    "msg_id",
    "start",
    "end",
    "label",
    "surface",
    "length",
    "source",
    "confidence",
)


def extract_one_by_one(
    model: BaseEntityTagger,
    documents: pd.DataFrame,
    *,
    text_column: str = "text",
    id_column: str = "msg_id",
) -> tuple[pd.DataFrame, list[float]]:
    """Extract the mentions of a frame, one message at a time, and time every call.

    Args:
        model: Fitted extractor.
        documents: Messages to annotate (text column and join key).
        text_column: Column holding the text.
        id_column: Column holding the message identifier.

    Returns:
        The mention table and the latency of every message, in milliseconds.

    Raises:
        ValueError: When the text column or the join key is missing.
    """
    missing = [name for name in (text_column, id_column) if name not in documents.columns]
    if missing:
        msg = f"Column(s) {missing} missing from the input frame: {sorted(documents.columns)}"
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for record in documents.to_dict(orient="records"):
        identifier = str(record[id_column])
        text = str(record[text_column])
        started = time.perf_counter()
        mentions: list[EntityMention] = model.predict([text], ids=[identifier])[0]
        latencies.append((time.perf_counter() - started) * 1000.0)
        rows.extend(mention.to_row() for mention in mentions)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(MENTION_COLUMNS)), latencies
    return frame, latencies


def deduplicate(mentions: Sequence[EntityMention]) -> list[EntityMention]:
    """Return the mentions without duplicates, in reading order.

    Args:
        mentions: Mentions to clean (a hybrid model can propose the same span twice).

    Returns:
        The deduplicated mentions, sorted by start offset then by end offset.
    """
    unique: dict[tuple[int, int, str], EntityMention] = {}
    for mention in mentions:
        unique.setdefault((mention.start, mention.end, mention.label), mention)
    return sort_mentions(list(unique.values()))


__all__ = ["MENTION_COLUMNS", "deduplicate", "extract_one_by_one"]
