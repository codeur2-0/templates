"""Stratégies de résumé : contrat commun et baselines de la famille.

Ce paquet est la surface publique des stratégies **de la famille** — le contrat
(:class:`~src.models.contract.BaseTextGenerator`) et les deux baselines extractives
(:class:`~src.models.lead.LeadSummarizer`, :class:`~src.models.textrank.TextRankSummarizer`). La
stratégie servie et sa fabrique viennent de la couche de stack, qui complète ce paquet
(``build_model``, ``load_model``).
"""

from src.models.contract import BaseTextGenerator, TextSummary
from src.models.lead import LeadSummarizer
from src.models.textrank import MMR_GRID, TextRankSummarizer

__all__ = [
    "MMR_GRID",
    "BaseTextGenerator",
    "LeadSummarizer",
    "TextRankSummarizer",
    "TextSummary",
]
