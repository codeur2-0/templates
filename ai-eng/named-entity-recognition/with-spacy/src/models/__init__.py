"""Couche modèles de la stack **spaCy**.

Surface publique, importée partout ailleurs dans le projet::

    from src.models import build_model, load_model        # fabrique + rechargement
    from src.models.contract import BaseEntityTagger      # contrat et objets de retour
    from src.models.factory import available_algorithms   # algorithmes servis par la stack

Rien en dehors de ce paquet n'importe spaCy : le framework reste un détail d'implémentation derrière
:class:`~src.models.contract.BaseEntityTagger` (inversion de dépendance), ce qui permet de comparer
une couche de règles et un tagger appris sur le même corpus sans toucher aux pipelines.
"""

from src.models.base import FitResult, ModelCard
from src.models.contract import BaseEntityTagger, EntityMention, sort_mentions, tracked_versions
from src.models.factory import (
    ALGORITHMS,
    AlgorithmSpec,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
    supported_tasks,
)
from src.models.rules import GazetteerIndex, RuleSpan, describe_rules, rule_mentions, rule_spans
from src.models.tagger import SpacyEntityTagger

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "BaseEntityTagger",
    "EntityMention",
    "FitResult",
    "GazetteerIndex",
    "ModelCard",
    "RuleSpan",
    "SpacyEntityTagger",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "describe_rules",
    "load_model",
    "rule_mentions",
    "rule_spans",
    "sort_mentions",
    "supported_tasks",
    "tracked_versions",
]
