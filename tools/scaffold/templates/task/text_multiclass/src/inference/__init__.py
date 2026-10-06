"""Couche d'inférence de la classification de texte.

Un seul objet public, :class:`~src.inference.predictor.TextClassificationPredictor` : il
recharge un artefact, classe un lot de tickets, explique chaque décision terme à terme et écrit
une table de prédictions validée par le contrat Pandera avant d'être persistée.
"""

from src.inference.predictor import TextClassificationPredictor

__all__ = ["TextClassificationPredictor"]
