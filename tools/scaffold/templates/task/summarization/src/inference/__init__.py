"""Inférence : résumer un corpus ou un fichier, publier la table et sa lecture rapide.

Le prédicteur publie ce qu'une génération vaut : longueurs, compression, couverture des faits quand le
document appartient au corpus annoté, et les valeurs que le document ne contient pas.
"""

from src.inference.predictor import DEFAULT_SAMPLE, SummaryPredictor, prediction_columns

__all__ = ["DEFAULT_SAMPLE", "SummaryPredictor", "prediction_columns"]
