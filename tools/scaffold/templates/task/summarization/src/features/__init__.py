"""Représentation des documents : le graphe de phrases, construit une fois et testé.

Un résumé extractif choisit des phrases, pas des mots : ce paquet transforme un document en matrice de
similarité entre ses phrases, avec sa centralité, ses positions et ses longueurs. La baseline TextRank
et les notebooks lisent la même représentation, donc leurs scores parlent du même texte.
"""

from src.features.build_features import IDF_SMOOTHING, SentenceFeatureBuilder, SentenceFeatures

__all__ = ["IDF_SMOOTHING", "SentenceFeatureBuilder", "SentenceFeatures"]
