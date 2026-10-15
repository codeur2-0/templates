"""Couche features de la classification de texte : traits de surface et test de raccourci.

Ces traits ne nourrissent pas le modèle — il lit le texte brut — ils servent à **vérifier** que ce
que le modèle apprend n'est pas déjà contenu dans une colonne triviale (longueur, canal, priorité).
"""

from src.features.build_features import (
    CATEGORICAL_CANDIDATES,
    TEXT_FEATURES,
    build_text_features,
    describe_features,
    feature_summary,
    group_accuracy,
    group_gain,
    shortcut_scores,
    stump_gain,
)

__all__ = [
    "CATEGORICAL_CANDIDATES",
    "TEXT_FEATURES",
    "build_text_features",
    "describe_features",
    "feature_summary",
    "group_accuracy",
    "group_gain",
    "shortcut_scores",
    "stump_gain",
]
