"""Traits de surface et test de raccourci : mesurer, pas supposer.

Le test de raccourci est la seule défense du projet contre une performance qui ne mesure rien —
un modèle qui aurait appris la longueur des tickets plutôt que leur contenu. Ces tests vérifient
que l'instrument fonctionne : il trouve un signal quand il y en a un (un trait construit pour
séparer deux classes), il ne trouve rien quand il n'y en a pas (la priorité, tirée au sort).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_the_surface_features_are_added_without_touching_the_input(
    documents: pd.DataFrame,
) -> None:
    """``build_text_features`` copie la frame par défaut : le corpus reste intact."""
    enriched = build_text_features(documents.head(20))

    assert set(TEXT_FEATURES).issubset(enriched.columns)
    assert not set(TEXT_FEATURES) & set(documents.columns)
    # ``list(...)`` est explicite : sous pandas 3, un tuple est une *seule* clé, plus une liste.
    assert enriched[list(TEXT_FEATURES)].notna().all().all()
    assert np.isfinite(enriched[list(TEXT_FEATURES)].to_numpy(dtype="float64")).all()


def test_the_in_place_variant_modifies_the_frame(documents: pd.DataFrame) -> None:
    """La variante en place sert aux pipelines qui ne veulent pas dupliquer le corpus."""
    frame = documents.head(10).copy()

    returned = build_text_features(frame, inplace=True)

    assert returned is frame
    assert set(TEXT_FEATURES).issubset(frame.columns)


def test_the_features_describe_the_text_they_received() -> None:
    """Un texte court, chiffré et signé doit activer exactement les traits attendus."""
    frame = pd.DataFrame(
        {
            "text": [
                "Bonjour, la facture CMD-12345 est erronée. Cordialement, Camille.",
                "Le colis n'est jamais arrivé.",
            ]
        }
    )

    enriched = build_text_features(frame)
    noisy, plain = enriched.iloc[0], enriched.iloc[1]

    assert noisy["has_greeting"] == 1.0
    assert noisy["has_signature"] == 1.0
    assert noisy["has_reference"] == 1.0
    assert noisy["digit_ratio"] > 0.0
    assert plain["has_greeting"] == 0.0
    assert plain["n_words"] < noisy["n_words"]


def test_a_missing_text_column_is_refused(documents: pd.DataFrame) -> None:
    """Sans colonne de texte, ces traits n'ont aucun sens : l'erreur doit le dire."""
    with pytest.raises(ValueError, match="Column 'body' missing"):
        build_text_features(documents, text_column="body")


def test_the_summary_averages_every_feature_per_class(documents: pd.DataFrame) -> None:
    """Le tableau par classe est celui que le rapport et le notebook impriment."""
    summary = feature_summary(build_text_features(documents))

    assert summary.index.name == "label"
    assert "all" in summary.index
    assert len(summary) == documents["label"].nunique() + 1
    assert set(TEXT_FEATURES).issubset(summary.columns)
    assert float(summary.loc["all", "support"]) == float(len(documents))


def test_a_separable_feature_is_found_by_the_stump() -> None:
    """Un trait qui sépare deux classes doit être détecté : l'instrument n'est pas aveugle."""
    frame = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            "label": ["a"] * 4 + ["b"] * 4,
        }
    )

    # Deux classes parfaitement séparées : la règle à un seuil fait 1,0 contre 0,5 pour la
    # meilleure constante, soit un gain de 0,5 — le maximum atteignable.
    assert stump_gain(frame, "signal") == pytest.approx(0.5)


def test_a_random_feature_is_not_a_shortcut() -> None:
    """Un trait tiré au sort ne fait pas mieux que la meilleure constante sur les mêmes lignes."""
    generator = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "noise": generator.normal(size=400),
            "label": ["a"] * 200 + ["b"] * 120 + ["c"] * 80,
            "channel": generator.choice(["x", "y", "z"], size=400),
        }
    )

    assert stump_gain(frame, "noise") < 0.1
    assert group_gain(frame, "channel") < 0.1


def test_the_priority_is_not_a_shortcut_for_the_label(documents: pd.DataFrame) -> None:
    """Distracteur déclaré : la priorité est tirée indépendamment du libellé."""
    scores = shortcut_scores(documents)

    assert scores["priority"] < 0.2


def test_the_shortcut_report_measures_every_candidate(documents: pd.DataFrame) -> None:
    """Le rapport de raccourci couvre les traits de surface et les colonnes catégorielles."""
    scores = shortcut_scores(documents)

    assert set(TEXT_FEATURES).issubset(scores)
    assert set(CATEGORICAL_CANDIDATES).issubset(scores)
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert scores["shortcut_margin"] == pytest.approx(
        max(scores[name] for name in (*TEXT_FEATURES, *CATEGORICAL_CANDIDATES)), abs=1e-12
    )
    assert scores["majority_baseline"] == pytest.approx(
        float(documents["label"].value_counts(normalize=True).max())
    )


def test_a_group_rule_can_be_right_by_construction() -> None:
    """``group_accuracy`` doit valider sa règle : un groupe = une classe donne 1.0."""
    frame = pd.DataFrame(
        {
            "channel": ["a", "a", "b", "b"],
            "label": ["x", "x", "y", "y"],
        }
    )

    assert group_accuracy(frame, "channel") == pytest.approx(1.0)


def test_the_features_are_documented() -> None:
    """Chaque trait a une description : un chiffre sans définition n'est pas une lecture."""
    descriptions = describe_features()

    assert set(descriptions) == set(TEXT_FEATURES)
    assert all(description for description in descriptions.values())
