"""Contrats Pandera : ce que le corpus et la table de prédictions promettent.

Un contrat qui n'échoue jamais ne prouve rien : chaque test de ce module valide d'abord le cas
nominal, puis **casse** la donnée d'une façon précise (libellé inconnu, identifiant dupliqué,
probabilités qui ne somment plus à 1) et vérifie que le contrat refuse. C'est la seule façon de
savoir qu'une garde est branchée.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from src.data import LABELS, PRIORITIES, SOURCES, SPLITS, STYLES
from src.data.schemas import probability_columns, validate_predictions, validate_tickets
from src.features.build_features import shortcut_scores


def test_the_generated_corpus_satisfies_its_contract(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """Le corpus livré est accepté par le contrat, et le contrat ne le modifie pas."""
    frame = schema_frames["tickets"]

    validated = validate_tickets(frame)

    assert len(validated) == len(frame)
    assert set(LABELS).issuperset(validated["label"].unique())
    assert set(SOURCES).issuperset(validated["source"].unique())
    assert set(STYLES).issuperset(validated["style"].unique())
    assert set(PRIORITIES).issuperset(validated["priority"].unique())
    assert set(SPLITS).issuperset(validated["split"].unique())


def test_the_contract_refuses_a_ticket_without_a_known_label(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """Un libellé hors nomenclature est refusé : aucune classe fantôme ne peut entrer."""
    frame = schema_frames["tickets"].copy()
    frame.loc[frame.index[0], "label"] = "reclamation"

    with pytest.raises(pa.errors.SchemaError):
        validate_tickets(frame)


def test_the_contract_refuses_a_duplicated_identifier(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """Deux tickets ne partagent pas un identifiant : les jointures d'artefacts en dépendent."""
    frame = schema_frames["tickets"].copy()
    frame.loc[frame.index[1], "doc_id"] = str(frame.loc[frame.index[0], "doc_id"])

    with pytest.raises(pa.errors.SchemaError):
        validate_tickets(frame)


def test_the_contract_refuses_an_out_of_range_token_count(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """La longueur en tokens est bornée : un ticket de deux mots n'est pas un ticket."""
    frame = schema_frames["tickets"].copy()
    frame.loc[frame.index[0], "n_tokens"] = 3

    with pytest.raises(pa.errors.SchemaError):
        validate_tickets(frame)


def test_the_declared_priority_is_independent_of_the_label(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """La priorité déclarée ne doit rien apprendre du libellé : sinon c'est une cible déguisée.

    On mesure l'exactitude de la meilleure règle « une priorité -> une classe » et on la compare à
    celle d'un modèle constant. Une priorité qui aurait fui le libellé ferait bondir ce chiffre de
    plusieurs dizaines de points ; la borne de 25 points est donc un vrai test, pas une formalité.
    """
    scores = shortcut_scores(schema_frames["tickets"])

    assert scores["priority"] < 0.2
    assert scores["source"] < 0.2


def test_the_shortcut_scores_describe_the_surface_of_the_corpus(
    schema_frames: dict[str, pd.DataFrame],
) -> None:
    """Tous les traits candidats sont mesurés, et la marge de raccourci est publiée."""
    scores = shortcut_scores(schema_frames["tickets"])

    assert "majority_baseline" in scores
    assert "shortcut_margin" in scores
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    # La longueur seule ne doit pas suffire : le corpus est étiqueté par le *contenu*, pas par
    # la forme — sinon le projet publierait une F1 qui ne mesure que la verbosité des clients.
    # Le hasard a un niveau : la part de la classe majoritaire. Un trait de forme qui ne fait
    # pas mieux que ce niveau n'explique rien — et c'est ce que la marge de raccourci publie.
    assert scores["n_words"] < scores["majority_baseline"] + 0.05
    assert scores["n_chars"] < scores["majority_baseline"] + 0.05


def test_the_prediction_contract_accepts_a_full_prediction_block(
    predictions_frame: pd.DataFrame,
) -> None:
    """Une table de prédictions complète — probabilités comprises — passe le contrat."""
    validated = validate_predictions(predictions_frame)

    assert len(validated) == len(predictions_frame)
    assert set(probability_columns()).issubset(validated.columns)


def test_the_prediction_contract_refuses_probabilities_that_do_not_sum_to_one(
    predictions_frame: pd.DataFrame,
) -> None:
    """Le bloc de probabilités doit être une distribution : le rapport en tire une calibration."""
    frame = predictions_frame.copy()
    frame.loc[frame.index[0], "p_autre"] = 0.0

    with pytest.raises(pa.errors.SchemaError):
        validate_predictions(frame)


def test_the_prediction_contract_refuses_an_unknown_predicted_label(
    predictions_frame: pd.DataFrame,
) -> None:
    """Une prédiction hors nomenclature est refusée avant d'être écrite."""
    frame = predictions_frame.copy()
    frame.loc[frame.index[0], "prediction"] = "inconnu"

    with pytest.raises(pa.errors.SchemaError):
        validate_predictions(frame)
