"""Contrats Pandera du corpus de classification de texte.

Un projet de classification de texte tient dans **un seul tableau** : une ligne = un document, avec
son texte, son libellé et son découpage. Les contrats sont exécutables et disent trois choses qu'un
fichier de données ne dit jamais de lui-même :

* le **libellé fait partie du contrat** (`LABELS`) : une catégorie inventée par le générateur est
  une étiquette qui n'existe pour personne, et le rapport la publierait comme une classe ;
* le **découpage est une donnée** (`split`) : il est écrit par le générateur, jamais tiré au moment
  de l'entraînement, ce qui rend deux exécutions comparables ;
* la **colonne `priority` est un distracteur déclaré** : elle est indépendante du libellé, et la
  suite de tests le vérifie. Un modèle qui la prendrait pour un signal s'entraînerait sur du bruit.

Les payloads d'inférence ont leur propre contrat, relaxé : une prédiction est produite par le
projet, elle doit être complète, et elle est validée **avant** d'être écrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

#: Identifiers are part of the contract: artefacts are joined on those keys.
DOC_ID_PATTERN = r"^TKT-\d{4}$"

#: Categories of the support tickets. Six classes: a real support desk's first-level routing.
LABELS: tuple[str, ...] = (
    "facturation",
    "livraison",
    "produit_defectueux",
    "remboursement",
    "compte_client",
    "autre",
)

#: Where the ticket came from. Declared as a feature, never as a label: the channel correlates
#: with the *form* of the text, not with its subject.
SOURCES: tuple[str, ...] = ("formulaire", "email", "chat", "courrier")

#: Editorial style of the ticket. ``canonique`` reuses the vocabulary of its category,
#: ``paraphrase`` writes the same request without it, ``bruite`` adds boilerplate.
STYLES: tuple[str, ...] = ("canonique", "paraphrase", "bruite")

#: Declared distractor: independent of the label by construction.
PRIORITIES: tuple[str, ...] = ("basse", "normale", "haute")

#: Splits of the corpus. ``train`` fits, ``val`` monitors, ``calibration`` is used by the notebook
#: to settle the class weights, ``test`` measures once.
SPLITS: tuple[str, ...] = ("train", "val", "calibration", "test")

#: Columns of the probability matrix written next to each prediction.
def probability_columns(labels: tuple[str, ...] = LABELS) -> tuple[str, ...]:
    """Return the probability column names of a label set.

    Args:
        labels: Labels in display order.

    Returns:
        The ``p_<label>`` column names, in the same order.
    """
    return tuple(f"p_{label}" for label in labels)


class TicketsSchema(pa.DataFrameModel):
    """Contract of the labelled corpus (``data/raw/support_tickets.parquet``)."""

    doc_id: Series[str] = pa.Field(
        unique=True, str_matches=DOC_ID_PATTERN, description="Identifiant stable du ticket."
    )
    text: Series[str] = pa.Field(
        str_length={"min_value": 40, "max_value": 1200},
        description="Texte du ticket, tel qu'il a été reçu.",
    )
    label: Series[str] = pa.Field(isin=LABELS, description="Catégorie attendue par le support.")
    source: Series[str] = pa.Field(isin=SOURCES, description="Canal d'arrivée du ticket.")
    style: Series[str] = pa.Field(
        isin=STYLES, description="Style rédactionnel du ticket (canonique, paraphrase, bruité)."
    )
    priority: Series[str] = pa.Field(
        isin=PRIORITIES,
        description="Priorité déclarée : un distracteur, indépendant du libellé par construction.",
    )
    published_at: Series[pa.DateTime] = pa.Field(description="Date de réception du ticket.")
    n_tokens: Series[int] = pa.Field(ge=8, le=400, description="Nombre de tokens du texte.")
    split: Series[str] = pa.Field(
        isin=SPLITS, description="Découpage du corpus : une donnée, pas un tirage."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


class PredictedTicketsSchema(pa.DataFrameModel):
    """Contract of the prediction table (``artifacts/reports/predictions.csv``)."""

    doc_id: Series[str] = pa.Field(str_matches=DOC_ID_PATTERN, description="Ticket classé.")
    text: Series[str] = pa.Field(description="Texte classé (extrait conservé pour la relecture).")
    expected_label: Series[str] = pa.Field(
        nullable=True, description="Libellé de référence, vide à l'inférence."
    )
    prediction: Series[str] = pa.Field(isin=LABELS, description="Libellé produit par le modèle.")
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité de la décision.")
    correct: Series[int] = pa.Field(
        isin=[0, 1], description="1 lorsque la référence est connue et égale à la prédiction."
    )
    latency_ms: Series[float] = pa.Field(ge=0.0, description="Latence de la prédiction, en ms.")
    # Probabilités par classe : nommées explicitement, parce qu'un rapport qui lit une colonne
    # `p_<classe>` doit échouer bruyamment le jour où la liste des classes change de nom.
    p_facturation: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité facturation.")
    p_livraison: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité livraison.")
    p_produit_defectueux: Series[float] = pa.Field(
        ge=0.0, le=1.0, description="Probabilité produit défectueux."
    )
    p_remboursement: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité remboursement.")
    p_compte_client: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité compte client.")
    p_autre: Series[float] = pa.Field(ge=0.0, le=1.0, description="Probabilité autre demande.")

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False

    @pa.dataframe_check
    @classmethod
    def _probabilities_sum_to_one(cls, frame: pd.DataFrame) -> bool:
        """Vérifie que le bloc de probabilités est bien une distribution, ligne par ligne.

        Args:
            frame: Prediction frame (the ``p_*`` columns are required by the schema).

        Returns:
            ``True`` when every row sums to one, within the tolerance of a float32 artefact.
        """
        columns = [name for name in probability_columns() if name in frame.columns]
        if not columns:
            return False
        return bool(np.allclose(frame[columns].to_numpy(dtype="float64").sum(axis=1), 1.0, atol=1e-6))


def validate_tickets(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate a labelled corpus against :class:`TicketsSchema`.

    Args:
        frame: Corpus to validate.
        lazy: Collect every violation before raising (useful on a user-provided file).

    Returns:
        The validated frame (coerced dtypes).
    """
    return TicketsSchema.validate(frame, lazy=lazy)


def validate_predictions(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the prediction table against :class:`PredictedTicketsSchema`.

    Args:
        frame: Prediction rows.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return PredictedTicketsSchema.validate(frame, lazy=lazy)


__all__ = [
    "DOC_ID_PATTERN",
    "LABELS",
    "PRIORITIES",
    "SOURCES",
    "SPLITS",
    "STYLES",
    "PredictedTicketsSchema",
    "TicketsSchema",
    "probability_columns",
    "validate_predictions",
    "validate_tickets",
]
