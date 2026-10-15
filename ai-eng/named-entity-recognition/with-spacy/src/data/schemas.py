"""Contrats Pandera du corpus annoté : deux tables, et des liens entre elles.

Un projet de reconnaissance d'entités tient dans **deux tables** : les messages, et les annotations
qui disent où sont les entités dans ces messages. Les contrats sont exécutables et disent ce qu'un
fichier de données ne dit jamais de lui-même :

* le **type d'une entité fait partie du contrat** (:data:`ENTITY_LABELS`) : un type inventé par le
  générateur serait publié comme une classe du rapport, alors qu'il n'existe pour personne ;
* le **découpage est une donnée** (``split``) : il est écrit par le générateur, jamais tiré au moment
  de l'entraînement, ce qui rend deux exécutions comparables ;
* la **surface est la copie exacte du texte** : ``surface == text[start:end]`` est vérifié sur le
  corpus entier par :func:`validate_corpus`. Sans ce contrôle, une annotation décalée d'un caractère
  passerait inaperçue jusqu'à ce qu'un rapport explique une F1 inexplicable ;
* les **mentions ne se chevauchent pas** dans un message : une entité ambiguë n'est pas une
  information pour le métier, c'est une erreur de lecture.

Le drapeau ``holdout`` est lui aussi contractuel : il marque les mentions dont la surface a été
**réservée aux splits d'évaluation**. Le publier avec les données permet de mesurer la dégradation
train / test au lieu de la commenter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
import pandera as pa
from pandera.typing import Series

#: Identifiers are part of the contract: the two tables are joined on them.
MESSAGE_ID_PATTERN = r"^MSG-\d{4}$"

#: Types d'entités annotés dans le corpus. Cinq types : trois à écriture régulière (référence de
#: commande, montant, date) et deux qui reposent sur un **nom** (produit, transporteur).
ENTITY_LABELS: tuple[str, ...] = ("produit", "commande", "montant", "date", "transporteur")

#: Types dont la reconnaissance repose sur un nom : ce sont ceux qu'une surface réservée met en
#: défaut, et c'est cette distinction que la couche de règles exploite.
NAME_LABELS: tuple[str, ...] = ("produit", "transporteur")

#: Canaux d'arrivée d'un message. Déclarés comme features, jamais comme cibles.
CANAUX: tuple[str, ...] = ("formulaire", "email", "chat", "courrier")

#: Styles rédactionnels du corpus : ``redige`` écrit les surfaces complètes, ``abrege`` les abrège.
STYLES: tuple[str, ...] = ("redige", "abrege")

#: Splits du corpus. ``train`` ajuste, ``val`` suit, ``calibration`` sert la table de confiance du
#: notebook, ``test`` mesure une fois.
SPLITS: tuple[str, ...] = ("train", "val", "calibration", "test")

#: Provenances possibles d'une mention prédite.
SOURCES: tuple[str, ...] = ("regle", "modele")


class MessagesSchema(pa.DataFrameModel):
    """Contract of the message table (``data/raw/sav_messages.parquet``)."""

    msg_id: Series[str] = pa.Field(
        unique=True, str_matches=MESSAGE_ID_PATTERN, description="Identifiant stable du message."
    )
    text: Series[str] = pa.Field(
        str_length={"min_value": 40, "max_value": 400}, description="Texte du message, tel qu'écrit."
    )
    canal: Series[str] = pa.Field(isin=list(CANAUX), description="Canal d'arrivée du message.")
    style: Series[str] = pa.Field(isin=list(STYLES), description="Style rédactionnel du message.")
    n_entities: Series[int] = pa.Field(
        ge=3, le=5, description="Nombre d'entités annotées dans le message (jamais prédit)."
    )
    n_tokens: Series[int] = pa.Field(ge=8, description="Nombre de tokens du texte.")
    received_at: Series[pd.Timestamp] = pa.Field(description="Date de réception du message.")
    split: Series[str] = pa.Field(isin=list(SPLITS), description="Split du message.")

    class Config:
        """Schema configuration: strict columns, ordered checks."""

        strict = True
        ordered = False
        coerce = False

    @pa.dataframe_check
    @classmethod
    def split_is_declared(cls, frame: pd.DataFrame) -> bool:
        """Chaque split déclaré est présent : un corpus sans test ne mesure rien.

        Args:
            frame: Message table.

        Returns:
            ``True`` when every declared split appears in the corpus.
        """
        return set(SPLITS) <= set(frame["split"].astype(str))


class SpansSchema(pa.DataFrameModel):
    """Contract of the annotation table (``data/raw/spans.parquet``)."""

    msg_id: Series[str] = pa.Field(
        str_matches=MESSAGE_ID_PATTERN, description="Message annoté (clé de jointure)."
    )
    start: Series[int] = pa.Field(ge=0, description="Décalage du premier caractère de la mention.")
    end: Series[int] = pa.Field(gt=0, description="Décalage du premier caractère après la mention.")
    label: Series[str] = pa.Field(isin=list(ENTITY_LABELS), description="Type de l'entité.")
    surface: Series[str] = pa.Field(
        str_length={"min_value": 3}, description="Texte exact de la mention (``text[start:end]``)."
    )
    holdout: Series[bool] = pa.Field(
        description="Surface réservée aux splits d'évaluation (jamais vue à l'entraînement)."
    )

    class Config:
        """Schema configuration: strict columns, ordered checks."""

        strict = True
        ordered = False
        coerce = False

    @pa.dataframe_check
    @classmethod
    def bounds_are_ordered(cls, frame: pd.DataFrame) -> bool:
        """Une mention vide (``end <= start``) n'est pas une mention.

        Args:
            frame: Annotation table.

        Returns:
            ``True`` when every span has a strictly positive length.
        """
        return bool((frame["end"] > frame["start"]).all())


class PredictedMentionsSchema(pa.DataFrameModel):
    """Contract of the prediction table written by the inference pipeline."""

    msg_id: Series[str] = pa.Field(str_matches=MESSAGE_ID_PATTERN)
    start: Series[int] = pa.Field(ge=0)
    end: Series[int] = pa.Field(gt=0)
    label: Series[str] = pa.Field(isin=list(ENTITY_LABELS))
    surface: Series[str] = pa.Field(str_length={"min_value": 1})
    source: Series[str] = pa.Field(isin=list(SOURCES))
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config:
        """Schema configuration: the prediction table is produced by the project itself."""

        strict = False
        ordered = False
        coerce = True


def validate_messages(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the message table against its contract.

    Args:
        frame: Message table.
        lazy: When ``True``, collect every violation before raising.

    Returns:
        The validated frame.
    """
    return MessagesSchema.validate(frame, lazy=lazy)


def validate_spans(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the annotation table against its contract.

    Args:
        frame: Annotation table.
        lazy: When ``True``, collect every violation before raising.

    Returns:
        The validated frame.
    """
    return SpansSchema.validate(frame, lazy=lazy)


def validate_predictions(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate a prediction table before it is written.

    Args:
        frame: Prediction table.
        lazy: When ``True``, collect every violation before raising.

    Returns:
        The validated frame.
    """
    return PredictedMentionsSchema.validate(frame, lazy=lazy)


def corpus_violations(documents: pd.DataFrame, spans: pd.DataFrame) -> list[str]:
    """Return the cross-table violations of a corpus (empty when everything holds).

    Trois contrôles ne peuvent pas vivre dans un schéma de table, parce qu'ils lient **les deux**
    tables ou deux lignes entre elles :

    * toute annotation référence un message existant ;
    * ``surface`` est la copie exacte de ``text[start:end]`` ;
    * deux mentions d'un même message ne se chevauchent jamais.

    Args:
        documents: Message table (``msg_id``, ``text``).
        spans: Annotation table.

    Returns:
        A list of human readable violations, in reading order.
    """
    violations: list[str] = []
    texts = dict(zip(documents["msg_id"], documents["text"], strict=True))
    unknown = sorted({str(value) for value in spans["msg_id"]} - set(texts))
    for message_id in unknown:
        violations.append(f"annotation d'un message inconnu : {message_id}")
    grouped = spans.sort_values(["msg_id", "start", "end"]).groupby("msg_id", sort=True)
    for message_id, group in grouped:
        text = texts.get(str(message_id))
        if text is None:
            continue
        previous_end = -1
        for row in group.itertuples(index=False):
            start, end = int(row.start), int(row.end)
            surface, label = str(row.surface), str(row.label)
            if end > len(text):
                violations.append(
                    f"{message_id} : la mention {label} dépasse le texte ({end} > {len(text)})"
                )
                continue
            if text[start:end] != surface:
                violations.append(
                    f"{message_id} : la surface '{surface}' n'est pas text[{start}:{end}] "
                    f"('{text[start:end]}')"
                )
            if start < previous_end:
                violations.append(
                    f"{message_id} : la mention {label} chevauche la précédente "
                    f"({start} < {previous_end})"
                )
            previous_end = max(previous_end, end)
    return violations


def validate_corpus(
    documents: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    lazy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate both tables **and** the links between them.

    Args:
        documents: Message table.
        spans: Annotation table.
        lazy: When ``True``, collect every schema violation before raising.

    Returns:
        The validated tables.

    Raises:
        pandera.errors.SchemaErrors: When the tables break their contract.
        ValueError: When a cross-table constraint is violated.
    """
    validated_documents = validate_messages(documents, lazy=lazy)
    validated_spans = validate_spans(spans, lazy=lazy)
    violations = corpus_violations(validated_documents, validated_spans)
    if violations:
        head = "; ".join(violations[:5])
        extra = f" (+{len(violations) - 5} autres)" if len(violations) > 5 else ""
        msg = f"Corpus invalide : {head}{extra}"
        raise ValueError(msg)
    return validated_documents, validated_spans


def label_counts(spans: pd.DataFrame, *, labels: Sequence[str] = ENTITY_LABELS) -> dict[str, int]:
    """Count the mentions of every declared type.

    Args:
        spans: Annotation table.
        labels: Declared entity types (also the types whose count is reported as zero).

    Returns:
        Mapping of entity type to mention count, in the declared order.
    """
    counts = spans["label"].value_counts().to_dict() if not spans.empty else {}
    return {label: int(counts.get(label, 0)) for label in labels}


def holdout_share(spans: pd.DataFrame, *, labels: Sequence[str] = ENTITY_LABELS) -> dict[str, float]:
    """Return the share of reserved surfaces, per entity type.

    Args:
        spans: Annotation table.
        labels: Declared entity types.

    Returns:
        Mapping of entity type to the share of mentions whose surface was reserved.
    """
    shares: dict[str, float] = {}
    for label in labels:
        subset = spans[spans["label"] == label]
        shares[label] = round(float(subset["holdout"].mean()), 4) if not subset.empty else 0.0
    return shares


def describe_labels(labels: Iterable[str] = ENTITY_LABELS) -> dict[str, str]:
    """Return the documentation of the entity types (used by the reports).

    Args:
        labels: Declared entity types.

    Returns:
        Mapping of entity type to a one-line definition.
    """
    definitions = {
        "produit": "Nom d'un produit du catalogue, écrit en entier ou abrégé (nom, donc appris).",
        "commande": "Référence de commande, écrite « CMD-1234 » ou « cmd 1234 » (forme régulière).",
        "montant": "Somme en euros, écrite « 89,90 € », « 89.90 EUR » ou « 45 euros ».",
        "date": "Date d'un évènement, écrite « 12 mars 2025 » ou « 12/03/2025 ».",
        "transporteur": "Nom d'un transporteur, écrit en entier ou en minuscules (nom, donc appris).",
    }
    return {label: definitions.get(label, "Type déclaré du corpus.") for label in labels}


__all__ = [
    "CANAUX",
    "ENTITY_LABELS",
    "MESSAGE_ID_PATTERN",
    "NAME_LABELS",
    "SOURCES",
    "SPLITS",
    "STYLES",
    "MessagesSchema",
    "PredictedMentionsSchema",
    "SpansSchema",
    "corpus_violations",
    "describe_labels",
    "holdout_share",
    "label_counts",
    "validate_corpus",
    "validate_messages",
    "validate_predictions",
    "validate_spans",
]
