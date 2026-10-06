"""Traits de surface d'une mention, et le test de raccourci qui va avec.

Un extracteur d'entités apprend la *forme* des mentions ; ce module lit cette forme séparément :
longueur, part de chiffres, séparateurs, casse, ponctuation. Ces traits ne nourrissent pas le
modèle — un pipeline spaCy apprend ses propres représentations — mais ils répondent à une question
qu'un projet honnête doit poser avant de publier une F1 macro :

    quelle part de chaque type est reconnaissable par sa **forme seule**, sans lexique ?

C'est la thèse de cette famille, mise en chiffres (:func:`shape_shortcut_scores`) : une référence de
commande, un montant ou une date sont des **motifs** — leur signature est stable, donc une règle de
forme les retrouve même sur des valeurs inédites ; un produit ou un transporteur sont des **noms**
— la même règle plafonne, et surtout elle ne reconnaît aucune forme qu'elle n'a jamais vue. Un
rapport qui publie la mesure entière lit la bonne conclusion : le score du type minoritaire dit ce
que le modèle a appris, pas ce que l'écriture donnait déjà.

:func:`shape_separability` complète la lecture : la part des mentions dont la signature de forme
n'apparaît **que** pour un seul type. Une signature partagée par deux types (un numéro de commande
et un numéro de facture, un montant et une quantité) est exactement là où les erreurs de type se
concentrent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.training.metrics import entity_scores, precision_recall_f1
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Form features added to the mention table, in the order they are added.
SPAN_FEATURES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "digit_ratio",
    "letter_ratio",
    "upper_ratio",
    "n_separators",
    "has_digit",
    "has_currency",
    "has_month_name",
    "has_slash_date",
    "has_upper_token",
    "signature",
)

#: Form features added to the message table.
MESSAGE_FEATURES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "n_mentions",
    "mentions_per_100_chars",
    "annotated_char_ratio",
    "mean_mention_length",
)

#: Mots qui trahissent une date rédigée (les gabarits de la famille écrivent « 12 mars 2025 »).
MONTH_NAMES: tuple[str, ...] = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

#: Symboles et mots qui trahissent un montant.
CURRENCY_TOKENS: tuple[str, ...] = ("€", "eur", "euros", "euro")

#: Caractères qui séparent les blocs d'un identifiant ou d'une date.
SEPARATORS: tuple[str, ...] = ("-", "_", "/", " ", ".", ",")


def build_span_features(documents: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Return the annotation table enriched with the form features of every mention.

    Args:
        documents: Message table (``msg_id``, ``text``, ``split``, ``canal``, ``style``).
        spans: Annotation table (``msg_id``, ``start``, ``end``, ``label``, ``surface``).

    Returns:
        The annotation table with the ``SPAN_FEATURES`` columns appended.

    Raises:
        ValueError: When a mentioned message is missing from the message table.
    """
    known = set(documents["msg_id"].astype(str))
    missing = sorted({str(value) for value in spans["msg_id"]} - known)
    if missing:
        msg = f"Annotations reference unknown messages: {missing[:5]}"
        raise ValueError(msg)
    frame = spans.copy()
    described = [_describe_surface(str(surface)) for surface in frame["surface"]]
    for feature in SPAN_FEATURES:
        frame[feature] = [item[feature] for item in described]
    split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
    frame["split"] = frame["msg_id"].map(split_of)
    return frame


def build_message_features(documents: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Return the message table enriched with the features of its annotations.

    Args:
        documents: Message table.
        spans: Annotation table.

    Returns:
        The message table with the ``MESSAGE_FEATURES`` columns appended (coverage, density,
        length of the annotated surfaces).
    """
    frame = documents.copy()
    lookup = {str(row.msg_id): str(row.text) for row in frame.itertuples(index=False)}
    lengths: dict[str, list[int]] = {}
    covered: dict[str, int] = {}
    for row in spans.itertuples(index=False):
        identifier = str(row.msg_id)
        lengths.setdefault(identifier, []).append(int(row.end) - int(row.start))
        covered[identifier] = covered.get(identifier, 0) + (int(row.end) - int(row.start))
    identifiers = [str(value) for value in frame["msg_id"]]
    texts = [lookup.get(identifier, "") for identifier in identifiers]
    frame["n_chars"] = [len(text) for text in texts]
    frame["n_words"] = [len(text.split()) for text in texts]
    frame["n_mentions"] = [len(lengths.get(identifier, [])) for identifier in identifiers]
    frame["mentions_per_100_chars"] = [
        round(100.0 * len(lengths.get(identifier, [])) / max(len(text), 1), 3)
        for identifier, text in zip(identifiers, texts, strict=True)
    ]
    frame["annotated_char_ratio"] = [
        round(covered.get(identifier, 0) / max(len(text), 1), 4)
        for identifier, text in zip(identifiers, texts, strict=True)
    ]
    frame["mean_mention_length"] = [
        round(float(np.mean(lengths[identifier])), 2) if lengths.get(identifier) else 0.0
        for identifier in identifiers
    ]
    return frame


def feature_summary(frame: pd.DataFrame, *, by: str = "label") -> pd.DataFrame:
    """Aggregate the form features of a mention table.

    Args:
        frame: Frame produced by :func:`build_span_features`.
        by: Column to group by (``label`` by default).

    Returns:
        One row per group with the number of mentions and the mean of every numeric feature.

    Raises:
        ValueError: When the grouping column is missing.
    """
    if by not in frame.columns:
        msg = f"Column '{by}' missing from the feature table: {sorted(frame.columns)}"
        raise ValueError(msg)
    numeric = [name for name in SPAN_FEATURES if name != "signature"]
    available = [name for name in numeric if name in frame.columns]
    grouped = frame.groupby(by, observed=True)
    summary = grouped[available].mean(numeric_only=True).round(3)
    summary.insert(0, "n_mentions", grouped.size())
    return summary.reset_index()


def shape_shortcut_scores(
    documents: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    split: str = "val",
    fit_split: str = "train",
    id_column: str = "msg_id",
    text_column: str = "text",
) -> pd.DataFrame:
    """Measure the F1 reachable by a rule that only reads the **shape** of a mention.

    La règle est volontairement pauvre, et déclarée ici : **« telle signature de forme veut dire tel
    type »**. Elle est *apprise* sur ``fit_split`` (le type majoritaire de chaque signature) puis
    *notée* sur ``split``, en classant toutes les mentions qui portent une signature apprise. Une
    forme jamais vue à l'apprentissage n'est donc reconnue par personne : c'est exactement le sort
    d'un nom de produit ou de transporteur dont le corpus réserve la surface au split d'évaluation.
    Une règle notée sur le split qui l'a apprise mesurerait sa mémoire, pas sa lecture, d'où le
    refus de ``fit_split == split``.

    Le calcul se fait sur les mentions **annotées** (la règle reçoit les bornes du corpus) : ce
    n'est pas une performance d'extraction, c'est une mesure de lisibilité de la forme. La colonne
    ``share_by_shape`` dit la part du split que la règle sélectionne, ``n_unseen`` le nombre de
    mentions du type dont la forme est inédite, et ``f1`` ce que cette règle d'une ligne obtiendrait
    sur le type visé.

    Args:
        documents: Message table.
        spans: Annotation table.
        split: Split to measure on — the one the returned scores describe.
        fit_split: Split the signature-to-type rule is read from (never the measured one).
        id_column: Join key between the two tables.
        text_column: Text column of the message table.

    Returns:
        One row per entity type of the measured split: number of reference spans, distinct
        signatures, spans whose signature is unseen, the signature retained for the type, the
        number of spans selected, the share, and the precision / recall / F1 such a rule would
        obtain.

    Raises:
        ValueError: When a split holds no annotated message, or when ``fit_split`` equals ``split``.
    """
    if fit_split == split:
        msg = (
            f"Fit split and measured split are both '{split}': a shape rule scored on the split it "
            "was fitted on measures its memory, not its reading"
        )
        raise ValueError(msg)
    split_of = dict(zip(documents[id_column], documents["split"], strict=True))
    annotated = spans.assign(split=spans[id_column].map(split_of))
    annotated = annotated[annotated["split"].isin([split, fit_split])]
    for role, value in (("Measured", split), ("Fit", fit_split)):
        if not bool((annotated["split"] == value).any()):
            msg = f"{role} split '{value}' holds no annotated message: nothing to measure"
            raise ValueError(msg)
    features = build_span_features(documents, annotated)
    # La casse ne fait pas la forme : « CMD-1125 » et « cmd-1125 » sont le même motif.
    features["shape"] = features["signature"].astype(str).str.casefold()
    fitted = features[features["split"] == fit_split]
    scored = features[features["split"] == split]
    # La règle, en une ligne : chaque signature reçoit le type majoritaire observé à
    # l'apprentissage. Le tri (taille décroissante puis libellé) départage à égalité, donc le
    # résultat est stable.
    mapping = (
        fitted.groupby(["shape", "label"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["shape", "n", "label"], ascending=[True, False, True])
        .drop_duplicates("shape")
        .set_index("shape")["label"]
        .astype(str)
        .to_dict()
    )
    # L'ordre est **partagé** par la vérité et la prédiction : deux listes triées séparément
    # finiraient désalignées dès que la règle ne sélectionne rien dans un message.
    selection = documents.loc[documents["split"] == split, id_column]
    identifiers = sorted(str(value) for value in selection)
    gold = _span_sets(scored, identifiers)
    rows: list[dict[str, Any]] = []
    for label in sorted({str(value) for value in scored["label"]}):
        subset = scored[scored["label"].astype(str) == label]
        dominant = fitted.loc[fitted["label"].astype(str) == label, "shape"]
        signature = str(dominant.mode().iloc[0]) if not dominant.empty else ""
        selected = scored[scored["shape"].map(mapping) == label]
        # La règle **classe** : elle attribue le type visé à tout ce qui porte une signature
        # apprise pour ce type, donc une mention d'un autre type devient un faux positif.
        predicted = _span_sets(selected.assign(label=label), identifiers)
        scores = entity_scores(gold, predicted)
        counts = (
            _count(gold, predicted, label=label, kind="tp"),
            _count(gold, predicted, label=label, kind="fp"),
            _count(gold, predicted, label=label, kind="fn"),
        )
        per_label = precision_recall_f1(*counts)
        rows.append(
            {
                "label": label,
                "n_spans": len(subset),
                "n_signatures": int(subset["shape"].nunique()),
                "n_unseen": int((~subset["shape"].isin(mapping)).sum()),
                "signature": signature,
                "n_selected": len(selected),
                "share_by_shape": round(len(selected) / max(len(scored), 1), 4),
                "precision": per_label["precision"],
                "recall": per_label["recall"],
                "f1": per_label["f1"],
                "micro_f1_of_rule": scores["entity_f1"],
            }
        )
    return pd.DataFrame(rows)


def shape_separability(documents: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Measure how unambiguous each form signature is.

    Args:
        documents: Message table (used by :func:`build_span_features`).
        spans: Annotation table.

    Returns:
        One row per signature: how many mentions carry it, how many distinct types are labelled with
        it, and the share of its mentions that belong to its dominant type. Une signature partagée
        par plusieurs types est le lieu naturel des erreurs de type.
    """
    features = build_span_features(documents, spans)
    if features.empty:
        columns = ["signature", "n_spans", "n_labels", "dominant_label", "purity"]
        return pd.DataFrame(columns=columns)
    grouped = features.groupby("signature", observed=True)
    rows: list[dict[str, Any]] = []
    for signature, subset in grouped:
        counts = subset["label"].astype(str).value_counts()
        rows.append(
            {
                "signature": str(signature),
                "n_spans": len(subset),
                "n_labels": int(subset["label"].nunique()),
                "dominant_label": str(counts.index[0]),
                "purity": round(float(counts.iloc[0]) / len(subset), 4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_labels", "n_spans"], ascending=[True, False])
        .reset_index(drop=True)
    )


def describe_features(names: Sequence[str] | None = None) -> Mapping[str, str]:
    """Return the documentation of the form features (used by the notebooks).

    Args:
        names: Feature names to document (defaults to the whole registry).

    Returns:
        Mapping of feature name to its definition.
    """
    registry: dict[str, str] = {
        "n_chars": "Nombre de caractères de la surface.",
        "n_words": "Nombre de mots de la surface.",
        "digit_ratio": "Part des caractères qui sont des chiffres.",
        "letter_ratio": "Part des caractères qui sont des lettres.",
        "upper_ratio": "Part des caractères qui sont des majuscules.",
        "n_separators": "Nombre de séparateurs (tiret, barre, espace, point, virgule).",
        "has_digit": "1 si la surface contient au moins un chiffre.",
        "has_currency": "1 si la surface porte un symbole monétaire ou le mot « euro ».",
        "has_month_name": "1 si la surface contient un mois écrit en toutes lettres.",
        "has_slash_date": "1 si la surface contient une barre oblique (date abrégée).",
        "has_upper_token": "1 si la surface contient un mot tout en majuscules.",
        "signature": "Signature de forme : suites de classes de caractères (chiffres, majuscules).",
        "n_mentions": "Nombre de mentions annotées dans le message.",
        "mentions_per_100_chars": "Densité de mentions pour cent caractères de texte.",
        "annotated_char_ratio": "Part du texte couverte par des mentions annotées.",
        "mean_mention_length": "Longueur moyenne, en caractères, des mentions du message.",
    }
    if names is None:
        return registry
    return {name: registry.get(name, "Trait déclaré par le projet.") for name in names}


def _describe_surface(surface: str) -> dict[str, Any]:
    """Describe the form of one surface.

    Args:
        surface: Text of the mention.

    Returns:
        The ``SPAN_FEATURES`` of that surface, signature included.
    """
    text = str(surface)
    length = max(len(text), 1)
    lowered = text.casefold()
    letters = [character for character in text if character.isalpha()]
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "digit_ratio": round(sum(character.isdigit() for character in text) / length, 4),
        "letter_ratio": round(len(letters) / length, 4),
        "upper_ratio": round(sum(character.isupper() for character in text) / length, 4),
        "n_separators": sum(character in SEPARATORS for character in text),
        "has_digit": int(any(character.isdigit() for character in text)),
        "has_currency": int(any(token in lowered for token in CURRENCY_TOKENS) or "€" in text),
        "has_month_name": int(any(month in lowered for month in MONTH_NAMES)),
        "has_slash_date": int("/" in text),
        "has_upper_token": int(any(word.isupper() and len(word) > 1 for word in text.split())),
        "signature": _signature(text),
    }


def _signature(text: str) -> str:
    """Return the coarse character-class signature of a surface.

    Args:
        text: Surface to describe.

    Returns:
        A string such as ``AAA-####`` (``A`` majuscule, ``a`` minuscule, ``#`` chiffre, ``-``
        séparateur) : c'est la forme, pas le contenu — ce qu'une règle de forme peut lire.
    """
    characters: list[str] = []
    for character in text:
        if character.isdigit():
            characters.append("#")
        elif character.isupper():
            characters.append("A")
        elif character.islower():
            characters.append("a")
        elif character in SEPARATORS:
            characters.append("-")
        else:
            characters.append(".")
    compressed: list[str] = []
    for character in characters:
        if not compressed or compressed[-1] != character:
            compressed.append(character)
    return "".join(compressed)


def _span_sets(frame: pd.DataFrame, identifiers: Sequence[str]) -> list[set[tuple[int, int, str]]]:
    """Return one span set per identifier, in the order given.

    Args:
        frame: Annotation or mention table.
        identifiers: Message identifiers defining the order (**shared** by both sides of a
            comparison).

    Returns:
        One set of spans per identifier (empty when the frame holds none for it).
    """
    grouped: dict[str, set[tuple[int, int, str]]] = {}
    for row in frame.itertuples(index=False):
        grouped.setdefault(str(row.msg_id), set()).add(
            (int(row.start), int(row.end), str(row.label))
        )
    return [grouped.get(identifier, set()) for identifier in identifiers]


def _count(
    gold: Sequence[set[tuple[int, int, str]]],
    predicted: Sequence[set[tuple[int, int, str]]],
    *,
    label: str,
    kind: str,
) -> int:
    """Count the true positives, false positives or false negatives of one type.

    Args:
        gold: Reference spans, one set per message.
        predicted: Spans proposed by the shape rule, one set per message.
        label: Entity type to count.
        kind: ``tp``, ``fp`` or ``fn``.

    Returns:
        The count of that kind for that type.
    """
    true_positives = false_positives = false_negatives = 0
    for truth, guess in zip(gold, predicted, strict=True):
        expected = {span for span in truth if span[2] == label}
        found = {span for span in guess if span[2] == label}
        true_positives += len(expected & found)
        false_positives += len(found - expected)
        false_negatives += len(expected - found)
    return {"tp": true_positives, "fp": false_positives, "fn": false_negatives}[kind]


__all__ = [
    "CURRENCY_TOKENS",
    "MESSAGE_FEATURES",
    "MONTH_NAMES",
    "SEPARATORS",
    "SPAN_FEATURES",
    "build_message_features",
    "build_span_features",
    "describe_features",
    "feature_summary",
    "shape_separability",
    "shape_shortcut_scores",
]
