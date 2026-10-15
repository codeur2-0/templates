"""Traits de surface d'un ticket, et le test de raccourci qui va avec.

Un classifieur de texte lit des mots ; ce module lit la **forme** : longueur, part de chiffres,
majuscules, ponctuation, présence d'une formule de politesse ou d'une signature. Ces traits ne
servent pas à classer — le corpus est étiqueté, pas annoté en longueur — mais à répondre à une
question qu'un projet honnête doit se poser avant de publier une F1 :

    le modèle lit-il la demande, ou seulement la longueur du ticket ?

C'est le **test de raccourci** (:func:`shortcut_scores`) : pour chaque trait, on mesure la
meilleure exactitude atteignable par une règle d'une ligne — un seuil (trait numérique) ou la
classe majoritaire du groupe (trait catégoriel). Si un trait seul atteint l'exactitude du modèle,
la performance publiée ne mesure pas ce qu'elle prétend mesurer, et le rapport doit le dire.

Le corpus est construit pour que ce test ait un sens : ``priority`` est tirée **indépendamment**
du libellé, donc son score doit rester au niveau de la classe majoritaire ; ``style``, lui, est un
vrai signal — les tickets canoniques sont plus faciles — et c'est bien pour cela qu'il est déclaré
et publié, pas caché.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.preprocessing.transformers import normalise_text, split_sentences, tokenize
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Features derived from the text, in the order they are added to the frame.
TEXT_FEATURES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "n_sentences",
    "mean_word_length",
    "digit_ratio",
    "uppercase_ratio",
    "punctuation_ratio",
    "has_greeting",
    "has_signature",
    "has_reference",
)

#: Opening formulas the corpus can prepend (see the family generator).
GREETINGS: tuple[str, ...] = ("bonjour", "bonsoir", "rebonjour", "salut")

#: Closing formulas the corpus can append.
SIGNATURES: tuple[str, ...] = ("cordialement", "merci d'avance", "bonne journée", "bien à vous")

#: Columns the shortcut test inspects besides the text features: two declared distractors
#: (``source``, ``priority``) and one declared signal (``style``).
CATEGORICAL_CANDIDATES: tuple[str, ...] = ("source", "style", "priority")


def build_text_features(
    documents: pd.DataFrame,
    *,
    text_column: str = "text",
    inplace: bool = False,
) -> pd.DataFrame:
    """Add the surface features of every text to a frame.

    Args:
        documents: Frame holding at least ``text_column``.
        text_column: Column holding the text to describe.
        inplace: When ``True``, modify the frame instead of copying it.

    Returns:
        The frame with one column per entry of :data:`TEXT_FEATURES`.

    Raises:
        ValueError: When the text column is missing.
    """
    if text_column not in documents.columns:
        msg = f"Column '{text_column}' missing from the frame: {sorted(documents.columns)}"
        raise ValueError(msg)
    frame = documents if inplace else documents.copy()
    texts = [normalise_text(str(value)) for value in frame[text_column]]
    rows = [_describe_text(text) for text in texts]
    for name in TEXT_FEATURES:
        frame[name] = np.asarray([row[name] for row in rows], dtype="float64")
    logger.debug("{} surface features added to {} rows", len(TEXT_FEATURES), len(frame))
    return frame


def feature_summary(
    documents: pd.DataFrame,
    *,
    label_column: str = "label",
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Average every feature per class (the table the notebook and the report print).

    Args:
        documents: Labelled frame, surface features included.
        label_column: Column holding the label.
        features: Features to summarise (defaults to :data:`TEXT_FEATURES`).

    Returns:
        One row per class, one column per feature, plus a synthetic ``all`` row.

    Raises:
        ValueError: When the label column or a requested feature is missing.
    """
    if label_column not in documents.columns:
        msg = f"Column '{label_column}' missing from the frame: {sorted(documents.columns)}"
        raise ValueError(msg)
    names = list(features or TEXT_FEATURES)
    missing = [name for name in names if name not in documents.columns]
    if missing:
        msg = f"Features missing from the frame: {missing}. Call build_text_features first."
        raise ValueError(msg)
    grouped = documents.groupby(documents[label_column].astype(str), observed=True)[names].mean()
    grouped.insert(0, "support", documents[label_column].astype(str).value_counts().sort_index())
    overall = documents[names].mean()
    overall_row = pd.concat([pd.Series({"support": float(len(documents))}), overall]).to_frame().T
    overall_row.index = pd.Index(["all"], name=label_column)
    summary = pd.concat([grouped, overall_row])
    summary.index.name = label_column
    return summary


def stump_gain(
    documents: pd.DataFrame,
    feature: str,
    *,
    label_column: str = "label",
) -> float:
    """Gain of the best one-threshold rule over the best constant rule, on the same rows.

    Une règle à un seuil sépare **deux** classes : les autres n'ont pas de rôle dans la règle. La
    seule comparaison honnête est donc de mesurer l'exactitude de la règle sur ces deux classes, et
    de la comparer à ce qu'un modèle constant obtiendrait sur ces mêmes lignes. Sinon le score
    s'envole artificiellement : il suffit de choisir la paire complémentaire de la classe
    majoritaire pour dépasser la référence sans rien apprendre.

    Args:
        documents: Labelled frame holding ``feature``.
        feature: Numeric feature to test.
        label_column: Column holding the label.

    Returns:
        The best gain over the constant rule on the same pair of classes, in ``[0, 1]``. A random
        feature scores near 0; a feature that separates two classes perfectly scores near 0.5.

    Raises:
        ValueError: When the feature or the label column is missing.
    """
    for column in (feature, label_column):
        if column not in documents.columns:
            msg = f"Column '{column}' missing from the frame: {sorted(documents.columns)}"
            raise ValueError(msg)
    values = documents[feature].to_numpy(dtype="float64")
    labels = documents[label_column].astype(str).to_numpy()
    best = 0.0
    classes = sorted(set(labels.tolist()))
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1 :]:
            mask = (labels == left) | (labels == right)
            pair_values = values[mask]
            pair_labels = labels[mask]
            if pair_values.size < 4:
                continue
            counts = np.unique(pair_labels, return_counts=True)[1]
            reference = float(counts.max()) / float(pair_labels.size)
            candidates = np.unique(pair_values)
            thresholds = (
                (candidates[:-1] + candidates[1:]) / 2.0 if candidates.size > 1 else candidates
            )
            for threshold in thresholds:
                rule = np.where(pair_values <= threshold, left, right)
                gain = float(np.mean(rule == pair_labels)) - reference
                best = max(best, gain)
    return min(max(best, 0.0), 1.0)


def group_accuracy(
    documents: pd.DataFrame,
    feature: str,
    *,
    label_column: str = "label",
) -> float:
    """Best accuracy a "majority label per group" rule reaches with a categorical feature.

    Args:
        documents: Labelled frame holding ``feature``.
        feature: Categorical feature to test.
        label_column: Column holding the label.

    Returns:
        The accuracy of the rule, in ``[0, 1]``.

    Raises:
        ValueError: When the feature or the label column is missing.
    """
    for column in (feature, label_column):
        if column not in documents.columns:
            msg = f"Column '{column}' missing from the frame: {sorted(documents.columns)}"
            raise ValueError(msg)
    frame = pd.DataFrame(
        {
            "group": documents[feature].astype(str),
            "label": documents[label_column].astype(str),
        }
    )
    counts = frame.groupby(["group", "label"], observed=True).size().rename("n").reset_index()
    winners = counts.sort_values("n", ascending=False).drop_duplicates("group")
    predicted = frame["group"].map(dict(zip(winners["group"], winners["label"], strict=True)))
    return float(np.mean(predicted.to_numpy() == frame["label"].to_numpy()))


def group_gain(
    documents: pd.DataFrame,
    feature: str,
    *,
    label_column: str = "label",
) -> float:
    """Gain of the best "majority label per group" rule over the global majority class.

    Args:
        documents: Labelled frame holding ``feature``.
        feature: Categorical feature to test.
        label_column: Column holding the label.

    Returns:
        The gain, in ``[0, 1]``. A feature drawn independently of the label scores near 0: the
        groups are too small to beat the global majority by much.
    """
    baseline = _majority_accuracy(documents[label_column].astype(str).to_numpy())
    return max(group_accuracy(documents, feature, label_column=label_column) - baseline, 0.0)


def shortcut_scores(
    documents: pd.DataFrame,
    *,
    label_column: str = "label",
    text_column: str = "text",
) -> dict[str, float]:
    """Measure what one column alone explains, and compare it with the majority baseline.

    Args:
        documents: Labelled frame (surface features are computed if absent).
        label_column: Column holding the label.
        text_column: Column holding the text.

    Returns:
        Mapping of candidate column to the **gain** of the best one-line rule built on it (see
        :func:`stump_gain` and :func:`group_gain`), plus ``majority_baseline`` (the best constant
        rule, for reference) and ``shortcut_margin`` (the strongest gain).

    Raises:
        ValueError: When the label column is missing.
    """
    if label_column not in documents.columns:
        msg = f"Column '{label_column}' missing from the frame: {sorted(documents.columns)}"
        raise ValueError(msg)
    frame = documents
    if not all(name in frame.columns for name in TEXT_FEATURES):
        frame = build_text_features(documents, text_column=text_column)
    baseline = _majority_accuracy(frame[label_column].astype(str).to_numpy())
    scores = {name: stump_gain(frame, name, label_column=label_column) for name in TEXT_FEATURES}
    for name in CATEGORICAL_CANDIDATES:
        if name in frame.columns:
            scores[name] = group_gain(frame, name, label_column=label_column)
    margin = float(max(scores.values(), default=0.0))
    scores["majority_baseline"] = baseline
    scores["shortcut_margin"] = margin
    return scores


def _describe_text(text: str) -> dict[str, float]:
    """Return the surface features of one text.

    Args:
        text: Normalised text.

    Returns:
        One value per entry of :data:`TEXT_FEATURES`.
    """
    tokens = tokenize(text)
    characters = [character for character in text if not character.isspace()]
    words = [token for token in tokens if any(character.isalpha() for character in token)]
    punctuation = sum(1 for character in characters if not character.isalnum() and character != "'")
    lowered = text.lower()
    return {
        "n_chars": float(len(text)),
        "n_words": float(len(tokens)),
        "n_sentences": float(len(split_sentences(text)) or 1),
        "mean_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "digit_ratio": float(
            sum(character.isdigit() for character in characters) / max(len(characters), 1)
        ),
        "uppercase_ratio": float(
            sum(character.isupper() for character in text) / max(len(text), 1)
        ),
        "punctuation_ratio": float(punctuation / max(len(characters), 1)),
        "has_greeting": float(any(lowered.startswith(greeting) for greeting in GREETINGS)),
        "has_signature": float(any(marker in lowered for marker in SIGNATURES)),
        "has_reference": float("cmd-" in lowered),
    }


def _majority_accuracy(labels: np.ndarray) -> float:
    """Accuracy of the constant prediction of the most frequent label."""
    if labels.size == 0:
        return 0.0
    counts = np.unique(labels, return_counts=True)[1]
    return float(counts.max()) / float(labels.size)


def describe_features(names: Sequence[str] | None = None) -> Mapping[str, str]:
    """Return a short French description of every feature (used by the report).

    Args:
        names: Feature names (defaults to :data:`TEXT_FEATURES`).

    Returns:
        Mapping of feature name to description.
    """
    descriptions = {
        "n_chars": "Longueur du texte en caractères.",
        "n_words": "Nombre de mots (jetons alphanumériques).",
        "n_sentences": "Nombre de phrases estimé par la ponctuation.",
        "mean_word_length": "Longueur moyenne d'un mot : un texte bâclé écrit court.",
        "digit_ratio": "Part de chiffres : références de commande et montants.",
        "uppercase_ratio": "Part de majuscules : un client qui écrit en criant.",
        "punctuation_ratio": "Densité de ponctuation : un texte haché ou soigné.",
        "has_greeting": "Le ticket commence par une formule de politesse (bruit partagé).",
        "has_signature": "Le ticket finit par une signature (bruit partagé).",
        "has_reference": "Le ticket cite une commande `CMD-....`.",
        "source": "Canal d'arrivée du ticket (distracteur déclaré).",
        "style": "Style rédactionnel du ticket, canonique / paraphrase / bruité.",
        "priority": "Priorité déclarée par le client (distracteur déclaré du corpus).",
        "majority_baseline": "Exactitude d'un modèle qui prédit toujours la classe majoritaire.",
        "shortcut_margin": "Gain du meilleur trait seul : 0 rien, 0,5 sépare deux classes.",
        "stump_gain": "Gain d'une règle à un seuil sur deux classes, sans apprendre de poids.",
        "group_gain": "Gain d'une règle « classe majoritaire du groupe », sans apprendre de poids.",
    }
    selected = list(names or TEXT_FEATURES)
    return {name: descriptions.get(name, "") for name in selected}


__all__ = [
    "CATEGORICAL_CANDIDATES",
    "GREETINGS",
    "SIGNATURES",
    "TEXT_FEATURES",
    "build_text_features",
    "describe_features",
    "feature_summary",
    "group_accuracy",
    "group_gain",
    "shortcut_scores",
    "stump_gain",
]
