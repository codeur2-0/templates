"""Traits de forme : ce que l'écriture d'une mention dit déjà, et ce qu'elle ne dit pas.

Ce module teste la thèse de la famille, chiffrée :

* une référence de commande, un montant et une date sont des **motifs** — leur signature de forme
est
  stable, donc une règle qui ne lit que la forme les retrouve ;
* un produit et un transporteur sont des **noms** — leur forme ne dit rien, donc la même règle
  s'effondre. C'est exactement la raison pour laquelle le corpus réserve des surfaces aux splits
  d'évaluation, et pour laquelle la F1 macro est publiée à côté de la F1 micro.

Le test vérifie aussi deux propriétés de construction : les traits sont **calculés** à partir des
surfaces (pas copiés d'une colonne), et l'agrégation par type conserve le nombre de mentions.
"""

from __future__ import annotations

import pytest

from src.data.schemas import ENTITY_LABELS
from src.features.build_features import (
    MESSAGE_FEATURES,
    SPAN_FEATURES,
    build_message_features,
    build_span_features,
    describe_features,
    feature_summary,
    shape_separability,
    shape_shortcut_scores,
)

#: Types dont l'écriture est régulière : une règle de forme doit les retrouver.
FORMAL_LABELS = ("commande", "date", "montant")
#: Types dont la reconnaissance repose sur un nom appris.
NAME_LABELS = ("produit", "transporteur")


def test_span_features_are_added_and_derived_from_the_surface(documents, spans) -> None:
    """Chaque trait est calculé à partir de la surface, sur les mêmes lignes."""
    features = build_span_features(documents, spans)

    assert len(features) == len(spans)
    assert set(SPAN_FEATURES) <= set(features.columns)
    assert set(features["split"]) <= set(documents["split"])
    commande = features[features["label"] == "commande"].iloc[0]
    assert commande["has_digit"] == 1
    assert "-" in str(commande["signature"]) or "#" in str(commande["signature"])
    date = features[features["has_month_name"] == 1].iloc[0]
    assert date["label"] == "date"


def test_the_signature_ignores_the_content_and_keeps_the_shape(documents, spans) -> None:
    """Deux surfaces de même forme ont la même signature, même si le contenu diffère."""
    features = build_span_features(documents, spans)
    commandes = features[features["label"] == "commande"]

    assert commandes["signature"].nunique() <= 3, "une référence de commande a une forme stable"
    assert commandes["signature"].nunique() < commandes["surface"].nunique()


def test_message_features_count_the_annotations_of_each_message(documents, spans) -> None:
    """Les traits de message comptent les mentions réelles, pas une colonne déclarée."""
    features = build_message_features(documents, spans)
    counted = spans.groupby("msg_id").size().to_dict()

    assert set(MESSAGE_FEATURES) <= set(features.columns)
    for row in features.sample(min(len(features), 30), random_state=0).itertuples(index=False):
        assert int(row.n_mentions) == int(counted.get(str(row.msg_id), 0))
        assert int(row.n_chars) == len(str(row.text))
        assert 0.0 <= float(row.annotated_char_ratio) <= 1.0


def test_feature_summary_aggregates_by_type(documents, spans) -> None:
    """L'agrégation par type conserve le nombre de mentions et couvre la nomenclature."""
    features = build_span_features(documents, spans)
    summary = feature_summary(features)

    assert set(summary["label"]) == set(ENTITY_LABELS)
    assert int(summary["n_mentions"].sum()) == len(spans)
    assert summary["n_chars"].gt(0).all()


def test_a_shape_only_rule_finds_formal_types_and_misses_names(documents, spans) -> None:
    """La règle de forme retrouve les types réguliers et plafonne sur les noms."""
    scores = shape_shortcut_scores(documents, spans, split="val")
    by_label = {str(row.label): row for row in scores.itertuples(index=False)}

    for label in FORMAL_LABELS:
        assert by_label[label].recall >= 0.8, f"{label} doit être reconnaissable par sa forme"
        assert by_label[label].f1 >= 0.8, f"{label} doit être retrouvable par une règle de forme"
        assert by_label[label].n_unseen == 0, f"{label} est un motif : aucune forme inédite"
    for label in NAME_LABELS:
        # Un nom n'a pas de forme stable : la signature que la règle apprend pour lui attrape aussi
        # des mentions d'autres types, donc sa précision ne peut pas dépasser celle d'un motif.
        assert by_label[label].f1 < by_label["commande"].f1
        assert by_label[label].precision <= by_label["commande"].precision
        # Être à la fois complet et exact sur un nom demanderait que sa forme suffise — la thèse
        # inverse de la famille. Si cette assertion tombe un jour, c'est la mesure qu'il faut
        # relire.
        assert min(by_label[label].precision, by_label[label].recall) < 1.0
    assert by_label["produit"].share_by_shape > 0.0
    # ``n_unseen`` compte les **signatures** inédites : une surface réservée peut garder la
    # signature d'un nom vu à l'entraînement, et c'est au niveau de la surface — ``holdout_recall``
    # — que l'écart se mesure. La table publie les deux lectures.
    assert (scores["n_unseen"] >= 0).all()
    assert (scores["n_signatures"] >= 1).all()
    assert (scores["n_spans"] > 0).all()
    assert len(scores) >= len(ENTITY_LABELS)


def test_the_shape_rule_refuses_to_be_measured_on_its_own_fit_split(documents, spans) -> None:
    """Notée sur le split qui l'a apprise, la règle mesurerait sa mémoire."""
    with pytest.raises(ValueError, match="measures its memory"):
        shape_shortcut_scores(documents, spans, split="train", fit_split="train")


def test_the_shape_rule_is_measured_on_a_split_it_never_read(documents, spans) -> None:
    """Apprise sur le train, la même règle se mesure sur n'importe quel split annoté."""
    scores = shape_shortcut_scores(documents, spans, split="test")

    assert set(scores["label"]) == set(ENTITY_LABELS)
    assert (scores["n_spans"] > 0).all()
    assert scores["f1"].between(0.0, 1.0).all()
    assert scores["precision"].between(0.0, 1.0).all()


def test_shape_separability_names_the_ambiguous_signatures(documents, spans) -> None:
    """Les signatures partagées par plusieurs types sont nommées."""
    separability = shape_separability(documents, spans)

    assert not separability.empty
    assert separability["purity"].between(0.0, 1.0).all()
    ambiguous = separability[separability["n_labels"] > 1]
    assert all(str(row.signature) for row in ambiguous.itertuples(index=False))
    assert int(separability["n_spans"].sum()) == len(spans)


def test_the_features_are_documented(documents, spans) -> None:
    """Chaque trait porte sa définition : un notebook ne peut pas se tromper de nom."""
    documented = describe_features()

    assert set(SPAN_FEATURES) <= set(documented)
    assert set(MESSAGE_FEATURES) <= set(documented)
    assert all(documented[name] for name in SPAN_FEATURES)
    assert set(describe_features(["n_chars", "signature"])) == {"n_chars", "signature"}
    assert describe_features(["inconnu"])["inconnu"]
