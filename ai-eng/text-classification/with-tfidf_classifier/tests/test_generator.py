"""Générateur de tickets : reproductibilité, parts annoncées, et propriétés déclarées.

Les tests de ce module ne vérifient pas des nombres « plausibles » : ils vérifient les propriétés
que le README annonce, parce que ce sont celles dont le reste du projet dépend — un corpus
déterministe, des styles aux parts annoncées, des paraphrases qui n'utilisent pas le vocabulaire de
leur classe, et une référence à mots-clés réellement meilleure que la classe majoritaire.

Le test le plus important est celui des **formulations retenues** : les splits d'évaluation ne
doivent contenir que des tournures que le split d'entraînement n'a jamais vues. Sans lui, le projet
pourrait afficher une F1 parfaite en recopiant ses propres gabarits, et personne ne le verrait.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.generators import (
    CLASS_WEIGHTS,
    PHRASING_HOLDOUT,
    PRODUCTS,
    SPLIT_SHARES,
    STYLE_SHARES,
    TEMPLATES,
    SyntheticTicketGenerator,
    class_keywords,
    keyword_baseline_accuracy,
)
from src.preprocessing.transformers import normalise_text

#: Référence de commande telle qu'elle apparaît dans les textes (en minuscules dans le bruité).
_ORDER = re.compile(r"cmd-\d{5}", re.IGNORECASE)


def _wording(text: str) -> str:
    """Return the *wording* of a ticket: its drawn values replaced by placeholders.

    Deux tickets partagent une formulation quand, leurs valeurs tirées masquées, ils s'écrivent
    pareil. C'est la bonne granularité pour parler de mémorisation : recopier un gabarit n'est pas
    recopier une référence de commande.

    Args:
        text: Ticket text.

    Returns:
        The text, normalised, with the product and the order reference replaced by placeholders.
    """
    lowered = normalise_text(str(text)).lower()
    for product in PRODUCTS:
        lowered = lowered.replace(product.lower(), "{product}")
    return _ORDER.sub("{order}", lowered)


def test_two_runs_at_the_same_seed_produce_the_same_corpus() -> None:
    """Le générateur est déterministe : le README publie des chiffres reproductibles."""
    first = SyntheticTicketGenerator(n_documents=120, seed=3).generate().documents
    second = SyntheticTicketGenerator(n_documents=120, seed=3).generate().documents

    pd.testing.assert_frame_equal(first, second)


def test_two_seeds_produce_different_corpora() -> None:
    """Changer de graine change le corpus : le déterminisme ne cache pas un générateur figé."""
    first = SyntheticTicketGenerator(n_documents=120, seed=3).generate().documents
    second = SyntheticTicketGenerator(n_documents=120, seed=4).generate().documents

    assert not first["text"].equals(second["text"])


def test_every_class_is_present(documents: pd.DataFrame) -> None:
    """Les six classes du service sont représentées, y compris la minoritaire."""
    counts = documents["label"].value_counts()

    assert set(counts.index) == set(CLASS_WEIGHTS)
    assert int(counts.min()) >= 10


def test_the_class_shares_follow_the_declared_weights(documents: pd.DataFrame) -> None:
    """L'écart aux parts annoncées reste sous le point de pourcentage attendu d'un tirage."""
    observed = documents["label"].value_counts(normalize=True)

    for label, weight in CLASS_WEIGHTS.items():
        assert abs(float(observed[label]) - weight) < 0.05, label


def test_the_editorial_styles_follow_their_shares(documents: pd.DataFrame) -> None:
    """Un tiers de paraphrases et un cinquième de tickets bruités : le contrat du corpus."""
    observed = documents["style"].value_counts(normalize=True)

    for style, share in STYLE_SHARES.items():
        assert abs(float(observed[style]) - share) < 0.05, style


def test_a_paraphrase_avoids_the_vocabulary_of_its_class(documents: pd.DataFrame) -> None:
    """C'est la propriété qui donne son sens au segment ``paraphrase``.

    Si une paraphrase réutilisait les mots de sa classe, le segment « difficile » ne mesurerait
    rien du tout : il ne serait qu'un second exemplier du segment facile.
    """
    keywords = class_keywords()
    paraphrases = documents[documents["style"] == "paraphrase"]

    for label, terms in keywords.items():
        if not terms:
            continue
        texts = paraphrases.loc[paraphrases["label"] == label, "text"]
        assert not texts.empty, label
        for text in texts:
            haystack = normalise_text(str(text)).lower()
            assert not any(term in haystack for term in terms), (label, text)


def test_a_canonical_ticket_uses_the_vocabulary_of_its_class(documents: pd.DataFrame) -> None:
    """Le segment facile doit l'être *par construction*, pas par chance."""
    keywords = class_keywords()
    canonical = documents[documents["style"] == "canonique"]

    for label, terms in keywords.items():
        if not terms:
            continue
        texts = canonical.loc[canonical["label"] == label, "text"]
        hits = [any(term in normalise_text(str(text)).lower() for term in terms) for text in texts]
        assert all(hits), label


def test_the_evaluation_splits_use_phrasings_the_training_split_never_saw(
    documents: pd.DataFrame,
) -> None:
    """Les formulations du test sont inédites : le score mesure une généralisation, pas une copie.

    Chaque pool (classe, style) réserve ses dernières formulations aux splits ``val``,
    ``calibration`` et ``test``. Si cette réserve disparaissait, le classifieur retrouverait des
    gabarits mémorisés et la F1 publiée flatterait le projet sans rien prouver.
    """
    for key, group in documents.groupby(["label", "style"], observed=True):
        learned = {_wording(text) for text in group.loc[group["split"] == "train", "text"]}
        evaluated = {_wording(text) for text in group.loc[group["split"] != "train", "text"]}

        assert learned, key
        assert evaluated, key
        assert not learned & evaluated, (key, sorted(learned & evaluated))


def test_the_splits_are_stratified_and_complete(documents: pd.DataFrame) -> None:
    """Chaque classe apparaît dans chaque split qui compte, et les parts sont celles annoncées.

    Le split ``calibration`` ne fait que 5 % : on ne peut pas exiger de chaque couple
    (classe, style) qu'il y soit représenté. ``train``, ``val`` et ``test``, eux, doivent
    contenir chaque classe — sinon la stratification n'est qu'une intention.
    """
    sizes = documents["split"].value_counts(normalize=True)

    assert set(documents["split"]) == set(SPLIT_SHARES)
    for split, share in SPLIT_SHARES.items():
        assert abs(float(sizes[split]) - share) < 0.05, split
    for label, group in documents.groupby("label", observed=True):
        for split in ("train", "val", "test"):
            assert (group["split"] == split).any(), (label, split)
    for (label, style), group in documents.groupby(["label", "style"], observed=True):
        assert (group["split"] == "train").any(), (label, style)


def test_the_identifiers_are_unique_and_stable(documents: pd.DataFrame) -> None:
    """Les identifiants suivent le motif et ne se répètent jamais."""
    assert documents["doc_id"].is_unique
    assert documents["doc_id"].str.match(r"^TKT-\d{4}$").all()


def test_the_metadata_carries_the_recipe_and_the_references(documents: pd.DataFrame) -> None:
    """Les métadonnées disent comment le corpus a été fait, et quel niveau battre."""
    generator = SyntheticTicketGenerator(n_documents=180, seed=5)
    corpus = generator.generate()
    metadata = corpus.metadata
    baseline = float(documents["label"].value_counts(normalize=True).max())

    assert corpus.queries.empty, "un corpus de classification est une seule table"
    assert metadata["n_documents"] == len(corpus.documents)
    assert metadata["n_classes"] == len(CLASS_WEIGHTS)
    assert metadata["seed"] == 5
    assert metadata["phrasing_holdout"] == PHRASING_HOLDOUT
    assert sum(metadata["split_sizes"].values()) == metadata["n_documents"]
    # La règle à mots-clés est la référence honnête : meilleure que la classe majoritaire, et
    # très loin d'être parfaite — sinon elle rendrait le projet inutile.
    rule = float(metadata["rule_baseline_accuracy"])
    assert baseline <= rule < 0.95
    assert rule == pytest.approx(keyword_baseline_accuracy(corpus.documents))


def test_the_reference_date_bounds_the_publication_dates(documents: pd.DataFrame) -> None:
    """Les dates de réception sont antérieures à la date de référence et cohérentes."""
    reference = date(2025, 3, 3)
    published = pd.to_datetime(documents["published_at"]).dt.date

    assert published.max() <= reference
    assert (published >= reference.replace(year=reference.year - 1)).all()


def test_the_generator_refuses_an_impossible_configuration() -> None:
    """Un corpus trop petit, des parts fausses ou une réserve trop large sont refusés au montage."""
    with pytest.raises(ValueError, match="shares must sum to 1"):
        SyntheticTicketGenerator(n_documents=60, style_shares={"canonique": 0.5, "bruite": 0.2})
    with pytest.raises(ValueError, match="n_documents must be at least"):
        SyntheticTicketGenerator(n_documents=12)
    # Réserver plus de formulations que les pools n'en contiennent laisserait le split
    # d'entraînement sans rien à apprendre : le générateur refuse plutôt que de produire un corpus
    # où le modèle ne pourrait pas gagner.
    with pytest.raises(ValueError, match="phrasing_holdout must leave"):
        SyntheticTicketGenerator(n_documents=180, phrasing_holdout=5)


def test_the_generator_reads_its_node_of_the_configuration() -> None:
    """``from_config`` lit le nœud ``data`` (mapping ou objet pydantic) sans coder de valeur."""
    generator = SyntheticTicketGenerator.from_config(
        {"n_samples": 96, "seed": 11, "corpus": {"reference_date": "2024-12-01"}}
    )
    corpus = generator.generate()

    assert len(corpus.documents) == 96
    assert generator.seed == 11
    assert generator.reference_date == date(2024, 12, 1)
    assert generator.dataset_name == "support_tickets"

    # La réserve de formulations est un paramètre du corpus, pas une constante enfouie.
    assert (
        SyntheticTicketGenerator.from_config({"corpus": {"phrasing_holdout": 2}}).phrasing_holdout
        == 2
    )


def test_the_corpus_holds_exactly_the_declared_columns(documents: pd.DataFrame) -> None:
    """Les colonnes sont celles du contrat Pandera, dans un ordre stable pour la relecture."""
    assert list(documents.columns) == [
        "doc_id",
        "text",
        "label",
        "source",
        "style",
        "priority",
        "published_at",
        "n_tokens",
        "split",
    ]
    assert documents["text"].str.len().between(40, 1200).all()
    assert np.issubdtype(documents["n_tokens"].dtype, np.integer)
    assert len(TEMPLATES) == len(CLASS_WEIGHTS)
