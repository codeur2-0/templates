"""Générateur du corpus annoté : déterminisme, réserves et distracteurs.

Trois propriétés du corpus sont testées ici, parce que **tout le reste du projet en dépend** :

* le **déterminisme** — même graine, mêmes textes, mêmes annotations, mêmes découpages. Sans lui,
  aucune comparaison entre deux exécutions n'a de sens ;
* les **réserves** — aucune surface annotée d'un split d'évaluation n'apparaît à l'entraînement. Si
  une seule passe, la mesure de généralisation ne mesure plus rien, et le test doit le dire ;
* les **distracteurs** — les villes, familles de produits, numéros de facture et années seules sont
  présents dans les textes et **jamais** annotés : sans eux, un système apprendrait à surligner tous
  les noms propres et la précision serait trompeuse.

Le test vérifie aussi que les annotations sont **écrites** (jamais retrouvées après coup) : la
surface
enregistrée est la copie exacte de ``text[start:end]``, ce qui est la condition d'un corpus
apprenable.
"""

from __future__ import annotations

import pandas as pd

from src.data.generators import SyntheticEntityCorpusGenerator
from src.data.schemas import ENTITY_LABELS, SPLITS, STYLES

#: Colonnes de la table d'annotations, dans l'ordre attendu.
ANNOTATION_COLUMNS = ("msg_id", "start", "end", "label", "surface", "holdout")


def test_the_same_seed_produces_the_same_corpus(app_config) -> None:
    """Deux générations à configuration identique rendent des tables identiques."""
    first = SyntheticEntityCorpusGenerator.from_config(app_config.data).generate()
    second = SyntheticEntityCorpusGenerator.from_config(app_config.data).generate()

    pd.testing.assert_frame_equal(first.documents, second.documents)
    pd.testing.assert_frame_equal(first.queries, second.queries)
    assert first.metadata["holdout"] == second.metadata["holdout"]


def test_a_different_seed_produces_a_different_corpus(app_config) -> None:
    """Changer la graine change les textes : le corpus n'est pas figé par accident."""
    other = app_config.data.model_copy(update={"seed": app_config.data.seed + 1})

    reference = SyntheticEntityCorpusGenerator.from_config(app_config.data).generate()
    shifted = SyntheticEntityCorpusGenerator.from_config(other).generate()

    assert not reference.documents["text"].equals(shifted.documents["text"])


def test_the_corpus_declares_both_tables_with_their_columns(documents, spans) -> None:
    """Les deux tables portent les colonnes déclarées, et les annotations la nomenclature."""
    assert set(ANNOTATION_COLUMNS) <= set(spans.columns)
    assert set(documents.columns) >= {
        "msg_id",
        "text",
        "canal",
        "style",
        "n_entities",
        "n_tokens",
        "received_at",
        "split",
    }
    assert set(spans["label"].unique()) <= set(ENTITY_LABELS)
    assert set(documents["style"].unique()) <= set(STYLES)
    assert set(documents["split"].unique()) == set(SPLITS)


def test_every_message_carries_between_three_and_five_annotated_entities(documents, spans) -> None:
    """Le corpus n'a pas de message sans entité : le compteur de la table le confirme."""
    counted = spans.groupby("msg_id").size()
    assert counted.min() >= 3
    assert counted.max() <= 5
    declared = dict(zip(documents["msg_id"], documents["n_entities"], strict=True))
    assert all(int(declared[identifier]) == int(count) for identifier, count in counted.items())


def test_the_annotations_are_written_and_never_inferred(documents, spans) -> None:
    """``surface == text[start:end]`` pour toutes les annotations, pas pour un échantillon."""
    texts = dict(zip(documents["msg_id"], documents["text"], strict=True))
    mismatches = [
        (row.msg_id, row.label)
        for row in spans.itertuples(index=False)
        if texts[str(row.msg_id)][int(row.start) : int(row.end)] != str(row.surface)
    ]

    assert mismatches == []


def test_mentions_never_overlap_inside_a_message(spans) -> None:
    """Deux mentions d'un même message ne se recouvrent jamais : la sortie est une liste."""
    overlaps = 0
    for _, group in spans.sort_values(["msg_id", "start", "end"]).groupby("msg_id"):
        previous_end = -1
        for row in group.itertuples(index=False):
            if int(row.start) < previous_end:
                overlaps += 1
            previous_end = max(previous_end, int(row.end))
    assert overlaps == 0


def test_reserved_surfaces_never_leak_into_the_training_split(documents, spans) -> None:
    """Aucune surface réservée n'est annotée dans le train, pour le même type."""
    split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
    annotated = spans.assign(split=spans["msg_id"].map(split_of))
    train = annotated[annotated["split"] == "train"]
    reserved = annotated[annotated["split"] != "train"]
    reserved = reserved[reserved["holdout"].astype(bool)]

    assert not reserved.empty
    train_surfaces = {
        (str(row.label), str(row.surface).casefold()) for row in train.itertuples(index=False)
    }
    leaking = [
        (str(row.label), str(row.surface))
        for row in reserved.itertuples(index=False)
        if (str(row.label), str(row.surface).casefold()) in train_surfaces
    ]
    assert leaking == []


def test_the_metadata_publishes_the_references_a_score_needs(corpus) -> None:
    """Les métadonnées publient la recette, les réserves et le plancher trivial."""
    metadata = corpus.metadata

    assert metadata["seed"] == 13
    assert metadata["n_documents"] == len(corpus.documents)
    assert metadata["n_entities"] == len(corpus.queries)
    assert metadata["holdout"]["share"] > 0.0
    assert set(metadata["holdout"]["per_label"]) == set(ENTITY_LABELS)
    assert metadata["distractors"]["literal_slots"]
    assert metadata["distractors"]["cities"]
    memorisation = metadata["surface_memorisation"]
    assert set(memorisation["test"]) == set(ENTITY_LABELS)
    # Les noms se répètent d'un split à l'autre, les motifs ne se répètent pas : un montant ou une
    # date inédits restent reconnaissables par leur forme, un nom inédit non.
    assert memorisation["test"]["transporteur"] > memorisation["test"]["montant"]
    assert metadata["trivial_floor"]["entity_f1"] == 0.0
    assert set(metadata["entity_counts"]) == set(ENTITY_LABELS)


def test_distractors_are_present_in_the_texts_and_never_annotated(documents, spans) -> None:
    """Les villes et les numéros de facture apparaissent dans les textes, sans être annotés."""
    annotated = [
        (str(row.msg_id), int(row.start), int(row.end)) for row in spans.itertuples(index=False)
    ]
    covered = {(message_id, start, end) for message_id, start, end in annotated}
    cities = ("Paris", "Lyon", "Marseille", "Lille", "Bordeaux")
    occurrences = 0
    for row in documents.itertuples(index=False):
        text = str(row.text)
        for city in cities:
            position = text.find(city)
            while position >= 0:
                occurrences += 1
                assert (str(row.msg_id), position, position + len(city)) not in covered
                position = text.find(city, position + 1)

    assert occurrences > 0, "le corpus doit contenir des distracteurs déclarés"
