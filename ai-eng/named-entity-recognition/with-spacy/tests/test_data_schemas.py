"""Contrats de données : deux tables, et trois liens qu'aucun schéma de table ne peut vérifier.

Cinq propriétés sont testées ici, et chacune a coûté un bug ailleurs :

* **les bornes** — ``surface == text[start:end]``. Un corpus décalé d'un caractère produit un modèle
  qui apprend à côté sans que rien ne casse : c'est le contrôle le plus important du projet ;
* **la jointure** — toute annotation référence un message existant ;
* **le non-chevauchement** — deux mentions d'un même message ne se recouvrent jamais : le
  back-office remplit un dossier avec une liste, pas avec un graphe d'ambiguïtés ;
* **les types déclarés** — un type hors nomenclature est refusé à l'écriture, pas au rapport ;
* **le contrat des prédictions** — une confiance hors de [0 ; 1] ou un décalage négatif échoue, et
  l'erreur est levée **avant** l'écriture du fichier consommé par le notebook.

Le corpus utilisé est celui du générateur de la famille : valider un contrat sur des données
inventées par le test validerait le test, pas le générateur.
"""

from __future__ import annotations

import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.schemas import (
    ENTITY_LABELS,
    MESSAGE_ID_PATTERN,
    corpus_violations,
    describe_labels,
    holdout_share,
    label_counts,
    validate_corpus,
    validate_messages,
    validate_predictions,
    validate_spans,
)


def test_the_generated_corpus_satisfies_both_contracts(documents, spans) -> None:
    """Les deux tables du générateur passent leurs contrats, et leurs liens tiennent."""
    validated_documents, validated_spans = validate_corpus(documents, spans)

    assert len(validated_documents) == len(documents)
    assert len(validated_spans) == len(spans)
    assert corpus_violations(documents, spans) == []
    # Le contrat est strict : il ne valide pas seulement, il refuse tout ce qui n'est pas déclaré.
    assert set(validated_documents.columns) == set(documents.columns)


def test_the_surface_is_exactly_the_text_slice(documents, spans) -> None:
    """``surface`` est exactement ``text[start:end]`` : ce contrôle rend le corpus apprenable."""
    for row in spans.head(50).itertuples(index=False):
        text = str(documents.loc[documents["msg_id"] == row.msg_id, "text"].iloc[0])
        assert str(row.surface) == text[int(row.start) : int(row.end)]
    assert int((spans["end"] > spans["start"]).sum()) == len(spans)


def test_a_shifted_surface_is_rejected(documents, spans) -> None:
    """Une surface décalée d'un caractère est refusée, avec la raison dans le message."""
    broken = spans.copy()
    broken.loc[broken.index[0], "start"] = int(broken.loc[broken.index[0], "start"]) + 1

    violations = corpus_violations(documents, broken)

    assert violations, "un décalage de borne doit être signalé"
    assert any("n'est pas text[" in item for item in violations)


def test_overlapping_annotations_are_rejected(documents, spans) -> None:
    """Deux mentions qui se recouvrent dans un même message sont refusées."""
    first = spans.iloc[0]
    overlapping = spans.copy()
    overlapping.loc[len(overlapping)] = {
        **first.to_dict(),
        "start": int(first["start"]) + 1,
        "end": int(first["end"]) + 1,
        "surface": "x" * (int(first["end"]) - int(first["start"])),
    }

    violations = corpus_violations(documents, overlapping)

    assert any("chevauche" in item.lower() for item in violations)


def test_an_unknown_entity_type_is_rejected(documents, spans) -> None:
    """Un type hors nomenclature casse le contrat de la table d'annotations."""
    broken = spans.copy()
    broken.loc[broken.index[0], "label"] = "telephone"

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_spans(broken)


def test_a_message_identifier_out_of_pattern_is_rejected(documents) -> None:
    """Le motif des identifiants est vérifié : ``MSG-123`` (trois chiffres) est refusé."""
    broken = documents.copy()
    broken.loc[broken.index[0], "msg_id"] = "MSG-123"

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_messages(broken)
    assert MESSAGE_ID_PATTERN.startswith("^MSG-")


def test_predictions_contract_rejects_an_impossible_confidence(predictions) -> None:
    """Une confiance hors de [0 ; 1] échoue avant l'écriture, pas dans le notebook."""
    broken = predictions.copy()
    broken.loc[broken.index[0], "confidence"] = 1.4

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_predictions(broken)

    validated = validate_predictions(predictions)
    assert validated["confidence"].between(0.0, 1.0).all()
    assert validated["source"].isin(["regle", "modele"]).all()
    assert (validated["end"] > validated["start"]).all()


def test_label_counts_and_holdout_share_describe_the_corpus(spans) -> None:
    """Les compteurs par type somment au total, et la part réservée est un taux."""
    counts = label_counts(spans)
    shares = holdout_share(spans)

    assert sum(counts.values()) == len(spans)
    assert set(counts) == set(ENTITY_LABELS)
    assert all(0.0 <= share <= 1.0 for share in shares.values())
    assert set(describe_labels()) == set(ENTITY_LABELS)
    assert all(describe_labels()[label] for label in ENTITY_LABELS)


def test_reserved_surfaces_never_appear_in_the_training_split(documents, spans) -> None:
    """Les surfaces réservées sont absentes du train : c'est le principe même du corpus."""
    split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
    annotated = spans.assign(split=spans["msg_id"].map(split_of))
    train = annotated[annotated["split"] == "train"]
    evaluation = annotated[annotated["split"] != "train"]

    reserved = evaluation[evaluation["holdout"].astype(bool)]
    assert not reserved.empty, "le corpus doit contenir des surfaces réservées"
    train_surfaces = {
        (str(row.label), str(row.surface).casefold()) for row in train.itertuples(index=False)
    }
    for row in reserved.itertuples(index=False):
        assert (str(row.label), str(row.surface).casefold()) not in train_surfaces


def test_the_two_tables_are_joined_by_the_message_identifier(documents, spans) -> None:
    """Toute annotation pointe un message réel : une clé orpheline est signalée."""
    orphan = spans.copy()
    orphan.loc[orphan.index[0], "msg_id"] = "MSG-9999"
    orphan.loc[orphan.index[0], "surface"] = "inconnu"

    violations = corpus_violations(documents, orphan)

    assert any("message inconnu" in item for item in violations)
