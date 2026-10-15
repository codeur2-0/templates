"""Métriques, boucle d'entraînement et déterminisme.

Les métriques d'extraction d'entités sont testées sur des cas **construits à la main**, pas sur le
corpus : un cas écrit à la main dit exactement ce que la métrique doit rendre, alors qu'un score lu
sur un corpus ne dit que « c'est plausible ». Six propriétés y sont vérifiées :

* un système qui n'annote rien obtient 0,0 (le plancher est le dénominateur de toute lecture) ;
* un span décalé d'un caractère est **faux** en F1 stricte et **correct** en F1 partielle : c'est la
  mesure du coût de l'exigence de bornes ;
* la F1 macro protège un type minoritaire que la F1 micro noierait ;
* les erreurs sont classées (inventée, bornes, manquée) — un diagnostic, pas un score ;
* la confiance est jugée sur pièces : précision observée par niveau ;
* un modèle parfait obtient 1,0 — sans ce cas, un bug qui sous-compterait les vrais positifs
  passerait pour de la rigueur.

La boucle est ensuite testée sur le petit corpus : elle doit produire un historique, mesurer la
validation au niveau entité, refuser de réentraîner pour rien, et **donner les mêmes prédictions à
graine fixée**.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import build_model
from src.models.contract import EntityMention
from src.training.metrics import (
    Span,
    confidence_gap,
    confidence_table,
    describe_metrics,
    entity_scores,
    error_frame,
    latency_stats,
    partial_scores,
    per_label_frame,
    percentile,
    precision_recall_f1,
    span_sets,
    trivial_floor,
)
from src.training.trainer import EntityTrainer, score_arrays

#: Deux messages, quatre entités de trois types : le corpus de test des métriques.
GOLD: list[set[Span]] = [
    {(0, 6, "commande"), (10, 15, "produit")},
    {(4, 9, "montant"), (12, 20, "produit")},
]
#: Le même corpus, prédit avec une erreur de bornes (produit décalé) et une mention manquée.
PREDICTED: list[set[Span]] = [
    {(0, 6, "commande"), (10, 16, "produit")},
    {(4, 9, "montant")},
]


def test_a_silent_system_gets_a_zero_f1() -> None:
    """Un système qui n'annote rien n'a ni précision ni rappel : le plancher est 0,0."""
    scores = entity_scores(GOLD, [set(), set()])

    assert scores["entity_f1"] == 0.0
    assert scores["entity_precision"] == 0.0
    assert scores["entity_recall"] == 0.0
    assert trivial_floor()["entity_f1"] == 0.0
    assert precision_recall_f1(0, 0, 0) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }


def test_a_perfect_prediction_scores_one() -> None:
    """Le cas parfait vaut 1,0 sur toutes les métriques : sans lui, un sous-comptage passerait."""
    scores = entity_scores(GOLD, GOLD)

    assert scores["entity_f1"] == 1.0
    assert scores["entity_precision"] == 1.0
    assert scores["entity_recall"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["type_accuracy"] == 1.0
    assert scores["boundary_accuracy"] == 1.0


def test_strict_scoring_counts_a_shifted_boundary_as_false() -> None:
    """Un span décalé d'un caractère est faux en strict, et vrai en partiel : l'écart est publié."""
    strict = entity_scores(GOLD, PREDICTED)
    partial = partial_scores(GOLD, PREDICTED)

    # Deux spans exacts sur quatre. Le span décalé compte **deux fois** : faux positif (les bornes
    # sont fausses) et faux négatif (l'entité de référence n'est pas retrouvée) — c'est le coût
    # d'une exigence de bornes, et il se lit dans les trois scores à la fois.
    assert strict["entity_precision"] == 0.6667
    assert strict["entity_recall"] == 0.5
    assert strict["entity_f1"] == 0.5714
    assert strict["boundary_accuracy"] < 1.0
    assert partial["partial_f1"] > strict["entity_f1"]
    assert "boundary_errors" in partial


def test_the_per_label_table_protects_a_minority_type() -> None:
    """La F1 macro compte les types à égalité : un type sacrifié y apparaît."""
    gold = [{(0, 3, "produit")}, {(0, 3, "produit")}, {(0, 4, "date")}]
    predicted = [{(0, 3, "produit")}, {(0, 3, "produit")}, {(0, 4, "montant")}]
    scores = entity_scores(gold, predicted)
    table = per_label_frame(gold, predicted, labels=["produit", "date", "montant"])

    assert "micro" in set(table["label"])
    assert scores["entity_f1"] > scores["macro_f1"], "un type sacrifié doit se voir dans la macro"
    date_row = table[table["label"] == "date"].iloc[0]
    assert float(date_row["f1"]) == 0.0
    # La table porte une ligne par type **plus** la ligne ``micro`` : le support de ``micro`` est le
    # nombre d'entités de référence, et la somme des supports par type le retrouve.
    micro_support = int(table[table["label"] == "micro"].iloc[0]["support"])
    assert micro_support == sum(len(spans) for spans in gold)
    assert int(table["support"].sum()) == 2 * micro_support


def test_errors_are_classified_with_their_context() -> None:
    """Les erreurs sont classées (inventée, bornes, manquée) et situées dans leur phrase."""
    documents = pd.DataFrame(
        {
            "msg_id": ["MSG-0001", "MSG-0002"],
            "text": ["CMD-1234 : le casque est arrive", "regle 89,90 EUR pour le produit"],
        }
    )
    gold = pd.DataFrame(
        [
            {"msg_id": "MSG-0001", "start": 0, "end": 8, "label": "commande"},
            {"msg_id": "MSG-0001", "start": 0, "end": 8, "label": "commande", "surface": "CMD-1234"},
            {"msg_id": "MSG-0002", "start": 0, "end": 5, "label": "date", "surface": "regle"},
            {
                "msg_id": "MSG-0002",
                "start": 9,
                "end": 17,
                "label": "montant",
                "surface": "89,90 EU",
            },
        ]
    )
    predicted = pd.DataFrame(
        [
            # bornes décalées sur la commande de MSG-0001
            {"msg_id": "MSG-0001", "start": 0, "end": 7, "label": "commande", "confidence": 0.9},
            # mention inventée sur MSG-0001
            {"msg_id": "MSG-0001", "start": 11, "end": 16, "label": "produit", "confidence": 0.4},
            # montant juste sur MSG-0002
            {"msg_id": "MSG-0002", "start": 9, "end": 17, "label": "montant", "confidence": 1.0},
        ]
    )
    frame = error_frame(documents, gold, predicted, limit=10)
    kinds = set(frame["kind"])

    assert {"bornes", "inventee", "manquee"} <= kinds
    assert frame["context"].str.len().gt(0).all()
    assert set(frame["label"]) <= {"commande", "produit", "montant", "date"}


def test_the_confidence_table_measures_what_the_level_announces() -> None:
    """La précision observée par tranche est mesurée, et l'écart de confiance a un signe."""
    gold = pd.DataFrame(
        [
            {"msg_id": "MSG-0001", "start": 0, "end": 3, "label": "produit", "surface": "aaa"},
            {"msg_id": "MSG-0001", "start": 5, "end": 8, "label": "montant", "surface": "bbb"},
        ]
    )
    predicted = pd.DataFrame(
        [
            {"msg_id": "MSG-0001", "start": 0, "end": 3, "label": "produit", "confidence": 0.95},
            {"msg_id": "MSG-0001", "start": 5, "end": 6, "label": "montant", "confidence": 0.15},
        ]
    )
    table = confidence_table(predicted, gold, bins=5)

    assert len(table) == 5
    edges = (0.0, 0.2, 0.4, 0.6, 0.8)
    assert set(table["bucket"]) == {f"[{low:.1f}; {low + 0.2:.1f}]" for low in edges}
    high = table[table["bucket"] == "[0.8; 1.0]"].iloc[0]
    low = table[table["bucket"] == "[0.0; 0.2]"].iloc[0]
    assert int(high["n_correct"]) == 1 and float(high["precision"]) == 1.0
    assert int(low["n_correct"]) == 0 and float(low["precision"]) == 0.0
    assert 0.0 <= confidence_gap(predicted, gold) <= 1.0


def test_latency_and_descriptions_are_published() -> None:
    """Les latences se résument, et chaque métrique porte sa définition."""
    stats = latency_stats([1.0, 2.0, 3.0, 4.0])

    assert stats["latency_p50_ms"] == pytest.approx(2.5, abs=0.01)
    assert stats["latency_max_ms"] == 4.0
    assert percentile([1.0, 2.0, 3.0], 1.0) == pytest.approx(3.0)
    assert latency_stats([])["latency_mean_ms"] == 0.0
    documented = describe_metrics()
    assert "entity_f1" in documented and documented["entity_f1"]
    assert describe_metrics(["entity_f1"])["entity_f1"] == documented["entity_f1"]


def test_span_sets_align_annotations_by_message(spans) -> None:
    """La conversion table -> ensembles conserve l'ordre des messages et ne perd aucune entité."""
    grouped = span_sets(spans)

    assert sum(len(item) for item in grouped) == len(spans)
    assert all(isinstance(item, set) for item in grouped)


def test_score_arrays_matches_the_metric_functions() -> None:
    """``score_arrays`` (utilisé par les notebooks) rend exactement les métriques du module."""
    direct = entity_scores(GOLD, PREDICTED)
    flattened = score_arrays(GOLD, PREDICTED)

    assert flattened["entity_f1"] == direct["entity_f1"]
    assert "partial_f1" in flattened


def test_the_trainer_measures_validation_at_entity_level(training_outcome) -> None:
    """L'entraînement produit un historique et des métriques de validation préfixées ``val_``."""
    metrics = training_outcome.metrics

    assert training_outcome.n_train_documents > 0
    assert training_outcome.n_train_entities > 0
    assert training_outcome.n_val_documents > 0
    assert 0.0 <= metrics["val_entity_f1"] <= 1.0
    assert 0.0 <= metrics["val_entity_precision"] <= 1.0
    assert 0.0 <= metrics["val_partial_f1"] <= 1.0
    assert metrics["alignment_aligned_rate"] == 1.0, "aucune annotation ne doit être ignorée"
    assert metrics["alignment_ignored"] == 0.0
    assert training_outcome.history, "l'historique des époques doit être archivé"
    assert not training_outcome.per_label.empty
    assert "micro" in set(training_outcome.per_label["label"])


def test_the_trainer_serialises_its_outcome(training_outcome) -> None:
    """Le bilan est sérialisable : un rapport en JSON ne doit pas buter sur un type numpy."""
    payload = training_outcome.to_dict()

    assert payload["metrics"]["val_entity_f1"] == training_outcome.metrics["val_entity_f1"]
    assert payload["n_train_documents"] == training_outcome.n_train_documents
    assert payload["top_errors"] == [] or isinstance(payload["top_errors"], list)


def _validation_scores(metrics: dict[str, float]) -> dict[str, float]:
    """Return the deterministic validation metrics.

    La latence est une **mesure**, pas un score : deux runs identiques ne la rendent pas identique.
    La comparaison de reproductibilité porte donc sur les scores, et les prédictions sont comparées
    mention par mention juste à côté.

    Args:
        metrics: Metrics of a training outcome.

    Returns:
        The ``val_`` metrics, latency excluded.
    """
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("val_") and "latency" not in key
    }


def test_a_single_epoch_training_is_reproducible(app_config, documents, spans) -> None:
    """Deux entraînements à graine fixée produisent les mêmes prédictions et les mêmes scores."""
    first_model = build_model(app_config.model_dump())
    second_model = build_model(app_config.model_dump())
    first = EntityTrainer(
        first_model,
        config=app_config.train.model_dump(),
        primary_metric=app_config.metrics.primary,
    ).run(documents, spans)
    second = EntityTrainer(
        second_model,
        config=app_config.train.model_dump(),
        primary_metric=app_config.metrics.primary,
    ).run(documents, spans)

    texts = [str(text) for text in documents["text"]]
    identifiers = list(documents["msg_id"])
    first_mentions = first_model.predict(texts, ids=identifiers)
    second_mentions = second_model.predict(texts, ids=identifiers)

    assert [mention.as_tuple() for found in first_mentions for mention in found] == [
        mention.as_tuple() for found in second_mentions for mention in found
    ]
    assert _validation_scores(first.metrics) == _validation_scores(second.metrics)
    assert first.n_train_entities == second.n_train_entities


def test_the_trainer_refuses_to_fit_without_annotations(app_config, documents, spans) -> None:
    """Sans annotations, il n'y a rien à apprendre : le trainer lève, il ne devine pas."""
    trainer = EntityTrainer(
        build_model(app_config.model_dump()), config=app_config.train.model_dump()
    )

    # Une table d'annotations de la bonne forme mais vide : rien à apprendre, et le message le dit.
    with pytest.raises(ValueError, match="nothing to learn"):
        trainer.run(documents, spans.iloc[:0])
    # Une table qui n'est même pas une table d'annotations : le message nomme ce qui manque.
    with pytest.raises(ValueError, match="missing the columns"):
        trainer.run(documents, documents.iloc[:0])


def test_every_published_mention_carries_its_provenance(predictions) -> None:
    """Chaque mention dit d'où elle vient et ce qu'elle vaut : c'est ce que lit le conseiller."""
    mentions = predictions

    sources = set(mentions["source"]) if not mentions.empty else {"modele"}
    assert sources <= {"regle", "modele"}
    for row in mentions.itertuples(index=False):
        assert EntityMention(
            msg_id=str(row.msg_id),
            start=int(row.start),
            end=int(row.end),
            label=str(row.label),
            surface=str(row.surface),
            source=str(row.source),
            confidence=float(row.confidence),
        ).as_tuple() == (int(row.start), int(row.end), str(row.label))
