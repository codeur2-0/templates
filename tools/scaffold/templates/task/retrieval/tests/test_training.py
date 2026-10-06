"""Métriques et entraînement : les formules, et ce qu'elles refusent de faire.

Une métrique de classement est facile à écrire faux — et un faux positif ici se paie en
semaines. Les tests fixent donc les cas limites qui distinguent les implémentations correctes des
implémentations approximatives : la normalisation de la nDCG par l'idéal atteignable, le rang du
premier passage pertinent, la moyenne qui **exclut** une question sans réponse plutôt que de la
compter zéro, et l'abstention qui n'est ni punie ni récompensée à tort.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.models import build_model
from src.schemas.config import AppConfig
from src.training.losses_metrics import (
    abstention_accuracy,
    answer_exact_match,
    answer_f1,
    average_precision_at_k,
    citation_precision,
    citation_recall,
    describe_metrics,
    hit_rate_at_k,
    mean_reciprocal_rank,
    metrics_for_task,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
)
from src.training.trainer import Trainer
from src.utils.paths import ProjectPaths


def test_recall_at_k_counts_the_share_of_relevant_documents() -> None:
    """Deux documents pertinents sur trois trouvés dans le top-3 : 2/3."""
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_recall_at_k_is_zero_when_the_cut_off_excludes_everything() -> None:
    """Un rappel dont le cut-off ne contient rien de pertinent vaut 0, pas NaN."""
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_precision_and_hit_rate_measure_different_things() -> None:
    """La précision compte les bons passages trouvés, le hit rate compte les questions servies."""
    ranking = ["a", "x", "b", "y", "z"]
    assert precision_at_k(ranking, {"a", "b"}, 5) == pytest.approx(2 / 5)
    assert hit_rate_at_k(ranking, {"a", "b"}, 5) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_mean_reciprocal_rank_rewards_the_first_hit() -> None:
    """Trouver la réponse au rang 1 doit battre le rang 3, et un échec doit valoir 0."""
    assert mean_reciprocal_rank(["a", "b"], {"b"}) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)
    assert mean_reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_normalises_by_the_ideal_ranking() -> None:
    """Avec deux pertinents, l'idéal est de 1.0 ; un ordre inversé coûte moins qu'une absence."""
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    reordered = ndcg_at_k(["b", "x", "a"], {"a", "b"}, 3)
    assert 0.0 < reordered < 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_average_precision_is_zero_without_relevant_documents() -> None:
    """Pas de pertinent dans la liste : MAP = 0, et surtout pas une division par zéro."""
    assert average_precision_at_k(["a"], {"z"}, 1) == 0.0
    assert average_precision_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_retrieval_metrics_aggregate_and_count_questions() -> None:
    """L'agrégat rapporte aussi le nombre de questions réellement notées."""
    # La troisième liste n'a aucun document pertinent : la question est *hors corpus* et sort du
    # dénominateur (sinon un système honnête serait puni pour avoir refusé de répondre).
    metrics = retrieval_metrics([["a"], ["b", "c"], ["x"]], [{"a"}, {"b"}, set()], ks=(1, 3))
    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert metrics["n_scored_recall_at_1"] == 2.0
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert "ndcg_at_3" in metrics


def test_answer_f1_rewards_partial_wording() -> None:
    """La réponse de référence et la réponse produite sont comparées token à token."""
    assert answer_f1("25 jours ouvrables par an", "25 jours ouvrables par an") == pytest.approx(1.0)
    assert answer_f1("25 jours", "25 jours ouvrables par an") < 1.0
    assert answer_exact_match("25 Jours.", "25 jours") == 1.0


def test_citation_metrics_separate_precision_from_recall() -> None:
    """Citer peu de choses justes et citer tout ne sont pas la même erreur."""
    assert citation_precision(["a", "x"], {"a"}) == pytest.approx(0.5)
    assert citation_recall(["a"], {"a", "b"}) == pytest.approx(0.5)
    # Aucune citation : la précision n'est *pas* définie (et vaut NaN), elle n'est pas nulle — une
    # abstention ne doit pas être comptée comme une citation fausse.
    assert math.isnan(citation_precision([], {"a"}))


def test_abstention_accuracy_rewards_both_directions() -> None:
    """Un refus correct n'est pas un refus à tout prix : les deux sens sont mesurés."""
    assert abstention_accuracy(abstained=True, should_abstain=True) == 1.0
    assert abstention_accuracy(abstained=False, should_abstain=False) == 1.0
    assert abstention_accuracy(abstained=True, should_abstain=False) == 0.0


def test_percentile_handles_an_empty_sample() -> None:
    """Une latence sans observation est NaN, pas 0 : 0 ms serait un mensonge flatteur."""
    assert math.isnan(percentile([], 0.5))
    assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)


def test_metrics_are_documented_for_the_report() -> None:
    """Chaque métrique annoncée a une description : le rapport ne réinvente pas le texte."""
    names = metrics_for_task("retrieval")
    descriptions = describe_metrics(names)
    assert "recall_at_k" in descriptions
    assert "abstention_accuracy" in descriptions
    assert all(descriptions.get(name) for name in names if name in descriptions)


def test_trainer_monitors_the_validation_split(
    app_config: AppConfig, documents: pd.DataFrame, queries: pd.DataFrame, tmp_path: Path
) -> None:
    """``Trainer.run`` rend les métriques de validation et le nombre de questions notées."""
    model = build_model(app_config.model_dump())
    trainer = Trainer(
        model,
        config=app_config.train.model_dump(),
        paths=ProjectPaths.from_root(tmp_path),
        metric_names=("recall_at_5", "mrr", "citation_precision"),
        ks=(1, 5),
    )
    validation = queries[queries["split"] == "val"]
    outcome = trainer.run(documents, validation)
    assert outcome.n_validation_questions == len(validation)
    assert outcome.n_val_questions_scored > 0
    assert 0.0 <= outcome.metrics["val_recall_at_5"] <= 1.0
    assert math.isfinite(outcome.metrics["val_mrr"])
    assert outcome.fit_result.n_documents == len(documents)
    assert outcome.chunks_path is None or outcome.chunks_path.exists()


def test_trainer_reports_a_missed_contract_through_the_callback(
    app_config: AppConfig, documents: pd.DataFrame, queries: pd.DataFrame, tmp_path: Path
) -> None:
    """Un seuil contractuel manqué se lit dans le callback, pas dans les logs qu'on espère lire."""
    from src.training.callbacks import MetricThresholdCallback

    model = build_model(app_config.model_dump())
    gate = MetricThresholdCallback(monitor="val_recall_at_5", threshold=1.01, mode="max")
    trainer = Trainer(
        model,
        config=app_config.train.model_dump(),
        paths=ProjectPaths.from_root(tmp_path),
        metric_names=("recall_at_5",),
        ks=(1, 5),
        min_primary_metric=1.01,
        primary_metric="recall_at_5",
        callbacks=[gate],
    )
    outcome = trainer.run(documents, queries[queries["split"] == "val"])
    assert outcome.metrics["val_recall_at_5"] < 1.01
    assert gate.satisfied is False
