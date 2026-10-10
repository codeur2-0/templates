"""Entraînement : l'ajustement, la comparaison des stratégies et le verdict de validation.

L'entraîneur est le seul endroit du projet qui **décide** : il choisit le split, ajuste la stratégie
servie, mesure les alternatives sur le même échantillon de validation et publie un verdict. Trois
propriétés sont verrouillées ici, parce que ce sont celles qui se dégradent en silence :

* le split est lu dans la colonne ``split`` (et non par position) : un corpus retrié n'apprend pas
  sur son propre test ;
* la comparaison contient **une ligne par algorithme**, la stratégie servie comprise — un tableau
  incomplet ferait croire à une supériorité qu'on n'a pas mesurée ;
* le verdict se lit uniquement sur la métrique contractuelle mesurée.

L'ajustement des fixtures est celui de la baseline extractive : le chemin exercé est celui de la
production (découpage, comparaison, verdict, persistance), sans payer l'entraînement du réseau.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.lead import LeadSummarizer
from src.schemas.config import AppConfig
from src.training.metrics import verdict_from_metrics
from src.training.trainer import (
    SummaryTrainer,
    SummaryTrainingOutcome,
    comparison_columns,
    coverage_columns,
)
from src.utils.config_access import node
from src.utils.paths import ProjectPaths


def _trainer(
    model: LeadSummarizer,
    app_config: AppConfig,
    paths: ProjectPaths,
    *,
    algorithm: str = "lead",
) -> SummaryTrainer:
    """Return a trainer wired like ``TrainPipeline`` wires one, on a throw-away layout."""
    return SummaryTrainer(
        model,
        config=app_config.train.model_dump(),
        paths=paths,
        algorithm=algorithm,
        text_column=str(node(app_config, "model").get("text_column", "text")),
        id_column=str(app_config.data.id_column or "doc_id"),
        compare=("lead",),
        comparison_documents=8,
        baseline=model,
    )


def test_the_trainer_measures_the_extractive_floor_on_validation(
    training_outcome: SummaryTrainingOutcome,
    val_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """L'ajustement publie ses métriques d'ajustement **et** de validation, sur le bon effectif."""
    documents, _, _ = val_split

    assert training_outcome.algorithm == "lead"
    assert training_outcome.fit_metrics
    assert training_outcome.validation_metrics
    assert training_outcome.validation_metrics["n_documents"] == float(len(documents))
    assert 0.0 <= training_outcome.validation_metrics["rouge1_f"] <= 1.0
    assert training_outcome.duration_seconds >= 0.0


def test_the_validation_publishes_the_coverage_columns_of_the_family(
    training_outcome: SummaryTrainingOutcome,
) -> None:
    """La couverture des faits est mesurée à l'entraînement, pas seulement à l'évaluation.

    Un entraînement qui ne publierait que le ROUGE laisserait passer un modèle qui apprend la
    forme des résumés de référence en perdant les valeurs qui les rendent utiles.
    """
    metrics = training_outcome.validation_metrics

    assert {"fact_coverage", "fact_precision"} <= set(metrics)
    assert metrics["fact_coverage"] <= 1.0
    # La couverture est publiée **par type de fait** : c'est ce qui distingue un modèle qui a
    # appris les gabarits fréquents d'un modèle qui a appris à recopier un identifiant.
    assert set(coverage_columns()) <= set(metrics)


def test_the_comparison_publishes_one_row_per_algorithm(
    training_outcome: SummaryTrainingOutcome,
) -> None:
    """Le tableau de comparaison contient la stratégie servie, une seule fois, et ses colonnes."""
    comparison = training_outcome.comparison

    assert not comparison.empty
    assert set(comparison_columns()) <= set(comparison.columns)
    assert comparison["algorithm"].is_unique
    assert "lead" in set(comparison["algorithm"])
    assert comparison["rouge1_f"].between(0.0, 1.0).all()
    assert comparison["fact_coverage"].between(0.0, 1.0).all()


def test_the_verdict_of_the_run_follows_the_measured_metric(
    training_outcome: SummaryTrainingOutcome, app_config: AppConfig
) -> None:
    """Le verdict est déduit de la mesure : aucun résultat n'est écrit en dur.

    C'est la garantie que le seuil publié par la famille (``min_primary``) est bien celui qui rend
    le verdict, et non une constante recopiée dans le code du projet.
    """
    minimum = app_config.metrics.min_primary
    assert minimum is not None, "la famille doit déclarer un seuil contractuel"
    verdict, detail = verdict_from_metrics(
        training_outcome.validation_metrics,
        primary=str(app_config.metrics.primary),
        minimum=float(minimum),
        direction=str(app_config.metrics.direction),
    )

    assert verdict in {"conforme", "non conforme"}
    assert detail["metric"] == app_config.metrics.primary
    assert detail["threshold"] == float(minimum)
    assert detail["observed"] == round(training_outcome.validation_metrics["rouge1_f"], 4)


def test_an_empty_train_split_is_refused_instead_of_fitting_nothing(
    lead_model: LeadSummarizer,
    app_config: AppConfig,
    tiny_paths: ProjectPaths,
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
) -> None:
    """Un corpus sans ligne de train lève ``ValueError`` : entraîner sur rien produirait du vide."""
    trainer = _trainer(lead_model, app_config, tiny_paths)
    no_train = documents.assign(split="test")

    with pytest.raises(ValueError, match="Training split is empty"):
        trainer.run(no_train, references, facts)


def test_two_runs_of_the_same_configuration_measure_the_same_numbers(
    app_config: AppConfig,
    tiny_paths: ProjectPaths,
    train_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    val_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    facts: pd.DataFrame,
) -> None:
    """Deux ajustements de la même configuration donnent les mêmes métriques de validation.

    Le déterminisme est vérifié **au niveau de la mesure**, pas seulement de la graine : c'est le
    couple (ajustement, scoring) qui doit être reproductible, sinon les chiffres du rapport ne
    veulent rien dire d'un run à l'autre. Les latences sont exclues de la comparaison : elles
    mesurent la machine, pas le modèle.
    """
    train_documents, train_references, _ = train_split
    val_documents, val_references, _ = val_split
    measured: list[dict[str, float]] = []
    for _ in range(2):
        model = LeadSummarizer(compression=0.45, max_output_tokens=60, seed=13)
        model.fit(train_documents, train_references)
        trainer = _trainer(model, app_config, tiny_paths)
        measured.append(trainer.score(model, val_documents, val_references, facts))

    stable = [
        {name: value for name, value in metrics.items() if not name.startswith("latency")}
        for metrics in measured
    ]

    assert stable[0] == stable[1]
    assert stable[0]["n_documents"] == float(len(val_documents))
    assert set(coverage_columns()) <= set(stable[0])
