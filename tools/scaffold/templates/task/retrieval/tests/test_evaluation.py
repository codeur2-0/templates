"""Évaluation : mesures de recherche, abstention, segments et rapport.

Ces tests portent sur la couche qui décide de la **publication** d'un chiffre. Trois règles y sont
vérifiées, parce qu'une erreur à ce niveau ne se voit pas dans les journaux :

* une question hors corpus est **exclue du dénominateur** du rappel : elle n'a pas de document
  pertinent, elle ne peut donc pas être « non retrouvée » ;
* une réponse sans citation produit ``citation_precision = NaN`` et non ``0`` — l'abstention n'est
  pas une citation fausse, et confondre les deux rendrait le seuil de refus optimisable au mauvais
  endroit ;
* le rapport publié est **celui de la mesure** : ses chiffres et son verdict viennent du même objet
  que le fichier de métriques, jamais d'un recalcul parallèle.

L'évaluateur est exercé sur le corpus minuscule des fixtures, avec un modèle réellement entraîné :
un évaluateur testé sur des données inventées mesurerait le test, pas la mesure.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pandas as pd
import pytest

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.reports import ReportBuilder
from src.models.base import BaseModel
from src.schemas.config import AppConfig
from src.utils.paths import ProjectPaths


def _evaluator(
    model: BaseModel,
    app_config: AppConfig,
    paths: ProjectPaths,
    *,
    ks: Sequence[int] = (1, 5, 10),
    answer_k: int = 5,
) -> Evaluator:
    """Build the evaluator with the very configuration used by ``mode=evaluate``."""
    return Evaluator(
        model,
        config=app_config.model_dump(),
        metrics_config=app_config.metrics.model_dump(),
        paths=paths,
        ks=ks,
        answer_k=answer_k,
    )


@pytest.fixture(scope="module")
def test_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """Return the test split of the tiny corpus."""
    return queries[queries["split"] == "test"].reset_index(drop=True)


def test_evaluation_reports_one_row_per_question(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Chaque question évaluée laisse une trace exploitable : c'est elle qui sert au diagnostic."""
    result = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)

    assert result.n_questions == len(test_queries)
    assert len(result.per_question) == len(test_queries)
    assert result.n_answerable == int((test_queries["answer_type"] != "unanswerable").sum())
    assert set(result.per_question["query_id"]) == set(test_queries["query_id"])
    expected = {"difficulty", "intent", "first_relevant_rank", "recall_at_5", "abstained"}
    assert expected <= set(result.per_question.columns)
    # Le rang du premier document pertinent vaut 0 (aucun) ou un entier positif : il ne doit jamais
    # être négatif, ce qui signalerait un classement construit à l'envers.
    assert (result.per_question["first_relevant_rank"] >= 0).all()


def test_out_of_corpus_questions_leave_the_recall_denominator(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Une question sans réponse ne peut pas être « non retrouvée » : elle sort du dénominateur."""
    result = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)

    assert result.metrics["n_scored_recall_at_5"] == float(result.n_answerable)
    unanswerable = result.per_question[result.per_question["answer_type"] == "unanswerable"]
    assert len(unanswerable) > 0, "le corpus de test doit contenir des questions hors corpus"
    # La note de pertinence d'une question hors corpus est vide : son rappel n'est pas « mauvais »,
    # il n'existe pas — et il doit être mesuré de la même façon dans la ligne et dans l'agrégat.
    assert unanswerable["recall_at_5"].isna().all()
    assert (unanswerable["n_relevant_retrieved"] == 0).all()
    assert (unanswerable["first_relevant_rank"] == 0).all()
    assert (unanswerable["should_abstain"] == 1).all()
    # La métrique agrégée est la moyenne des questions qui ont un document pertinent.
    answerable = result.per_question[result.per_question["answer_type"] != "unanswerable"]
    assert result.metrics["recall_at_5"] == pytest.approx(
        float(answerable["recall_at_5"].mean(skipna=True))
    )


def test_the_model_beats_the_trivial_references(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Les planchers sont mesurés sur les mêmes questions : ils rendent le score lisible."""
    result = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)
    primary = str(app_config.metrics.primary)

    assert {"random", "corpus_order"} <= set(result.baselines)
    for name, baseline in result.baselines.items():
        assert primary in baseline, f"le plancher '{name}' ne mesure pas {primary}"
        assert result.metrics[primary] > baseline[primary], (
            f"le modèle ne bat pas le plancher '{name}' : {result.metrics[primary]:.4f} "
            f"contre {baseline[primary]:.4f}"
        )


def test_a_model_that_always_abstains_is_measured_not_guessed(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Un modèle qui refuse tout ne cite rien : la précision des citations est NaN, pas zéro.

    C'est la distinction qui protège la lecture du rapport : refuser une question sans réponse est
    un comportement correct, et le compter comme une citation fausse ferait passer le meilleur
    comportement possible pour un échec.
    """
    # Copie du modèle partagé : un test ne doit pas modifier la fixture des autres.
    refusing = copy.deepcopy(fitted_model)
    refusing.abstention_threshold = 1e12
    result = _evaluator(refusing, app_config, tiny_paths).evaluate(test_queries, documents)

    assert result.per_question["abstained"].eq(1).all()
    assert result.per_question["citation_precision"].isna().all()
    # ``flatten_metrics`` ne publie que des nombres finis : une métrique non applicable est
    # **absente** du fichier de métriques, et le rapport la rend en ``n/a``.
    assert "citation_precision" not in result.metrics
    # En revanche, refuser une question qui **avait** une réponse se voit immédiatement.
    answerable = result.per_question[result.per_question["answer_type"] != "unanswerable"]
    assert (answerable["abstention_correct"] == 0.0).all()
    assert result.metrics["abstention_recall"] == pytest.approx(1.0)


def test_cut_offs_are_exactly_the_requested_ones(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Les métriques publiées suivent les points de coupe demandés, ni plus ni moins."""
    result = _evaluator(fitted_model, app_config, tiny_paths, ks=(1, 5)).evaluate(
        test_queries, documents
    )

    assert result.ks == (1, 5)
    assert "recall_at_5" in result.metrics
    assert "recall_at_3" not in result.metrics
    assert "recall_at_10" not in result.metrics


def test_segments_cover_every_difficulty_of_the_test_split(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """La ventilation par difficulté doit totaliser exactement les questions évaluées."""
    result = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)

    difficulty = {
        key.split("=", 1)[1]: value
        for key, value in result.segments.items()
        if key.startswith("difficulty=")
    }
    assert set(difficulty) == set(test_queries["difficulty"])
    for label, metrics in difficulty.items():
        assert metrics["n_questions"] == float((test_queries["difficulty"] == label).sum())
        if label == "hors_corpus":
            # Ce segment n'a pas de document pertinent : il se juge à l'abstention, et publier un
            # rappel pour lui reviendrait à noter une question sans réponse.
            assert "recall_at_5" not in metrics
            assert "abstention_rate" in metrics
        else:
            assert "recall_at_5" in metrics
            assert "mrr" in metrics


def test_evaluation_is_reproducible(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Deux évaluations identiques donnent les mêmes chiffres : les planchers sont graines."""
    first = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)
    second = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)

    # La latence est une mesure de temps : elle est exclue de la comparaison, comme dans le
    # rapport, où elle est publiée à part justement parce qu'elle n'est pas reproductible.
    timing = {name for name in first.metrics if "latency" in name}
    assert timing, "les métriques de latence doivent être publiées"
    assert {name: value for name, value in first.metrics.items() if name not in timing} == {
        name: value for name, value in second.metrics.items() if name not in timing
    }
    assert first.baselines == second.baselines
    pd.testing.assert_frame_equal(
        first.per_question.drop(columns=["latency_ms"]),
        second.per_question.drop(columns=["latency_ms"]),
    )


def test_evaluator_refuses_an_empty_question_frame(
    fitted_model: BaseModel, app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Évaluer zéro question produirait un rapport plausible sur rien : c'est refusé."""
    empty = pd.DataFrame(columns=["query_id", "question", "difficulty", "answer_type"])
    with pytest.raises(ValueError, match="No question to evaluate"):
        _evaluator(fitted_model, app_config, tiny_paths).evaluate(empty)


def test_report_builder_publishes_the_measured_verdict(
    fitted_model: BaseModel,
    app_config: AppConfig,
    documents: pd.DataFrame,
    test_queries: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> None:
    """Le rapport, ses tableaux et son verdict viennent de la mesure, pas d'un recalcul."""
    result = _evaluator(fitted_model, app_config, tiny_paths).evaluate(test_queries, documents)
    builder = ReportBuilder(app_config.model_dump(), paths=tiny_paths)

    bundle = builder.build(
        result,
        model_summary={
            "name": fitted_model.name,
            "framework": fitted_model.framework,
            "algorithm": fitted_model.algorithm,
            "task": fitted_model.task,
            "n_chunks": len(fitted_model.chunks),
        },
        metadata={"n_documents": len(documents), "seed": app_config.seed},
    )

    assert bundle.report_path.exists()
    assert bundle.verdict in {"conforme", "non conforme", "indéterminé", "sans seuil contractuel"}
    minimum = app_config.metrics.min_primary
    primary = float(result.metrics[str(app_config.metrics.primary)])
    expected = "conforme" if minimum is None or primary >= minimum else "non conforme"
    assert bundle.verdict == expected

    report = bundle.report_path.read_text(encoding="utf-8")
    assert str(app_config.metrics.primary) in report
    assert f"{primary:.4f}" in report, "le rapport doit citer la valeur réellement mesurée"
    assert "abstention" in report
    names = {path.name for path in bundle.artifacts}
    assert {"evaluation_report.md", "per_question.csv", "segment_metrics.csv"} <= names
    assert (tiny_paths.reports_dir / "segment_metrics.csv").exists()


def test_report_builder_refuses_to_publish_an_empty_measurement(
    app_config: AppConfig, tiny_paths: ProjectPaths
) -> None:
    """Un rapport sans métrique serait un document creux : le constructeur refuse d'en écrire un."""
    empty = EvaluationResult(metrics={}, per_question=pd.DataFrame())
    with pytest.raises(ValueError, match="no metric"):
        ReportBuilder(app_config.model_dump(), paths=tiny_paths).build(empty)
