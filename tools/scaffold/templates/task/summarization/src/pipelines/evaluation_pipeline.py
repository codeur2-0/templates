"""Évaluation : mesurer l'artefact entraîné sur le split de test, une seule fois.

``mode=evaluate`` ne réentraîne rien et ne règle rien. Il recharge l'artefact écrit
par ``mode=train``, mesure le split de **test**, compare la stratégie servie à ses
références **sur les mêmes lignes** (le résumé vide, la baseline extractive publiée
par la famille, et l'algorithme de comparaison déclaré par la configuration quand
il existe) puis délègue la lecture des chiffres au constructeur de rapport.

Deux artefacts sont toujours écrits :

* ``artifacts/metrics/evaluation_metrics.json`` — lisible par une machine, pour un portail de CI ;
* ``artifacts/reports/evaluation_report.md`` — lisible par un humain, avec
  le verdict contractuel, la couverture par type de fait, la ventilation
  par segment, les erreurs relues et les limites de la mesure.

Le pipeline refuse de tourner sans artefact de modèle : évaluer un modèle vide produirait un
rapport plausible sur rien. Les références (résumé vide, baseline extractive, algorithmes comparés)
sont ajustées sur le **train** et départagées sur le **test** : aucune ligne de test ne sert à
régler quoi que ce soit, sinon la comparaison ne mesurerait plus rien.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.data.loaders import SummaryCorpusLoader
from src.evaluation.evaluator import SummaryEvaluation, SummaryEvaluator
from src.evaluation.reports import ReportBuilder
from src.models import build_model, load_model
from src.models.contract import BaseTextGenerator
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    """Recharger l'artefact, mesurer le test, écrire le rapport et les figures."""

    name = "evaluate"

    def _execute(self) -> PipelineResult:
        """Run the evaluation flow.

        Returns:
            The pipeline result, whose payload is the :class:`SummaryEvaluation`.

        Raises:
            FileNotFoundError: When the artefact produced by ``mode=train`` is missing.
        """
        self.paths.ensure()
        loader = SummaryCorpusLoader(
            self.paths,
            dataset_name=str(self.config.data.dataset_name),
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        model = self._load_model()
        documents, references, facts = loader.load_corpus()
        test_documents, test_references, test_facts = _select_split(
            documents, references, facts, "test"
        )
        train_documents, train_references, _ = _select_split(
            documents, references, facts, "train"
        )
        val_documents, val_references, _ = _select_split(documents, references, facts, "val")
        model_node = node(self.config, "model")

        evaluator = SummaryEvaluator(
            model,
            config=self.config.model_dump(),
            metrics_config=self.config.metrics.model_dump(),
            paths=self.paths,
            text_column=str(model_node.get("text_column", "text")),
            id_column=str(model_node.get("id_column", "doc_id")),
            baselines=self._baselines(
                (train_documents, train_references), (val_documents, val_references)
            ),
        )
        result = evaluator.evaluate(test_documents, test_references, test_facts)

        metrics_path = write_json(
            self.paths.metrics_dir / "evaluation_metrics.json",
            {
                **result.metrics,
                "verdict": result.verdict,
                "verdict_detail": dict(result.verdict_detail),
            },
        )
        bundle = ReportBuilder(self.config.model_dump(), paths=self.paths).build(result)
        logger.info(
            "Évaluation terminée : ROUGE-1 {}, couverture {}, verdict {}",
            result.metrics.get("rouge1_f", 0.0),
            result.metrics.get("fact_coverage", 0.0),
            result.verdict,
        )
        return PipelineResult(
            name=self.name,
            metrics=result.metrics,
            artifacts=[
                str(metrics_path),
                str(bundle.report),
                *[str(path) for path in bundle.tables.values()],
                *[str(path) for path in bundle.figures.values()],
            ],
            payload=result,
            messages=self._notes(result),
        )

    def _load_model(self) -> BaseTextGenerator:
        """Reload the artefact written by the training pipeline.

        Returns:
            The fitted strategy.

        Raises:
            FileNotFoundError: When the artefact is missing.
        """
        target = self.paths.models_dir / str(self.config.train.artifacts.model_file)
        if not target.is_file():
            msg = (
                f"Artefact de modèle introuvable : {target}. Exécuter `make train` "
                "(ou `python -m src.main mode=train`) avant d'évaluer."
            )
            raise FileNotFoundError(msg)
        model = load_model(target, config=self.config.model_dump())
        logger.info("Artefact rechargé : {} ({})", target, model.summary())
        return model

    def _baselines(
        self,
        train: tuple[pd.DataFrame, pd.DataFrame],
        validation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> dict[str, BaseTextGenerator]:
        """Build and fit the reference strategies measured on the same test lines.

        Trois décisions, toutes visibles dans le rapport :

        * **une référence s'ajuste sur le train**, avec le même couple de validation que le modèle
          servi : un ``textrank`` calibré sur le test ne serait plus une référence, et ``lead``
          n'apprend rien mais le dit (son ajustement publie zéro paramètre) ;
        * **la stratégie servie est mesurée depuis son artefact**, jamais réajustée ici : deux
          exécutions du même algorithme ne donnent pas les mêmes poids, donc la réentraîner
          afficherait deux lignes « transformer_tiny » aux chiffres différents dans le même
          rapport ;
        * **une référence que la fabrique ne sait pas construire est ignorée avec un
          avertissement**, jamais au prix de l'évaluation entière ; un même algorithme enregistré
          sous deux noms (``baseline_extractive`` et ``lead``) n'est mesuré qu'une fois, sinon le
          rapport publie deux lignes identiques avec un effectif doublé.

        Args:
            train: ``(documents, references)`` du split d'entraînement, servis à l'ajustement.
            validation: ``(documents, references)`` du split de validation (calibration des
                références qui en ont besoin).

        Returns:
            Mapping ``nom publié -> stratégie ajustée``.
        """
        train_documents, train_references = train
        val_documents, val_references = validation
        validation_pair = (
            (val_documents, val_references)
            if not val_documents.empty and not val_references.empty
            else None
        )
        served = str(node(self.config, "model").get("algorithm") or "")
        baselines: dict[str, BaseTextGenerator] = {}
        candidates = {"baseline_extractive": str(self.config.metrics.baseline or "lead")}
        model_node = node(self.config, "model")
        for name in model_node.get("strategies_to_compare", []) or []:
            candidates.setdefault(str(name), str(name))
        # Un **algorithme** ne se mesure qu'une fois, même s'il est enregistré sous plusieurs noms
        # (`baseline_extractive` est le nom publié de `lead`) : sans ce garde-fou, la même stratégie
        # apparaît deux fois dans le rapport, avec un effectif doublé et deux lignes identiques — de
        # quoi faire douter du reste du tableau.
        measured: dict[str, str] = {}
        for name, algorithm in candidates.items():
            if algorithm in {"aucune", "none", ""}:
                continue
            if algorithm == served:
                logger.info(
                    "Stratégie servie '{}' lue depuis son artefact : pas de réajustement ici",
                    algorithm,
                )
                continue
            if algorithm in measured:
                logger.info(
                    "Référence '{}' déjà mesurée sous le nom '{}' : pas de doublon",
                    algorithm,
                    measured[algorithm],
                )
                continue
            try:
                candidate = build_model(self.config, algorithm=algorithm)
            except (ValueError, KeyError) as exc:
                logger.warning("Baseline '{}' indisponible : {}", algorithm, exc)
                continue
            candidate.fit(train_documents, train_references, validation=validation_pair)
            baselines[name] = candidate
            measured[algorithm] = name
        logger.info(
            "Références ajustées sur {} document(s) de train : {}",
            len(train_documents),
            sorted(baselines) or "aucune",
        )
        return baselines

    def _notes(self, result: SummaryEvaluation) -> list[str]:
        """Human-readable lines published with the evaluation.

        Args:
            result: The :class:`SummaryEvaluation` of the test split.

        Returns:
            The summary lines, verdict and warnings included.
        """
        metrics = dict(result.metrics)
        baselines = dict(result.baselines)
        lines = [
            f"ROUGE-1 {metrics.get('rouge1_f', 0.0):.4f}, "
            f"ROUGE-L {metrics.get('rouge_l_f', 0.0):.4f}, "
            f"couverture des faits {metrics.get('fact_coverage', 0.0):.4f} sur "
            f"{int(metrics.get('n_documents', 0))} document(s) de test.",
            f"{metrics.get('unsupported_share', 0.0):.1%} des résumés contiennent au moins "
            "une valeur "
            "absente du document : la détection d'hallucination s'arrête aux valeurs vérifiables.",
        ]
        for name, block in sorted(baselines.items()):
            lines.append(
                f"Référence `{name}` sur les mêmes lignes : "
                f"ROUGE-1 {block.get('rouge1_f', 0.0):.4f}, "
                f"couverture {block.get('fact_coverage', 0.0):.4f}."
            )
        detail = dict(result.verdict_detail or {})
        if detail.get("threshold") is not None:
            lines.append(
                f"Verdict {result.verdict} : "
                f"{detail.get('metric')} = {detail.get('observed')} contre un seuil de "
                f"{detail.get('threshold')} (marge {detail.get('margin')})."
            )
        return lines


def _select_split(
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return one split of an already-loaded corpus, without reading the disk again.

    Args:
        documents: Document table of the whole corpus.
        references: Reference summaries of the whole corpus.
        facts: Fact table of the whole corpus.
        split: Split name (``train``, ``val``, ``calibration`` or ``test``).

    Returns:
        The three tables restricted to that split, re-indexed.
    """
    selected = documents.loc[documents["split"] == split].reset_index(drop=True)
    keys = set(selected["doc_id"])
    return (
        selected,
        references.loc[references["doc_id"].isin(keys)].reset_index(drop=True),
        facts.loc[facts["doc_id"].isin(keys)].reset_index(drop=True),
    )


def report_path(paths: ProjectPaths) -> Path:
    """Return the expected path of the evaluation report.

    Args:
        paths: Project filesystem layout.

    Returns:
        The report path inside ``artifacts/reports``.
    """
    return paths.reports_dir / "evaluation_report.md"


__all__ = ["EvaluationPipeline", "report_path"]
