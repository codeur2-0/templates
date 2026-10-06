"""Évaluation : mesurer l'artefact entraîné sur le split de test, une seule fois.

``mode=evaluate`` ne réentraîne rien et ne règle rien. Il recharge l'artefact écrit par ``mode=train``,
mesure le split de **test**, compare la stratégie servie à ses références **sur les mêmes lignes** (le
résumé vide, la baseline extractive publiée par la famille, et l'algorithme de comparaison déclaré par
la configuration quand il existe) puis délègue la lecture des chiffres au constructeur de rapport.

Deux artefacts sont toujours écrits :

* ``artifacts/metrics/evaluation_metrics.json`` — lisible par une machine, pour un portail de CI ;
* ``artifacts/reports/evaluation_report.md`` — lisible par un humain, avec le verdict contractuel, la
  couverture par type de fait, la ventilation par segment, les erreurs relues et les limites de la
  mesure.

Le pipeline refuse de tourner sans artefact de modèle : évaluer un modèle vide produirait un rapport
plausible sur rien.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders import SummaryCorpusLoader
from src.evaluation.evaluator import SummaryEvaluator
from src.evaluation.reports import ReportBuilder
from src.models import build_model, load_model
from src.models.contract import BaseTextGenerator
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

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
        test_documents = documents.loc[documents["split"] == "test"].reset_index(drop=True)
        keys = set(test_documents["doc_id"])
        test_references = references.loc[references["doc_id"].isin(keys)].reset_index(drop=True)
        test_facts = facts.loc[facts["doc_id"].isin(keys)].reset_index(drop=True)
        model_node = node(self.config, "model")

        evaluator = SummaryEvaluator(
            model,
            config=self.config.model_dump(),
            metrics_config=self.config.metrics.model_dump(),
            paths=self.paths,
            text_column=str(model_node.get("text_column", "text")),
            id_column=str(model_node.get("id_column", "doc_id")),
            baselines=self._baselines(),
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

    def _baselines(self) -> dict[str, BaseTextGenerator]:
        """Build the reference strategies measured on the same test lines.

        Returns:
            Mapping ``nom publié -> stratégie``; a reference the factory cannot build is ignored with
            a warning rather than failing the evaluation.
        """
        baselines: dict[str, BaseTextGenerator] = {}
        candidates = {"baseline_extractive": str(self.config.metrics.baseline or "lead")}
        model_node = node(self.config, "model")
        for name in model_node.get("strategies_to_compare", []) or []:
            candidates.setdefault(str(name), str(name))
        for name, algorithm in candidates.items():
            if algorithm in {"aucune", "none", ""}:
                continue
            try:
                baselines[name] = build_model(self.config, algorithm=algorithm)
            except (ValueError, KeyError) as exc:
                logger.warning("Baseline '{}' indisponible : {}", algorithm, exc)
        return baselines

    def _notes(self, result: object) -> list[str]:
        """Human-readable lines published with the evaluation.

        Args:
            result: The :class:`SummaryEvaluation` of the test split.

        Returns:
            The summary lines, verdict and warnings included.
        """
        metrics = getattr(result, "metrics", {})
        baselines = getattr(result, "baselines", {})
        lines = [
            f"ROUGE-1 {metrics.get('rouge1_f', 0.0):.4f}, ROUGE-L {metrics.get('rouge_l_f', 0.0):.4f}, "
            f"couverture des faits {metrics.get('fact_coverage', 0.0):.4f} sur "
            f"{int(metrics.get('n_documents', 0))} document(s) de test.",
            f"{metrics.get('unsupported_share', 0.0):.1%} des résumés contiennent au moins une valeur "
            "absente du document : la détection d'hallucination s'arrête aux valeurs vérifiables.",
        ]
        for name, block in sorted(baselines.items()):
            lines.append(
                f"Référence `{name}` sur les mêmes lignes : ROUGE-1 {block.get('rouge1_f', 0.0):.4f}, "
                f"couverture {block.get('fact_coverage', 0.0):.4f}."
            )
        detail = dict(getattr(result, "verdict_detail", {}) or {})
        if detail.get("threshold") is not None:
            lines.append(
                f"Verdict {getattr(result, 'verdict', 'indéterminé')} : "
                f"{detail.get('metric')} = {detail.get('observed')} contre un seuil de "
                f"{detail.get('threshold')} (marge {detail.get('margin')})."
            )
        return lines


def report_path(paths: object) -> Path:
    """Return the expected path of the evaluation report.

    Args:
        paths: Project filesystem layout (a :class:`~src.utils.paths.ProjectPaths`).

    Returns:
        The report path inside ``artifacts/reports``.
    """
    return Path(str(getattr(paths, "reports_dir"))) / "evaluation_report.md"


__all__ = ["EvaluationPipeline", "report_path"]
