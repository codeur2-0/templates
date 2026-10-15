"""Évaluation : mesurer un artefact entraîné sur le split de test, une seule fois.

``mode=evaluate`` ne réentraîne rien et ne règle rien. Il recharge l'artefact écrit par
``mode=train`` (pipeline spaCy + index de règles appris), mesure le split de **test** au niveau
entité, le compare à deux références mesurées sur les mêmes lignes (le plancher trivial et la
couche
de règles) et délègue la lecture des chiffres au constructeur de rapport.

Deux artefacts sont toujours écrits :

* ``artifacts/metrics/evaluation_metrics.json`` — lisible par une machine, pour un portail de CI ;
* ``artifacts/reports/evaluation_report.md`` — lisible par un humain, avec le verdict contractuel,
la
  ventilation par type et par segment, la table de précision par niveau de confiance, la
  comparaison des surfaces réservées et les erreurs classées.

Le pipeline refuse de tourner sans artefact de modèle : évaluer un modèle vide produirait un
rapport
plausible sur rien.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders import EntityCorpusLoader
from src.evaluation.evaluator import EntityEvaluator
from src.evaluation.reports import ReportBuilder
from src.models import load_model
from src.models.contract import BaseEntityTagger
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
            The pipeline result, whose payload is the evaluation result.

        Raises:
            FileNotFoundError: When the artefact produced by ``mode=train`` is missing.
        """
        self.paths.ensure()
        loader = EntityCorpusLoader(
            self.paths,
            dataset_name=str(self.config.data.dataset_name),
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        model = self._load_model()
        documents, spans = loader.load_corpus()
        test_documents = loader.split("test", frame=documents)
        test_spans = loader.split_annotations("test", documents=documents, spans=spans)
        model_node = node(self.config, "model")

        evaluator = EntityEvaluator(
            model,
            config=self.config.model_dump(),
            paths=self.paths,
            text_column=str(model_node.get("text_column", "text")),
            target_column=str(model_node.get("target", "label")),
            id_column=str(self.config.data.id_column or "msg_id"),
            split_column=str(self.config.data.group_column or "split"),
            primary_metric=self.config.metrics.primary,
            min_primary_metric=self.config.metrics.min_primary,
        )
        result = evaluator.evaluate(test_documents, test_spans)

        metrics_path = write_json(
            self.paths.metrics_dir / "evaluation_metrics.json", result.to_dict()
        )
        bundle = ReportBuilder(self.config.model_dump(), paths=self.paths).build(
            result,
            model_summary={
                "name": model.name,
                "framework": model.framework,
                "algorithm": model.algorithm,
                "task": model.task,
                "labels": model.labels,
                "state": model.state,
            },
            metadata=loader.load_metadata(),
            documents=documents,
            spans=spans,
        )
        artifacts = [str(metrics_path), *[str(path) for path in bundle.artifacts]]
        logger.info(
            "Evaluation done | {} messages | {}={}",
            result.n_documents,
            self.config.metrics.primary,
            _format(result.metrics.get(self.config.metrics.primary)),
        )
        return PipelineResult(
            name=self.name,
            metrics=result.metrics,
            artifacts=artifacts,
            payload=result,
            messages=[
                bundle.summary,
                f"Métrique principale {self.config.metrics.primary}="
                f"{_format(result.metrics.get(self.config.metrics.primary))} "
                f"(seuil contractuel : {self.config.metrics.min_primary}).",
                str(result.verdict_detail.get("message", "")),
            ],
        )

    def _load_model(self) -> BaseEntityTagger:
        """Reload the model artefact declared by the stack (a directory)."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())


def _format(value: float | None) -> str:
    """Format a metric for the logs and the messages."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["EvaluationPipeline"]
