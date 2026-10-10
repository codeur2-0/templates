"""Évaluation : mesurer un artefact entraîné sur le split de test, une seule fois.

``mode=evaluate`` ne réentraîne rien et ne règle rien. Il recharge les artefacts écrits par
``mode=train`` (modèle, configuration de prétraitement), mesure le split de **test**, le compare à
deux références triviales (classe majoritaire, tirage stratifié) et délègue la lecture des chiffres
au constructeur de rapport.

Deux artefacts sont toujours écrits :

* ``artifacts/metrics/evaluation_metrics.json`` — lisible par une machine, pour un portail de CI ;
* ``artifacts/reports/evaluation_report.md`` — lisible par un humain, avec le verdict contractuel,
  la ventilation par classe et par segment, et les erreurs les plus confiantes.

Le pipeline refuse de tourner sans artefact de modèle : évaluer un modèle vide produirait un
rapport plausible sur rien.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders import TextLabelLoader
from src.evaluation.evaluator import ClassificationEvaluator
from src.evaluation.reports import ReportBuilder
from src.models import load_model
from src.models.contract import BaseTextClassifier
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    """Recharger les artefacts, mesurer le test, écrire le rapport et les figures."""

    name = "evaluate"

    def _execute(self) -> PipelineResult:
        """Run the evaluation flow.

        Returns:
            The pipeline result, whose payload is the evaluation result.

        Raises:
            FileNotFoundError: When the artefacts produced by ``mode=train`` are missing.
        """
        self.paths.ensure()
        loader = TextLabelLoader(
            self.paths,
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        model = self._load_model()
        documents = loader.load_documents()
        test = loader.split("test", frame=documents)

        evaluator = ClassificationEvaluator(
            model,
            config=self.config.model_dump(),
            metrics_config=self.config.metrics.model_dump(),
            paths=self.paths,
            text_column=str(node(self.config, "model").get("text_column", "text")),
            target_column=str(self.config.data.target or "label"),
        )
        result = evaluator.evaluate(test)

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
                "n_features": model.n_features,
                "classes": model.labels,
            },
            metadata=loader.load_metadata(),
            documents=documents,
        )
        artifacts = [str(metrics_path), *[str(path) for path in bundle.artifacts]]
        logger.info(
            "Evaluation done | {} tickets | {}={}",
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
            ],
        )

    def _load_model(self) -> BaseTextClassifier:
        """Reload the model artefact declared by the stack."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())


def _format(value: float | None) -> str:
    """Format a metric for the logs and the messages."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["EvaluationPipeline"]
