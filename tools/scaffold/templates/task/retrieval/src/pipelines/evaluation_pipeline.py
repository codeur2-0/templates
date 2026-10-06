"""Evaluation pipeline: measure a trained artefact on the held-out test questions.

``mode=evaluate`` never retrains and never re-tunes. It reloads the artefacts written by
``mode=train`` (model, preprocessing configuration, chunk table), measures the **test** split
exactly once, compares the model to trivial references (random ranking, first-passage ranking)
and delegates the reading of the numbers to the task-specific report builder.

Two artefacts are always written:

* ``artifacts/metrics/evaluation_metrics.json`` — machine readable, for CI gates and dashboards;
* ``artifacts/reports/evaluation_report.md`` — human readable, with the verdict on the
  contractual objectives and the figures referenced inline.

The pipeline refuses to run without a model artefact: silently evaluating an empty index would
produce a plausible-looking report about nothing.
"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders import TextCorpusLoader
from src.evaluation.evaluator import Evaluator
from src.evaluation.reports import ReportBuilder
from src.models import load_model
from src.models.base import BaseModel
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    """Reload the artefacts, measure the test split, write the report and the figures."""

    name = "evaluate"

    def _execute(self) -> PipelineResult:
        """Run the evaluation flow.

        Returns:
            The pipeline result, whose payload is the evaluation result.

        Raises:
            FileNotFoundError: When the artefacts produced by ``mode=train`` are missing.
        """
        self.paths.ensure()
        loader = TextCorpusLoader(
            self.paths,
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        model = self._load_model()
        documents = loader.load_documents()
        test_queries = loader.load_queries("test")

        evaluator = Evaluator(
            model,
            config=self.config.model_dump(),
            metrics_config=self.config.metrics.model_dump(),
            paths=self.paths,
            ks=tuple(node(self.config, "train").get("retrieval_ks", [1, 3, 5, 10])),
            answer_k=int(node(self.config, "predict").get("answer_k", 5)),
            chunks_per_document=int(node(self.config, "retrieval").get("chunks_per_document", 3)),
        )
        result = evaluator.evaluate(test_queries, documents)

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
                "n_chunks": len(model.chunks),
                "llm": model.llm.describe(),
            },
            metadata=loader.load_metadata(),
        )
        artifacts = [str(metrics_path), *[str(path) for path in bundle.artifacts]]
        logger.info(
            "Evaluation done | {} questions | {}={}",
            result.n_questions,
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

    def _load_model(self) -> BaseModel:
        """Reload the model artefact declared by the stack."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())


def _format(value: float | None) -> str:
    """Format a metric for the logs and the messages."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["EvaluationPipeline"]
