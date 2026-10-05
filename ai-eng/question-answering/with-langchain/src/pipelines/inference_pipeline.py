"""Inference pipeline: answer questions with a trained artefact.

``mode=predict`` (and ``make predict``) is the serving path. It reloads the artefact, reads the
questions either from a user-provided file (``predict.input``) or from a sample of the annotated
test questions (which keeps the demo runnable without any external input), answers them, and
writes ``artifacts/reports/predictions.csv``.

The predictions are validated against :class:`~src.data.schemas.PredictedAnswersSchema` *before*
being written: a malformed answer (negative latency, unknown abstention flag) must fail here, not
in the notebook that reads the file three days later. The latency measured here is the *warm*
latency of the serving path — the artefact is already loaded — and the report states that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import TextCorpusLoader
from src.data.schemas import validate_predictions
from src.inference.predictor import Predictor
from src.models import load_model
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline(BasePipeline):
    """Reload the artefact, answer questions, write the predictions."""

    name = "predict"

    def _execute(self) -> PipelineResult:
        """Run the inference flow.

        Returns:
            The pipeline result, whose payload is the predictions frame.

        Raises:
            FileNotFoundError: When the model artefact is missing.
            ValueError: When the requested sample is empty.
        """
        self.paths.ensure()
        loader = TextCorpusLoader(self.paths, formats=self.config.data.formats)
        model = self._load_model()
        questions = self._questions(loader)
        if questions.empty:
            msg = "No question to answer: check predict.input or the generated dataset"
            raise ValueError(msg)

        predictor = Predictor(
            model,
            config=self.config.model_dump(),
            paths=self.paths,
            k=int(node(self.config, "predict").get("answer_k", 5)),
        )
        predictions = predictor.predict(questions)
        validated = validate_predictions(predictions)
        output = Path(self.config.predict.output)
        if not output.is_absolute():
            output = self.paths.root / output
        written = write_table(validated, output)

        metrics = self._metrics(validated)
        logger.info(
            "Inference done | {} questions | abstention={:.1%} | p95={:.1f} ms",
            len(validated),
            metrics["abstention_rate"],
            metrics["latency_p95_ms"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=[str(written), *predictor.written_artifacts],
            payload=validated,
            messages=[
                f"{len(validated)} réponses écrites dans {written.name}.",
                f"Taux d'abstention {metrics['abstention_rate']:.1%}, "
                f"latence p95 {metrics['latency_p95_ms']:.1f} ms, "
                f"{metrics['mean_citations']:.1f} citation(s) par réponse.",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _load_model(self) -> object:
        """Reload the model artefact declared by the stack."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())

    def _questions(self, loader: TextCorpusLoader) -> pd.DataFrame:
        """Return the questions to answer (user file or a sample of the test split)."""
        configured = node(self.config, "predict").get("input")
        if configured:
            return loader.load_inference_questions(configured)
        for split in ("test", "val"):
            try:
                sample = loader.load_queries(split)
            except (ValueError, FileNotFoundError):
                continue
            return sample.head(int(node(self.config, "predict").get("n_samples", 5))).loc[
                :, ["query_id", "question"]
            ]
        msg = "No question source available: generate the dataset first (`make data`)"
        raise FileNotFoundError(msg)

    @staticmethod
    def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
        """Summarise an inference run.

        Args:
            predictions: Validated predictions.

        Returns:
            Finite metrics describing the run.
        """
        latency = predictions["latency_ms"]
        return {
            "n_predictions": float(len(predictions)),
            "abstention_rate": float(predictions["abstained"].mean()),
            "mean_citations": float(predictions["n_citations"].mean()),
            "mean_top_score": float(predictions["top_score"].mean()),
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(latency.quantile(0.50)),
            "latency_p95_ms": float(latency.quantile(0.95)),
            "n_chunks_indexed": float(predictions["n_chunks_indexed"].iloc[0]),
        }


__all__ = ["InferencePipeline"]
