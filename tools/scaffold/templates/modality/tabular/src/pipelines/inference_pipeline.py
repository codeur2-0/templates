"""Inference pipeline: score new records with the trained artefacts.

The input can be:

* a file (``predict.input=data/raw/dataset.csv``), validated with ``InferenceDataSchema``,
* or a synthetic sample generated on the fly (``predict.n_samples``), which makes the demo
  runnable immediately after training without preparing an input file.

Predictions are written to ``artifacts/reports/predictions.csv`` and summarised in the logs
and in the pipeline result (row count, score distribution, execution time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.predictor import Predictor
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.io import write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline(BasePipeline):
    """Validate, predict, persist and summarise predictions."""

    name = "predict"

    def _execute(self) -> PipelineResult:
        """Run the inference flow.

        Returns:
            The pipeline result (payload = predictions ``DataFrame``).
        """
        self.paths.ensure()
        predictor = Predictor.from_config(self.config, self.paths)
        inputs = self._load_inputs(predictor)

        predictions = predictor.predict(inputs)
        output_path = Path(self.config.predict.output)
        if not output_path.is_absolute():
            output_path = self.paths.root / output_path
        write_table(predictions, output_path)

        result = PipelineResult(name=self.name)
        result.add_artifact(output_path)
        result.payload = predictions
        result.metrics = _summarise(predictions)
        source = self._input_label()
        result.messages.append(
            f"{len(predictions)} prediction(s) written to {output_path} (from {source})"
        )
        logger.info("Inference done | {}", result.messages[-1])
        return result

    def _input_label(self) -> str:
        """Describe where the scored records came from."""
        source = self.config.predict.input
        return (
            f"file {source}"
            if source
            else f"synthetic sample ({self.config.predict.n_samples} rows)"
        )

    def _load_inputs(self, predictor: Predictor) -> pd.DataFrame:
        """Load the records to score: configured file or generated sample."""
        source = self.config.predict.input
        if source:
            return predictor.load_inputs(Path(source))
        return predictor.sample_inputs(int(self.config.predict.n_samples))


def _summarise(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute a compact, log-friendly summary of the predictions."""
    summary: dict[str, float] = {"n_predictions": float(len(predictions))}
    score_columns = [
        column
        for column in predictions.columns
        if column.endswith(("score", "probability", "proba", "prediction"))
        and pd.api.types.is_numeric_dtype(predictions[column])
    ]
    for column in score_columns[:4]:
        values = predictions[column].to_numpy(dtype="float64")
        summary[f"{column}_mean"] = float(np.nanmean(values)) if len(values) else float("nan")
        summary[f"{column}_min"] = float(np.nanmin(values)) if len(values) else float("nan")
        summary[f"{column}_max"] = float(np.nanmax(values)) if len(values) else float("nan")
    if "prediction" in predictions.columns:
        counts = predictions["prediction"].value_counts(normalize=True)
        for label, share in list(counts.items())[:4]:
            summary[f"share_{label}"] = float(share)
    return summary
