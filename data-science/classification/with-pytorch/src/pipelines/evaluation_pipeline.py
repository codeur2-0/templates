"""Evaluation pipeline: score a trained artefact and produce reports.

``mode=evaluate`` never retrains. It reloads:

* the test split persisted by the training pipeline (``data/processed/split_test.parquet``),
* the fitted preprocessing pipeline (``artifacts/models/preprocessing.joblib``),
* the trained model (``artifacts/models/<model_file>``),

then delegates to the family-specific :class:`src.evaluation.evaluator.Evaluator` and
:class:`src.evaluation.reports.ReportBuilder`. Every metric is written to
``artifacts/metrics/evaluation_metrics.json`` so that CI, dashboards or a model registry can
consume it without parsing logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.reports import ReportBuilder
from src.models import load_model
from src.pipelines.base import BasePipeline, PipelineResult
from src.preprocessing.pipelines import PreprocessingPipeline
from src.utils.io import read_table, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    """Reload artefacts, evaluate on the held-out test split, write reports."""

    name = "evaluate"

    def _execute(self) -> PipelineResult:
        """Run the evaluation flow.

        Returns:
            The pipeline result (payload = :class:`EvaluationResult`).

        Raises:
            FileNotFoundError: When the artefacts produced by ``mode=train`` are missing.
        """
        self.paths.ensure()
        model = self._load_model()
        preprocessing = self._load_preprocessing()
        test_features, test_frame = self._load_test_split()

        X_test = self._transform_test(preprocessing, test_frame, test_features)
        y_test = self._extract_target(test_frame)
        evaluator = Evaluator(
            model,
            metrics_config=self.config.metrics.model_dump(),
            paths=self.paths,
            task=self.config.metrics.task,
        )
        evaluation: EvaluationResult = evaluator.evaluate(
            X_test, y_test, split="test", context=test_frame
        )
        # `threshold_analysis` n'existe que sur les évaluateurs supervisés binaires : le
        # clustering et la régression n'ont pas de seuil de décision à analyser.
        analyse_thresholds = getattr(evaluator, "threshold_analysis", None)
        thresholds = (
            analyse_thresholds(X_test, y_test)
            if analyse_thresholds is not None
            and y_test is not None
            and self.config.metrics.task == "binary"
            else None
        )

        metrics_path = write_json(
            self.paths.metrics_dir / "evaluation_metrics.json",
            {
                "task": self.config.metrics.task,
                "primary_metric": self.config.metrics.primary,
                "metrics": _sanitise(evaluation.metrics),
                "n_samples": evaluation.n_samples,
                "model": model.summary(),
                "artifacts": {
                    "model": str(self._model_path()),
                    "preprocessing": str(self._preprocessing_path()),
                },
                "extras": _sanitise_extras(evaluation.extras),
            },
        )

        report_builder = ReportBuilder(self.paths, config=self.config.model_dump())
        report_files = report_builder.build(evaluation, model=model, thresholds=thresholds)

        result = PipelineResult(name=self.name)
        result.metrics = {key: float(value) for key, value in evaluation.metrics.items()}
        result.payload = evaluation
        result.add_artifact(metrics_path)
        for path in report_files.values():
            result.add_artifact(path)
        result.messages.append(
            "Evaluation | "
            + ", ".join(f"{key}={value:.5f}" for key, value in list(evaluation.metrics.items())[:6])
        )
        result.messages.append(f"Report: {report_files.get('report', 'n/a')}")
        return result

    def _transform_test(
        self,
        preprocessing: PreprocessingPipeline,
        test_frame: pd.DataFrame | None,
        test_features: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Rebuild the test matrix with the persisted preprocessing.

        Args:
            preprocessing: Fitted pipeline loaded from the artefacts.
            test_frame: Enriched raw test split (preferred: identical to training flow).
            test_features: Already processed matrix (fallback).

        Returns:
            The model-ready test matrix.
        """
        if test_frame is not None:
            excluded = set(self.config.data.drop_columns)
            target = self.config.data.target
            if target:
                excluded.add(str(target))
            columns = [column for column in test_frame.columns if column not in excluded]
            return preprocessing.transform(test_frame.loc[:, columns])
        if test_features is not None:
            return test_features
        msg = "Aucun split de test exploitable : relancez `python scripts/train.py`"
        raise FileNotFoundError(msg)

    def _extract_target(self, test_frame: pd.DataFrame | None) -> pd.Series | None:
        """Extract the ground truth from the enriched test split."""
        target = self.config.data.target
        if test_frame is None or not target or target not in test_frame.columns:
            return None
        return test_frame[target]

    # ------------------------------------------------------------------ artefacts -------
    def _model_path(self) -> Path:
        """Path of the trained model artefact."""
        return self.paths.models_dir / str(self.config.train.artifacts.model_file)

    def _preprocessing_path(self) -> Path:
        """Path of the fitted preprocessing artefact."""
        return self.paths.models_dir / str(self.config.train.artifacts.pipeline_file)

    def _load_model(self) -> Any:
        """Load the trained model, with an explicit hint when it is missing."""
        path = self._model_path()
        if not path.exists():
            msg = (
                f"No trained model at {path}. Run the training first: "
                "`python scripts/train.py` (or `make train`)."
            )
            raise FileNotFoundError(msg)
        return load_model(path, config=self.config)

    def _load_preprocessing(self) -> PreprocessingPipeline:
        """Load the fitted preprocessing pipeline."""
        path = self._preprocessing_path()
        if not path.exists():
            msg = (
                f"No preprocessing artefact at {path}. Run the training first: "
                "`python scripts/train.py` (or `make train`)."
            )
            raise FileNotFoundError(msg)
        return PreprocessingPipeline.load(path)

    def _load_test_split(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Load the processed test matrix and the enriched raw test split.

        Returns:
            ``(processed_features_or_None, enriched_raw_or_None)``.
        """
        features_path = self.paths.processed_dir / "features_test.parquet"
        split_path = self.paths.processed_dir / "split_test.parquet"
        features = read_table(features_path) if features_path.exists() else None
        split = read_table(split_path) if split_path.exists() else None
        if features is None and split is None:
            msg = (
                f"No test split found in {self.paths.processed_dir}. Run `python scripts/train.py` "
                "first: the training pipeline persists the splits."
            )
            raise FileNotFoundError(msg)
        rows = len(split) if split is not None else (0 if features is None else len(features))
        logger.info("Test split loaded | rows={}", rows)
        return features, split


def _sanitise(metrics: dict[str, Any]) -> dict[str, float]:
    """Coerce metrics to floats."""
    return {
        str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }


def _sanitise_extras(extras: dict[str, Any]) -> dict[str, Any]:
    """Make evaluator extras JSON serialisable (arrays -> lists, NaN -> None)."""
    payload: dict[str, Any] = {}
    for key, value in extras.items():
        if isinstance(value, pd.DataFrame):
            payload[key] = {"shape": list(value.shape), "columns": list(value.columns)}
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
        elif hasattr(value, "tolist"):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload
