"""End-to-end training pipeline.

Sequence executed by ``mode=train`` (and by ``make train``):

1. load the raw dataset and validate it (``RawDataSchema``),
2. split rows into train / validation / test (chronological when a time column is declared),
3. build derived features (learned on the train split only),
4. fit the preprocessing pipeline on train, transform every split,
5. validate the model-ready matrices (``ProcessedDataSchema``),
6. persist the splits (Parquet) so that evaluation and inference reuse exactly the same rows,
7. build the model, train it with callbacks, compute validation metrics,
8. persist the model, the preprocessing pipeline, the model card, the metrics and the **resolved
   configuration** (``artifacts/models/resolved_config.json``), so that a trained artefact carries
   the exact Hydra overrides, seed and family-specific blocks that produced it.

The test split is deliberately **never** seen by the trainer: it is only used by
:class:`src.pipelines.evaluation_pipeline.EvaluationPipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import (
    DatasetBundle,
    DatasetSplitter,
    RawDataLoader,
    feature_target_split,
    save_split,
    target_distribution,
)
from src.data.schemas import ProcessedDataSchema
from src.features.build_features import (
    FeatureBuilder,
    select_feature_columns,
    split_by_dtype,
)
from src.models import build_model
from src.pipelines.base import BasePipeline, PipelineResult
from src.preprocessing.pipelines import PreprocessingPipeline
from src.training.losses_metrics import metric_extra_from_config
from src.training.trainer import Trainer, TrainingData
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainPipeline(BasePipeline):
    """Load -> validate -> split -> preprocess -> fit -> persist."""

    name = "train"

    def _execute(self) -> PipelineResult:
        """Run the whole training flow.

        Returns:
            The pipeline result, whose payload is the :class:`TrainingOutcome`.
        """
        config_dict = self.config.model_dump()
        self.paths.ensure()

        raw = self._load_raw()
        splits = self._split(raw)
        enriched, feature_builder = self._build_features(splits)
        matrices = self._preprocess(enriched)
        self._persist_splits(enriched, matrices)

        model = build_model(self.config, feature_names=matrices["feature_names"])
        trainer = Trainer(
            model,
            config=self.config.train.model_dump(),
            paths=self.paths,
            metric_names=self.config.metrics.all_metrics,
            task=self.config.metrics.task,
            metric_extra=self._metric_extra(config_dict),
            min_primary_metric=self.config.metrics.min_primary,
            primary_metric=f"val_{self.config.metrics.primary}",
            primary_direction=str(self.config.metrics.direction),
        )
        outcome = trainer.train(
            TrainingData(
                X_train=matrices["X_train"],
                y_train=matrices["y_train"],
                X_val=matrices["X_val"],
                y_val=matrices["y_val"],
                groups_val=matrices.get("groups_val"),
                feature_names=matrices["feature_names"],
                task=self.config.metrics.task,
            )
        )

        preprocessing_artifact = self._save_preprocessing(matrices["pipeline"])
        builder_artifact = self._save_feature_builder(feature_builder)
        config_artifact = self._save_config(config_dict)
        result = PipelineResult(name=self.name)
        result.metrics = {
            key: float(value)
            for key, value in outcome.metrics.items()
            if isinstance(value, (int, float))
        }
        result.payload = outcome
        for path in [
            *outcome.artifacts.values(),
            str(preprocessing_artifact),
            str(builder_artifact),
            str(config_artifact),
        ]:
            result.add_artifact(path)
        result.messages.append(
            "Model trained | "
            + ", ".join(f"{key}={value:.5f}" for key, value in list(result.metrics.items())[:5])
        )
        result.messages.append(f"Preprocessing saved to {preprocessing_artifact}")
        logger.debug("Train pipeline configuration: {}", _compact(config_dict))
        return result

    # ------------------------------------------------------------------ steps -----------
    def _load_raw(self) -> pd.DataFrame:
        """Load and validate the raw dataset."""
        data_config = self.config.data
        loader = RawDataLoader(
            self.paths,
            dataset_name=str(data_config.dataset_name),
            formats=tuple(data_config.formats),
            validate=bool(data_config.validation.raw),
            lazy_validation=bool(data_config.validation.lazy),
            strict=bool(data_config.validation.strict),
            limit=data_config.sampling.limit,
        )
        return loader.load()

    def _split(self, raw: pd.DataFrame) -> Any:
        """Split rows into train / validation / test."""
        splitter = DatasetSplitter.from_config(self.config.model_dump(), seed=self.config.seed)
        splits = splitter.split(raw, target=self.config.data.target)
        logger.info(
            "Splits | {} | target distribution (train)={}",
            splits.sizes,
            {
                key: round(value, 4)
                for key, value in target_distribution(
                    splits.train[self.config.data.target] if self.config.data.target else None
                ).items()
            },
        )
        return splits

    def _build_features(self, splits: Any) -> tuple[dict[str, pd.DataFrame], FeatureBuilder]:
        """Apply the declarative feature builder (fitted on train only).

        Returns:
            The enriched splits and the fitted builder (persisted for inference).
        """
        builder = FeatureBuilder.from_config(
            self.config.model_dump(), target=self.config.data.target
        )
        if builder.recipes:
            builder.fit(splits.train)
        enriched = {
            "train": builder.transform(splits.train),
            "val": None if splits.val is None else builder.transform(splits.val),
            "test": builder.transform(splits.test),
        }
        logger.info(
            "Feature engineering | recipes={} output_columns={}",
            len(builder.recipes),
            len(builder.output_names),
        )
        return enriched, builder

    @staticmethod
    def _metric_extra(config_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Return the metric context read from the root configuration.

        A ranking metric is meaningless without its cutoff: ``ndcg_at_k`` must know how many slots
        are published. That number lives in the task node of the root configuration, which the
        trainer does not receive, so the pipeline extracts it here rather than letting the metric
        fall back on a default.

        Args:
            config_dict: Root configuration, as a plain mapping.

        Returns:
            The metric context (empty when the task declares none).
        """
        return metric_extra_from_config(config_dict)

    def _preprocess(self, enriched: dict[str, Any]) -> dict[str, Any]:
        """Fit the preprocessing on train and transform every split."""
        target = self.config.data.target
        drop_columns = list(self.config.data.drop_columns)
        train_frame: pd.DataFrame = enriched["train"]

        feature_columns = select_feature_columns(
            train_frame, drop_columns=drop_columns, target=target
        )
        numeric_columns, categorical_columns = split_by_dtype(train_frame, feature_columns)
        explicit = self.config.preprocessing.model_dump().get("columns") or {}
        numeric_columns = list(explicit.get("numeric") or numeric_columns)
        categorical_columns = list(explicit.get("categorical") or categorical_columns)

        pipeline = PreprocessingPipeline(
            numeric_features=numeric_columns,
            categorical_features=categorical_columns,
            config=self.config.preprocessing.model_dump(),
            target=target,
        )

        X_train_frame, y_train = feature_target_split(train_frame, target, drop_columns)
        _, y_val = (
            feature_target_split(enriched["val"], target, drop_columns)
            if enriched["val"] is not None
            else (None, None)
        )
        X_train = pipeline.fit_transform(X_train_frame, y_train)
        X_val = (
            None
            if enriched["val"] is None
            else pipeline.transform(enriched["val"].loc[:, X_train_frame.columns])
        )
        X_test = pipeline.transform(enriched["test"].loc[:, X_train_frame.columns])

        # The group column (the user, in a ranking task) is dropped from the features by the
        # preprocessing, so it travels separately, aligned with ``X_val``, letting the validation
        # metrics be computed per group. ``transform`` preserves the row order, which is what
        # guarantees the alignment.
        group_column = str(getattr(self.config.data, "group_column", "") or "")
        val_frame = enriched["val"]
        groups_val = (
            val_frame[group_column].to_numpy()
            if val_frame is not None and group_column and group_column in val_frame.columns
            else None
        )

        if self.config.data.validation.processed:
            validated: list[str] = []
            for name, matrix in (("train", X_train), ("val", X_val), ("test", X_test)):
                if matrix is not None:
                    ProcessedDataSchema.validate(
                        matrix, lazy=bool(self.config.data.validation.lazy)
                    )
                    validated.append(name)
            logger.debug("Processed matrices validated against ProcessedDataSchema: {}", validated)

        return {
            "pipeline": pipeline,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": feature_target_split(enriched["test"], target, drop_columns)[1],
            "groups_val": groups_val,
            "feature_names": pipeline.feature_names_out,
        }

    def _persist_splits(self, enriched: dict[str, Any], matrices: dict[str, Any]) -> None:
        """Persist the enriched splits and the processed matrices."""
        for name in ("train", "val", "test"):
            frame = enriched.get(name)
            if frame is not None:
                save_split(frame, self.paths, f"split_{name}")
        for name in ("X_train", "X_val", "X_test"):
            matrix = matrices.get(name)
            if matrix is not None:
                save_split(matrix, self.paths, f"features_{name}")

    def _save_preprocessing(self, pipeline: PreprocessingPipeline) -> Any:
        """Persist the fitted preprocessing pipeline."""
        artifact_name = str(self.config.train.artifacts.pipeline_file)
        return pipeline.save(self.paths.models_dir / artifact_name)

    def _save_config(self, config_dict: dict[str, Any]) -> Path:
        """Persist the resolved configuration next to the model artefacts.

        A trained model is reproducible only if the configuration that produced it is stored with
        it: Hydra overrides, seeds and family-specific blocks are otherwise lost between two runs,
        and nobody can tell whether an artefact matches the code reading it. Downstream components
        (evaluator, predictor, reports) read this snapshot instead of re-composing Hydra, which
        keeps them usable outside ``src.main`` — in a notebook, a test or a batch job.

        Args:
            config_dict: Resolved configuration mapping (``AppConfig.model_dump``).

        Returns:
            The written path.
        """
        path = Path(self.paths.models_dir) / "resolved_config.json"
        write_json(path, config_dict)
        logger.info("Resolved configuration saved to {}", path)
        return path

    def _save_feature_builder(self, feature_builder: FeatureBuilder) -> Any:
        """Persist the fitted feature builder (inference must rebuild the same features)."""
        from src.utils.io import save_pickle

        artifact_name = str(self.config.train.artifacts.feature_builder_file)
        return save_pickle(feature_builder, self.paths.models_dir / artifact_name)


def build_dataset_bundle(matrices: dict[str, Any], task: str) -> DatasetBundle:
    """Assemble a :class:`DatasetBundle` from preprocessed matrices.

    Args:
        matrices: Mapping produced by :meth:`TrainPipeline._preprocess`.
        task: Learning task identifier.

    Returns:
        The bundle (used by notebooks and the evaluator).
    """
    return DatasetBundle(
        X_train=matrices["X_train"],
        y_train=matrices["y_train"],
        X_val=matrices.get("X_val"),
        y_val=matrices.get("y_val"),
        X_test=matrices["X_test"],
        y_test=matrices.get("y_test"),
        feature_names=list(matrices["feature_names"]),
        task=task,
    )


def _compact(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a trimmed configuration for debug logging."""
    return {key: config_dict[key] for key in ("mode", "seed") if key in config_dict}
