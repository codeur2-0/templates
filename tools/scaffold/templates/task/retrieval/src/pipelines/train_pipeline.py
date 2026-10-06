"""Training pipeline: chunk the corpus, build the index, persist the artefacts.

Sequence executed by ``mode=train`` (and ``make train``):

1. load the raw corpus and the annotated questions, validating both,
2. keep the **validation** split for monitoring — the test split is never touched here,
3. build the model declared by the configuration,
4. fit it: chunking, embedding, index construction happen inside the model, which owns the
   chunking parameters it will need again at inference time,
5. monitor the validation questions with the shared callbacks (early stopping, thresholds,
   metric history),
6. persist the passages (``data/processed/chunks.parquet``), the model artefact, the
   preprocessing configuration, the metrics, the model card and the resolved configuration.

Persisting the resolved configuration next to the artefact is what makes a run reproducible: the
artefact alone does not say which override produced it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.loaders import TextCorpusLoader
from src.models import build_model
from src.pipelines.base import BasePipeline, PipelineResult
from src.preprocessing.pipelines import TextPreprocessor
from src.training.callbacks import (
    BaseCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    MetricThresholdCallback,
    ProgressBarCallback,
)
from src.training.trainer import Trainer
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainPipeline(BasePipeline):
    """Load -> chunk -> index -> monitor -> persist."""

    name = "train"

    def _execute(self) -> PipelineResult:
        """Run the whole training flow.

        Returns:
            The pipeline result, whose payload is the :class:`TrainingOutcome`.
        """
        self.paths.ensure()
        loader = self._loader()
        documents = loader.load_documents()
        val_queries = self._validation_questions(loader)

        model = build_model(self.config)
        trainer = Trainer(
            model,
            config=self.config.train.model_dump(),
            paths=self.paths,
            metric_names=self.config.metrics.all_metrics,
            ks=tuple(node(self.config, "train").get("retrieval_ks", [1, 3, 5, 10])),
            min_primary_metric=self.config.metrics.min_primary,
            primary_metric=self.config.metrics.primary,
            answer_k=int(node(self.config, "predict").get("answer_k", 5)),
            chunks_per_document=int(node(self.config, "retrieval").get("chunks_per_document", 3)),
            callbacks=self._callbacks(),
        )
        calibration_queries = self._calibration_questions(loader)
        outcome = trainer.run(documents, val_queries, calibration=calibration_queries)

        chunks = model.chunks
        chunks_written = loader.save_chunks(chunks) if not chunks.empty else {}
        model_path = model.save(self.paths.models_dir / self._model_file())
        preprocessor = TextPreprocessor.from_config(node(self.config, "preprocessing"))
        preprocessor_path = preprocessor.save(self.paths.models_dir / "preprocessing.joblib")
        metrics_path = write_json(
            self.paths.metrics_dir / "training_metrics.json", outcome.to_dict()
        )
        card = model.model_card(
            metrics=outcome.metrics,
            artifact=model_path.name,
            notes=self._notes(),
        )
        card_path = card.to_json(self.paths.models_dir / "model_card.json")
        config_path = write_json(
            self.paths.models_dir / "resolved_config.json", self.config.model_dump()
        )

        artifacts = [
            str(model_path),
            str(preprocessor_path),
            str(metrics_path),
            str(card_path),
            str(config_path),
            *[str(path) for path in chunks_written.values()],
        ]
        logger.info(
            "Training done | {} | val {}={}",
            model.summary(),
            self.config.metrics.primary,
            _format(outcome.metrics.get(f"val_{self.config.metrics.primary}")),
        )
        return PipelineResult(
            name=self.name,
            # Le trainer nomme déjà ses métriques (``val_*`` pour la validation, ``index_*``
            # pour l'index) : les renommer ici transformerait un compteur d'index en métrique de
            # validation, et le rapport publierait deux fois la même grandeur.
            metrics=dict(outcome.metrics),
            artifacts=artifacts,
            payload=outcome,
            messages=[
                f"Index construit : {outcome.fit_result.n_chunks} passages issus de "
                f"{outcome.fit_result.n_documents} documents en "
                f"{outcome.fit_result.duration_seconds:.1f}s.",
                f"Validation sur {outcome.n_validation_questions} questions "
                f"(métrique principale : {self.config.metrics.primary}).",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _loader(self) -> TextCorpusLoader:
        """Build the loader from the configuration."""
        return TextCorpusLoader(
            self.paths,
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )

    def _validation_questions(self, loader: TextCorpusLoader) -> Any:
        """Load the validation questions, tolerating an absent split.

        Args:
            loader: Configured loader.

        Returns:
            The validation questions, or an empty frame when the split does not exist (a
            one-split corpus is a legitimate configuration for a quick experiment).
        """
        try:
            return loader.load_queries("val")
        except ValueError as error:
            logger.warning("No validation split available ({}): monitoring is disabled", error)
            return pd.DataFrame()

    def _calibration_questions(self, loader: TextCorpusLoader) -> Any:
        """Load the calibration questions used to tune the decision threshold.

        Args:
            loader: Configured loader.

        Returns:
            The calibration questions, or an empty frame when the split does not exist.
        """
        try:
            return loader.load_queries("calibration")
        except ValueError as error:
            logger.warning(
                "No calibration split available ({}): threshold kept as configured", error
            )
            return pd.DataFrame()

    def _callbacks(self) -> list[BaseCallback]:
        """Build the callbacks declared by the configuration."""
        train = self.config.train
        callbacks: list[BaseCallback] = []
        if train.callbacks.progress_bar:
            callbacks.append(ProgressBarCallback(enabled=False))
        if train.callbacks.logging_every:
            callbacks.append(LoggingCallback(every=int(train.callbacks.logging_every)))
        if train.callbacks.metric_history:
            callbacks.append(MetricHistoryCallback())
        early = train.early_stopping
        if early.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=str(early.monitor).removeprefix("val_"),
                    patience=int(early.patience),
                    min_delta=float(early.min_delta),
                    mode=str(early.mode),
                )
            )
        if self.config.metrics.min_primary is not None:
            callbacks.append(
                MetricThresholdCallback(
                    monitor=f"val_{self.config.metrics.primary}",
                    threshold=float(self.config.metrics.min_primary),
                    mode=_maximisation_mode(str(self.config.metrics.direction)),
                )
            )
        return callbacks

    def _model_file(self) -> str:
        """Resolve the artefact file name declared for the stack."""
        return str(self.config.train.artifacts.model_file)

    def _notes(self) -> list[str]:
        """Return the traceability notes archived in the model card."""
        return [
            "Le split de test n'est jamais utilisé pendant l'entraînement : il est mesuré une "
            "seule fois par le pipeline d'évaluation.",
            "Le découpage en passages est appris (configuré) ici et persisté avec le modèle : "
            "l'inférence réutilise exactement le même découpage.",
            "Le corpus est synthétique : aucune donnée personnelle, aucune dépendance réseau.",
        ]


def _maximisation_mode(direction: str) -> str:
    """Translate the contractual direction of a metric into a callback mode.

    ``conf/config.yaml`` states the direction in words (``maximize`` / ``minimize``) because it is
    the vocabulary used by the report; the callback compares in symbols (``max`` / ``min``). The
    translation lives here, in one place, instead of being duplicated in every stack: a mismatch
    silently inverts the threshold check and warns about metrics that are perfectly fine.

    Args:
        direction: ``maximize`` or ``minimize``.

    Returns:
        ``max`` or ``min``.
    """
    return "min" if direction.strip().lower().startswith("min") else "max"


def _format(value: float | None) -> str:
    """Format a metric for the logs."""
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["TrainPipeline"]
