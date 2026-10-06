"""Entraînement : vectoriser, ajuster, monitorer, persister.

Séquence exécutée par ``mode=train`` (et ``make train``) :

1. charger le corpus étiqueté et le valider,
2. garder les lignes ``train`` pour l'ajustement et ``val`` pour le suivi — le split de test n'est
   jamais touché ici,
3. construire le modèle déclaré par la configuration (fabrique de la stack),
4. l'ajuster : la vectorisation et l'estimateur sont internes au modèle, qui possède donc la
   représentation dont il aura besoin à l'inférence,
5. surveiller le split de validation avec les callbacks partagés (historique, seuil,
   arrêt anticipé),
6. persister le modèle, la configuration de prétraitement, les métriques, la fiche de modèle et la
   configuration résolue.

Persister la configuration résolue à côté de l'artefact est ce qui rend un run reproductible :
l'artefact seul ne dit pas quelle surcharge l'a produit.
"""

from __future__ import annotations

from typing import Any

from src.data.loaders import TextLabelLoader
from src.models import build_model
from src.models.contract import BaseTextClassifier
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
from src.training.trainer import ClassificationTrainer
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainPipeline(BasePipeline):
    """Charger -> vectoriser -> ajuster -> monitorer -> persister."""

    name = "train"

    def _execute(self) -> PipelineResult:
        """Run the whole training flow.

        Returns:
            The pipeline result, whose payload is the :class:`ClassificationOutcome`.
        """
        self.paths.ensure()
        loader = TextLabelLoader(
            self.paths,
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        documents = loader.load_documents()
        trainer = ClassificationTrainer(
            self._build_model(),
            config=self.config.train.model_dump(),
            text_column=str(node(self.config, "model").get("text_column", "text")),
            target_column=str(self.config.data.target or "label"),
            split_column="split",
            metric_names=self.config.metrics.all_metrics,
            primary_metric=self.config.metrics.primary,
            min_primary_metric=self.config.metrics.min_primary,
            callbacks=self._callbacks(),
        )
        outcome = trainer.run(documents)

        model = trainer.model
        model_path = model.save(self.paths.models_dir / self._model_file())
        preprocessor = TextPreprocessor.from_config(node(self.config, "preprocessing"))
        preprocessor_path = preprocessor.save(self.paths.models_dir / "preprocessing.joblib")
        metrics_path = write_json(
            self.paths.metrics_dir / "training_metrics.json", outcome.to_dict()
        )
        card = model.model_card(
            metrics=outcome.metrics,
            artifact=model_path.name,
            notes=self._notes(outcome),
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
        ]
        logger.info(
            "Training done | {} | val {}={}",
            model.summary(),
            self.config.metrics.primary,
            _format(outcome.metrics.get(f"val_{self.config.metrics.primary}")),
        )
        return PipelineResult(
            name=self.name,
            metrics=dict(outcome.metrics),
            artifacts=artifacts,
            payload=outcome,
            messages=[
                f"Modèle ajusté sur {outcome.n_train_documents} tickets "
                f"({len(model.labels)} classes) en "
                f"{outcome.fit_result.duration_seconds:.3f}s.",
                f"Validation sur {outcome.n_val_documents} tickets "
                f"(métrique principale : {self.config.metrics.primary}="
                f"{_format(outcome.metrics.get(f'val_{self.config.metrics.primary}'))}).",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _build_model(self) -> BaseTextClassifier:
        """Build the model declared by the configuration."""
        return build_model(self.config)

    def _model_file(self) -> str:
        """Resolve the artefact file name declared for the stack."""
        return str(self.config.train.artifacts.model_file)

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

    def _notes(self, outcome: Any) -> list[str]:
        """Return the traceability notes archived in the model card."""
        return [
            "Le split de test n'est jamais utilisé pendant l'entraînement : il est mesuré une "
            "seule fois par le pipeline d'évaluation.",
            "Le vocabulaire et l'estimateur sont appris sur le split d'entraînement uniquement : "
            "un terme absent du train n'a pas de poids, et le modèle le publie au lieu de "
            "l'ignorer.",
            "La colonne `priority` est un distracteur déclaré du corpus : elle est indépendante "
            "du libellé, et la suite de tests le vérifie.",
            "Le corpus est synthétique : aucune donnée personnelle, aucune dépendance réseau.",
        ]


def _maximisation_mode(direction: str) -> str:
    """Translate the contractual direction of a metric into a callback mode.

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
