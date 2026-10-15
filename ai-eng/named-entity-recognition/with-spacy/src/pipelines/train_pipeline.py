"""Entraînement : charger les deux tables, ajuster, monitorer, persister.

Séquence exécutée par ``mode=train`` (et ``make train``) :

1. charger les messages et leurs annotations, et valider la jointure entre les deux,
2. garder les messages ``train`` pour l'ajustement et ``val`` pour le suivi — le split de test
n'est
   jamais touché ici,
3. construire l'extracteur déclaré par la configuration (fabrique de la stack),
4. l'ajuster sur les annotations du train : la couche de règles apprend les surfaces du **train**
   uniquement, et le tagger apprend la forme des mentions sur les mêmes lignes,
5. surveiller le split de validation au niveau **entité** avec les callbacks partagés (historique,
   seuil, arrêt anticipé),
6. persister l'artefact (un répertoire : pipeline spaCy, index de règles appris, manifeste), les
   métriques, la fiche de modèle et la configuration résolue.

L'artefact mérite une phrase : un pipeline spaCy s'écrit avec ``nlp.to_disk``, et l'index de
règles
appris sur le train vit à côté de lui. Un **fichier unique** ne saurait pas porter les trois
choses
sans les confondre, et la fiche de modèle publie donc un répertoire, pas un fichier.

Persister la configuration résolue à côté de l'artefact est ce qui rend un run reproductible :
l'artefact seul ne dit pas quelle surcharge l'a produit.
"""

from __future__ import annotations

from typing import Any

from src.data.loaders import EntityCorpusLoader
from src.models import build_model
from src.models.contract import BaseEntityTagger
from src.pipelines.base import BasePipeline, PipelineResult
from src.training.callbacks import (
    BaseCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    MetricThresholdCallback,
    ProgressBarCallback,
)
from src.training.trainer import EntityTrainer
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainPipeline(BasePipeline):
    """Charger -> ajuster -> monitorer au niveau entité -> persister."""

    name = "train"

    def _execute(self) -> PipelineResult:
        """Run the whole training flow.

        Returns:
            The pipeline result, whose payload is the :class:`EntityTrainingOutcome`.
        """
        self.paths.ensure()
        loader = EntityCorpusLoader(
            self.paths,
            dataset_name=str(self.config.data.dataset_name),
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        documents, spans = loader.load_corpus()
        model_node = node(self.config, "model")
        trainer = EntityTrainer(
            self._build_model(),
            config=self.config.train.model_dump(),
            text_column=str(model_node.get("text_column", "text")),
            target_column=str(model_node.get("target", "label")),
            id_column=str(self.config.data.id_column or "msg_id"),
            split_column=str(self.config.data.group_column or "split"),
            metric_names=self.config.metrics.all_metrics,
            primary_metric=self.config.metrics.primary,
            min_primary_metric=self.config.metrics.min_primary,
            callbacks=self._callbacks(),
        )
        outcome = trainer.run(documents, spans)

        model = trainer.model
        artefact = model.save(self.paths.models_dir / self._model_file())
        metrics_path = write_json(
            self.paths.metrics_dir / "training_metrics.json", outcome.to_dict()
        )
        card = model.model_card(
            metrics=outcome.metrics,
            artifact=str(artefact.name),
            notes=self._notes(outcome),
        )
        card_path = card.to_json(self.paths.models_dir / "model_card.json")
        config_path = write_json(
            self.paths.models_dir / "resolved_config.json", self.config.model_dump()
        )

        artifacts = [str(artefact), str(metrics_path), str(card_path), str(config_path)]
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
                f"Modèle ajusté sur {outcome.n_train_documents} messages "
                f"({outcome.n_train_entities} entités) en "
                f"{outcome.fit_result.duration_seconds:.3f}s.",
                f"Validation sur {outcome.n_val_documents} messages "
                f"(métrique principale : {self.config.metrics.primary}="
                f"{_format(outcome.metrics.get(f'val_{self.config.metrics.primary}'))}).",
                f"Artefact écrit dans {artefact.name}/ : pipeline spaCy, index de règles "
                f"({outcome.fit_result.metrics.get('gazetteer_size', 0.0):.0f} surfaces apprises "
                "sur le train) et manifeste.",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _build_model(self) -> BaseEntityTagger:
        """Build the extractor declared by the configuration."""
        return build_model(self.config)

    def _model_file(self) -> str:
        """Resolve the artefact name declared for the stack (a directory name)."""
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
        alignment = outcome.fit_result.extra.get("alignment", {})
        return [
            "Le split de test n'est jamais utilisé pendant l'entraînement : il est mesuré une "
            "seule fois par le pipeline d'évaluation.",
            "La couche de règles apprend ses surfaces dans le **train** uniquement : une surface "
            "réservée aux splits d'évaluation lui est donc invisible, et la dégradation observée "
            "sur le test est un résultat, pas un accident.",
            (
                "Toutes les annotations tombent sur des frontières de tokens du tokenizer spaCy "
                f"({alignment.get('aligned_rate', 0.0):.1%} d'annotations apprenables, "
                f"{alignment.get('ignored', 0.0):.0f} ignorée(s)) : un corpus décalé produirait un "
                "score moyen sans que personne ne sache pourquoi."
            ),
            "La confiance publiée avec chaque mention est un niveau de corroboration par la couche "
            "de règles, pas une probabilité : sa précision est mesurée par niveau dans le "
            "rapport d'évaluation.",
            "Le corpus est synthétique : aucune donnée personnelle, aucun accès réseau, aucun "
            "modèle pré-entraîné téléchargé.",
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
