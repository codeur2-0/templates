"""Entraînement : charger les trois tables, ajuster, comparer, persister.

Séquence exécutée par ``mode=train`` (et ``make train``) :

1. charger les documents, leurs résumés de référence et leurs faits, et valider les liens entre les
   trois tables,
2. garder les documents ``train`` pour l'ajustement et ``val``
   pour le suivi — le split de test n'est jamais touché ici,
3. construire la stratégie servie **et** les stratégies de
   comparaison déclarées par la configuration (fabrique de la stack),
4. ajuster la stratégie servie, surveiller la validation (ROUGE-1 et couverture), comparer les
   algorithmes sur un échantillon déterministe de la validation,
5. persister l'artefact, sa fiche de modèle, les métriques d'entraînement et la configuration
   résolue.

Les avertissements du trainer sont publiés tels quels : si la baseline extractive
fait mieux que le modèle servi sur l'échantillon de comparaison, le rapport et les
métriques d'entraînement le disent — un résultat honnête vaut mieux qu'un silence.
"""

from __future__ import annotations

from typing import Any

from src.data.loaders import SummaryCorpusLoader
from src.models import build_model
from src.models.contract import BaseTextGenerator
from src.pipelines.base import BasePipeline, PipelineResult
from src.training.callbacks import (
    BaseCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    ProgressBarCallback,
)
from src.training.trainer import SummaryTrainer, default_model_file
from src.utils.config_access import node
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainPipeline(BasePipeline):
    """Charger -> ajuster -> comparer -> persister."""

    name = "train"

    def _execute(self) -> PipelineResult:
        """Run the whole training flow.

        Returns:
            The pipeline result, whose payload is the :class:`SummaryTrainingOutcome`.
        """
        self.paths.ensure()
        loader = SummaryCorpusLoader(
            self.paths,
            dataset_name=str(self.config.data.dataset_name),
            formats=self.config.data.formats,
            validation_enabled=self.config.data.validation.raw,
            lazy_validation=self.config.data.validation.lazy,
        )
        documents, references, facts = loader.load_corpus()
        model_node = node(self.config, "model")
        trainer = SummaryTrainer(
            build_model(self.config, algorithm=str(self.config.model.algorithm)),
            config=self.config.model_dump(),
            paths=self.paths,
            algorithm=str(self.config.model.algorithm),
            text_column=str(model_node.get("text_column", "text")),
            id_column=str(model_node.get("id_column", "doc_id")),
            compare=self._strategies_to_compare(),
            builders=self._builders(),
            baseline=self._baseline(),
            callbacks=self._callbacks(),
        )
        outcome = trainer.run(documents, references, facts)
        outcome = trainer.persist(
            outcome,
            model_file=str(self.config.train.artifacts.model_file),
            resolved_config=self.config.model_dump(),
        )
        metrics = {
            **{f"val_{name}": value for name, value in outcome.validation_metrics.items()},
            **{f"fit_{name}": value for name, value in outcome.fit_metrics.items()},
            "training_seconds": round(outcome.duration_seconds, 3),
            "n_comparison_algorithms": float(len(outcome.comparison)),
        }
        verdict, _ = trainer.verdict(outcome.validation_metrics)
        logger.info(
            "Entraînement terminé : algorithme '{}', ROUGE-1 de validation {}, verdict {}",
            outcome.algorithm,
            outcome.validation_metrics.get("rouge1_f", 0.0),
            verdict,
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=[
                str(path)
                for path in (
                    outcome.model_path,
                    outcome.model_card_path,
                    outcome.metrics_path,
                    outcome.config_path,
                )
                if path is not None
            ],
            payload=outcome,
            messages=self._notes(outcome),
        )

    def _strategies_to_compare(self) -> list[str]:
        """Algorithms the configuration asks to compare.

        Returns:
            The algorithm names (empty when the configuration compares nothing).
        """
        model_node = node(self.config, "model")
        declared = model_node.get("strategies_to_compare", []) or []
        return [str(value) for value in declared]

    def _builders(self) -> dict[str, BaseTextGenerator]:
        """Build the strategies that will be compared to the served one.

        Returns:
            Mapping ``algorithm -> strategy``, each built
            by the stack factory (never by the pipeline).
        """
        builders: dict[str, BaseTextGenerator] = {}
        for name in self._strategies_to_compare():
            if name == str(self.config.model.algorithm):
                continue
            try:
                builders[name] = build_model(self.config, algorithm=name)
            except (ValueError, KeyError) as exc:
                logger.warning("Algorithme de comparaison '{}' ignoré : {}", name, exc)
        return builders

    def _baseline(self) -> BaseTextGenerator | None:
        """Build the published reference strategy, whatever else is compared.

        Returns:
            The baseline declared by the metrics node (``lead`` by default), or ``None`` when the
            factory cannot build it — the run must not fail for a reference it merely publishes.
        """
        candidate = str(self.config.metrics.baseline or "")
        if not candidate or candidate in {"aucune", "none"}:
            return None
        try:
            return build_model(self.config, algorithm=candidate)
        except (ValueError, KeyError) as exc:
            logger.warning("Baseline '{}' indisponible : {}", candidate, exc)
            return None

    def _callbacks(self) -> list[BaseCallback]:
        """Build the callbacks declared by the ``train`` node of the configuration.

        Chaque callback est **déclaré** dans ``conf/train/default.yaml`` : l'historique des
        métriques, le journal des époques, l'arrêt anticipé (avec sa propre règle : ``monitor``,
        ``patience``, ``mode``) et la barre de progression. Le seuil de qualité contractuel n'est
        pas un callback : il est lu par le trainer (verdict du run) puis par l'évaluateur (verdict
        du rapport), donc le dupliquer ici donnerait deux lectures du même seuil, libres de
        diverger.

        Returns:
            The configured callbacks (history, logging, early stopping, progress bar).
        """
        callbacks_config = self.config.train.callbacks
        early = self.config.train.early_stopping
        callbacks: list[BaseCallback] = []
        if callbacks_config.metric_history:
            callbacks.append(MetricHistoryCallback())
        if callbacks_config.logging_every >= 1:
            callbacks.append(LoggingCallback(every=int(callbacks_config.logging_every)))
        if early.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=str(early.monitor),
                    patience=int(early.patience),
                    min_delta=float(early.min_delta),
                    mode=str(early.mode),
                )
            )
        if callbacks_config.progress_bar:
            callbacks.append(ProgressBarCallback(enabled=False))
        return callbacks

    def _notes(self, outcome: Any) -> list[str]:
        """Human-readable lines published with the run.

        Args:
            outcome: The training outcome.

        Returns:
            The summary lines, warnings included.
        """
        lines = [
            f"Algorithme servi : `{outcome.algorithm}` ({outcome.model.strategy}) — "
            f"{outcome.model.summary()}.",
        ]
        if outcome.validation_metrics:
            lines.append(
                f"Validation : ROUGE-1 {outcome.validation_metrics.get('rouge1_f', 0.0):.4f}, "
                f"couverture des faits {outcome.validation_metrics.get('fact_coverage', 0.0):.4f}, "
                f"{outcome.validation_metrics.get('unsupported_facts_mean', 0.0):.2f} "
                "valeur(s) non "
                "supportée(s) par résumé."
            )
        if not outcome.comparison.empty:
            best = outcome.comparison.iloc[0]
            lines.append(
                f"Comparaison des algorithmes sur {int(outcome.comparison['n_documents'].max())} "
                f"document(s) de validation : le meilleur ROUGE-1 est obtenu par "
                f"`{best['algorithm']}` ({float(best['rouge1_f']):.4f})."
            )
        for warning in outcome.warnings:
            lines.append(f"⚠️ {warning}")
        return lines


def model_file_name(config: Any) -> str:
    """Return the artefact file name declared by a composed configuration.

    Args:
        config: Composed application configuration.

    Returns:
        The file name inside ``artifacts/models``.
    """
    return default_model_file(config.model_dump())


__all__ = ["TrainPipeline", "model_file_name"]
