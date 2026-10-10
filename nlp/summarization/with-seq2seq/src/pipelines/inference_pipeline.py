"""Inférence : résumer un fichier fourni ou un échantillon du corpus, puis publier.

``mode=predict`` recharge l'artefact entraîné (jamais un modèle vide), prépare l'entrée, résume et
écrit deux artefacts :

* ``artifacts/reports/predictions.csv`` — la table publiée : identifiant, stratégie, résumé produit,
  longueurs, latence et, quand la vérité terrain est connue, la couverture des faits saillants ;
* ``artifacts/reports/inference_summary.json`` — la lecture rapide : longueur moyenne, compression,
  part de résumés qui atteignent leur budget.

Le pipeline refuse de tourner sans artefact : un modèle non entraîné produirait
des résumés vides, et un tableau vide ne se distingue pas d'un tableau juste.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import SummaryCorpusLoader
from src.inference.predictor import SummaryPredictor
from src.models import load_model
from src.models.contract import BaseTextGenerator
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class InferencePipeline(BasePipeline):
    """Recharger l'artefact, résumer l'entrée, publier la table et sa lecture rapide."""

    name = "predict"

    def _execute(self) -> PipelineResult:
        """Run the inference flow.

        Returns:
            The pipeline result, whose payload is the prediction table.

        Raises:
            FileNotFoundError: When the artefact produced by ``mode=train`` is missing.
        """
        self.paths.ensure()
        model = self._load_model()
        model_node = node(self.config, "model")
        predictor = SummaryPredictor(
            model,
            paths=self.paths,
            config=self.config.model_dump(),
            text_column=str(model_node.get("text_column", "text")),
            id_column=str(model_node.get("id_column", "doc_id")),
        )
        frame = predictor.load_input(
            self.config.predict.input, n_samples=int(self.config.predict.n_samples)
        )
        predictions = predictor.predict(frame)
        path = predictor.save(predictions, path=self._output_path())
        summary = predictor.summary_frame(predictions)
        logger.info("Prédictions écrites : {}", path)
        return PipelineResult(
            name=self.name,
            metrics=self._metrics(predictions),
            artifacts=[str(path)],
            payload=predictions,
            messages=self._notes(predictions, summary),
        )

    def _output_path(self) -> Path:
        """Resolve the published prediction table against the project layout.

        ``predict.output`` est écrit relativement dans la configuration (``artifacts/reports/…``) :
        le résoudre contre la racine du projet, et non contre le répertoire courant, est ce qui
        permet d'exécuter ce pipeline depuis un bac à sable (notebooks, tests) sans écrire dans les
        artefacts publiés du dépôt.

        Returns:
            The absolute destination of the prediction table.
        """
        output = Path(str(self.config.predict.output))
        return output if output.is_absolute() else self.paths.root / output

    def _load_model(self) -> BaseTextGenerator:
        """Reload the artefact written by the training pipeline.

        Returns:
            The fitted strategy.

        Raises:
            FileNotFoundError: When the artefact is missing.
        """
        target = self.paths.models_dir / str(self.config.train.artifacts.model_file)
        if not target.is_file():
            msg = (
                f"Artefact de modèle introuvable : {target}. Exécuter `make train` "
                "(ou `python -m src.main mode=train`) avant de prédire."
            )
            raise FileNotFoundError(msg)
        model = load_model(target, config=self.config.model_dump())
        logger.info("Artefact rechargé : {} ({})", target, model.summary())
        return model

    @staticmethod
    def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
        """Aggregate the prediction table into flat metrics.

        Args:
            predictions: Prediction table.

        Returns:
            Documents, mean words, mean compression, coverage and truncated share.
        """
        frame = predictions
        if getattr(frame, "empty", True):
            return {"n_documents": 0.0}
        return {
            "n_documents": float(len(frame)),
            "n_words_mean": round(float(frame["n_words"].mean()), 2),
            "compression_mean": round(float(frame["compression"].mean()), 4),
            "fact_coverage": round(float(frame["fact_coverage"].mean()), 4),
            "unsupported_facts_mean": round(float(frame["unsupported_facts"].mean()), 3),
            "hit_max_length_share": round(float(frame["hit_max_length"].mean()), 4),
            "latency_p50_ms": round(float(frame["latency_ms"].median()), 3),
        }

    @staticmethod
    def _notes(predictions: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
        """Human-readable lines published with the inference.

        Args:
            predictions: Prediction table.
            summary: Aggregated reading of that table.

        Returns:
            The summary lines, one per strategy when several were measured.
        """
        if getattr(predictions, "empty", True):
            return ["Aucun document à résumer : l'entrée est vide."]
        lines = [
            f"{len(predictions)} document(s) résumé(s), "
            f"{float(predictions['n_words'].mean()):.1f} mots "
            f"produits en moyenne ({float(predictions['compression'].mean()):.1%} du document)."
        ]
        if getattr(summary, "empty", True) is False:
            for row in summary.itertuples(index=False):
                lines.append(
                    f"Stratégie `{row.strategy}` : couverture des faits "
                    f"{getattr(row, 'fact_coverage', 0.0):.4f}, "
                    f"{getattr(row, 'hit_max_length_share', 0.0):.1%} des résumés atteignent leur "
                    "budget de longueur."
                )
        lines.append(
            "La couverture n'est mesurée que lorsque le document appartient au corpus annoté ; sur "
            "un fichier externe, elle est publiée à 0,0 et doit être lue comme non mesurée."
        )
        return lines


def predictions_path(paths: ProjectPaths) -> Path:
    """Return the expected path of the published predictions.

    Args:
        paths: Project filesystem layout.

    Returns:
        The CSV path inside ``artifacts/reports``.
    """
    return paths.reports_dir / "predictions.csv"


def loader_for(paths: ProjectPaths, dataset_name: str) -> SummaryCorpusLoader:
    """Build the corpus loader of a project layout (used by the scripts and the tests).

    Args:
        paths: Project filesystem layout.
        dataset_name: Stem of the document table on disk.

    Returns:
        The configured loader.
    """
    return SummaryCorpusLoader(paths, dataset_name=str(dataset_name))


__all__ = ["InferencePipeline", "loader_for", "predictions_path"]
