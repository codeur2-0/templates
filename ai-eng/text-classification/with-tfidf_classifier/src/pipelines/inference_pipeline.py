"""Inférence : classer de nouveaux tickets avec un artefact entraîné.

``mode=predict`` (et ``make predict``) est le chemin de service. Il recharge l'artefact, lit les
tickets soit dans un fichier fourni par l'utilisateur (``predict.input``), soit dans un échantillon
du corpus annoté — ce qui garde la démonstration exécutable sans aucune entrée externe — et écrit
``artifacts/reports/predictions.csv``.

Les prédictions sont validées contre leur contrat Pandera **avant** d'être écrites : une confiance
négative ou une latence aberrante doit échouer ici, pas dans le notebook qui lira le fichier trois
jours plus tard. La latence mesurée est celle du chemin de service, texte par texte, artefact déjà
chargé ; le rapport le dit, parce qu'un chiffre de latence sans son protocole ne veut rien dire.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schemas import validate_predictions
from src.inference.predictor import TextClassificationPredictor
from src.models import load_model
from src.models.contract import BaseTextClassifier
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline(BasePipeline):
    """Recharger l'artefact, classer les tickets, écrire les prédictions."""

    name = "predict"

    def _execute(self) -> PipelineResult:
        """Run the inference flow.

        Returns:
            The pipeline result, whose payload is the validated prediction frame.

        Raises:
            FileNotFoundError: When the model artefact is missing.
            ValueError: When there is nothing to classify.
        """
        self.paths.ensure()
        model = self._load_model()
        predictor = TextClassificationPredictor(
            model,
            config=self.config.model_dump(),
            paths=self.paths,
            text_column=str(node(self.config, "model").get("text_column", "text")),
        )
        payload = predictor.load_input()
        predictions = predictor.predict(payload)
        validated = validate_predictions(predictions)
        output = Path(self.config.predict.output)
        if not output.is_absolute():
            output = self.paths.root / output
        written = write_table(validated, output)
        metrics = self._metrics(validated)
        logger.info(
            "Inference done | {} tickets | exactitude {:.1%} | p95 {:.1f} ms",
            len(validated),
            metrics["accuracy"],
            metrics["latency_p95_ms"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=[str(written)],
            payload=validated,
            messages=[
                f"{len(validated)} tickets classés dans {written.name}.",
                f"Exactitude {metrics['accuracy']:.1%} sur les tickets dont la référence est "
                f"connue, confiance moyenne {metrics['mean_confidence']:.1%}, latence p95 "
                f"{metrics['latency_p95_ms']:.1f} ms.",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _load_model(self) -> BaseTextClassifier:
        """Reload the model artefact declared by the stack."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())

    @staticmethod
    def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
        """Summarise an inference run.

        Args:
            predictions: Validated prediction frame.

        Returns:
            Finite metrics describing the run.
        """
        latency = predictions["latency_ms"]
        labelled = predictions[predictions["expected_label"].notna()]
        return {
            "n_predictions": float(len(predictions)),
            "n_labelled": float(len(labelled)),
            "accuracy": float(labelled["correct"].mean()) if len(labelled) else 0.0,
            "mean_confidence": float(predictions["confidence"].mean()),
            "share_high_confidence": float((predictions["confidence"] >= 0.8).mean()),
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(latency.quantile(0.50)),
            "latency_p95_ms": float(latency.quantile(0.95)),
        }


__all__ = ["InferencePipeline"]
