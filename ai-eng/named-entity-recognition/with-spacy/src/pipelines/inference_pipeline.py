"""Inférence : extraire les entités de nouveaux messages avec un artefact entraîné.

``mode=predict`` (et ``make predict``) est le chemin de service. Il recharge l'artefact, lit les
messages soit dans un fichier fourni par l'utilisateur (``predict.input``), soit dans un
échantillon
du corpus annoté — ce qui garde la démonstration exécutable sans aucune entrée externe — et écrit
``artifacts/reports/predictions.csv``.

Deux sorties, parce que deux lectures servent :

* la **table de mentions** (une ligne par entité) : identifiant, décalages, type, surface,
  provenance, confiance, et le verdict quand la référence est connue ;
* le **résumé par message** : nombre de mentions, latence du message.

Les prédictions sont validées contre leur contrat Pandera **avant** d'être écrites : une confiance
hors de [0 ; 1] ou un décalage négatif doit échouer ici, pas dans le notebook qui lira le fichier
trois jours plus tard. La latence mesurée est celle du chemin de service, message par message,
artefact déjà chargé ; le rapport le dit, parce qu'un chiffre de latence sans son protocole ne
veut
rien dire.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import EntityCorpusLoader
from src.data.schemas import validate_predictions
from src.inference.predictor import EntityPredictor
from src.models import load_model
from src.models.contract import BaseEntityTagger
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.config_access import node
from src.utils.io import write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline(BasePipeline):
    """Recharger l'artefact, extraire les entités, écrire les mentions et le résumé."""

    name = "predict"

    def _execute(self) -> PipelineResult:
        """Run the inference flow.

        Returns:
            The pipeline result, whose payload is the validated mention table.

        Raises:
            FileNotFoundError: When the model artefact is missing.
            ValueError: When there is nothing to annotate.
        """
        self.paths.ensure()
        model = self._load_model()
        predictor = EntityPredictor(
            model,
            config=self.config.model_dump(),
            paths=self.paths,
            text_column=str(node(self.config, "model").get("text_column", "text")),
            id_column=str(self.config.data.id_column or "msg_id"),
        )
        payload = predictor.load_input()
        spans = self._reference(payload)
        mentions = predictor.predict(payload, spans=spans)
        validated = validate_predictions(mentions)
        output = Path(self.config.predict.output)
        if not output.is_absolute():
            output = self.paths.root / output
        written = write_table(validated, output)
        summary_path = write_table(
            predictor.summary(payload), self.paths.reports_dir / "inference_summary.csv"
        )
        metrics = self._metrics(validated)
        logger.info(
            "Inference done | {} messages | {} mentions | p95 {:.1f} ms",
            len(payload),
            metrics["n_mentions"],
            metrics["latency_p95_ms"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=[str(written), str(summary_path)],
            payload=validated,
            messages=[
                f"{metrics['n_mentions']:.0f} mentions extraites de {metrics['n_documents']:.0f} "
                f"messages dans {written.name}.",
                f"Provenance : {metrics['share_from_rules']:.1%} des mentions viennent des règles, "
                f"{1 - metrics['share_from_rules']:.1%} du tagger ; confiance moyenne "
                f"{metrics['mean_confidence']:.2f}.",
                f"Latence p50 {metrics['latency_p50_ms']:.1f} ms, p95 "
                f"{metrics['latency_p95_ms']:.1f} ms par message, artefact chaud.",
            ],
        )

    # ------------------------------------------------------------------ interne -----------
    def _load_model(self) -> BaseEntityTagger:
        """Reload the model artefact declared by the stack (a directory)."""
        path = Path(self.paths.models_dir) / str(self.config.train.artifacts.model_file)
        return load_model(path, config=self.config.model_dump())

    def _reference(self, payload: pd.DataFrame) -> pd.DataFrame | None:
        """Return the reference annotations of an input sample, when they exist.

        Args:
            payload: Frame to annotate.

        Returns:
            The annotations of the sampled messages, or ``None`` when the corpus is unavailable
            (a user-provided file has no reference, which is the normal case).
        """
        loader = EntityCorpusLoader(self.paths, dataset_name=str(self.config.data.dataset_name))
        try:
            documents, spans = loader.load_corpus()
        except (FileNotFoundError, ValueError):
            return None
        identifiers = set(payload[str(self.config.data.id_column or "msg_id")].astype(str))
        known = set(documents["msg_id"].astype(str))
        if not identifiers <= known:
            return None
        return spans[spans["msg_id"].astype(str).isin(identifiers)].reset_index(drop=True)

    @staticmethod
    def _metrics(mentions: pd.DataFrame) -> dict[str, float]:
        """Summarise an inference run.

        Args:
            mentions: Validated mention table.

        Returns:
            Finite metrics describing the run.
        """
        latency = (
            mentions["latency_ms"].astype(float)
            if "latency_ms" in mentions.columns
            else pd.Series([0.0])
        )
        if "expected_label" in mentions.columns:
            labelled = mentions[mentions["expected_label"].notna()]
        else:
            labelled = mentions.iloc[:0]
        return {
            "n_mentions": float(len(mentions)),
            "n_documents": float(mentions["msg_id"].nunique()) if not mentions.empty else 0.0,
            "n_labelled": float(len(labelled)),
            "accuracy": float(labelled["correct"].mean()) if len(labelled) else 0.0,
            "mean_confidence": float(mentions["confidence"].astype(float).mean())
            if not mentions.empty
            else 0.0,
            "share_from_rules": float((mentions["source"].astype(str) == "regle").mean())
            if not mentions.empty
            else 0.0,
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(latency.quantile(0.50)),
            "latency_p95_ms": float(latency.quantile(0.95)),
        }


__all__ = ["InferencePipeline"]
