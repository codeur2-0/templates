"""Pipelines du projet : génération, entraînement, évaluation, inférence.

Le contrat commun (:class:`~src.pipelines.base.BasePipeline`, ``PipelineResult``) vient du socle ;
ce
module assemble les quatre implémentations de l'extraction d'entités.
"""

from src.pipelines.base import BasePipeline, PipelineResult
from src.pipelines.data_pipeline import DataGenerationPipeline
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.train_pipeline import TrainPipeline

__all__ = [
    "BasePipeline",
    "DataGenerationPipeline",
    "EvaluationPipeline",
    "InferencePipeline",
    "PipelineResult",
    "TrainPipeline",
]
