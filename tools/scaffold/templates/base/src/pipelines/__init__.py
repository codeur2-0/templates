"""End-to-end pipelines: the only entry points used by ``src/main.py`` and ``scripts/``.

The concrete pipelines (data generation, training, evaluation, inference) are provided by the
modality layer of the project template; this module exposes the shared contract.
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
