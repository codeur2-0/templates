"""End-to-end pipelines: the only entry points used by ``src/main.py`` and ``scripts/``."""

from src.pipelines.base import BasePipeline, PipelineResult
from src.pipelines.data_pipeline import DataGenerationPipeline, load_generated_dataset
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.train_pipeline import TrainPipeline, build_dataset_bundle

__all__ = [
    "BasePipeline",
    "DataGenerationPipeline",
    "EvaluationPipeline",
    "InferencePipeline",
    "PipelineResult",
    "TrainPipeline",
    "build_dataset_bundle",
    "load_generated_dataset",
]
