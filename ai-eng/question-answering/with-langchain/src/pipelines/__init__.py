"""End-to-end pipelines of the text modality.

Four pipelines cover the lifecycle and are the only entry points used by ``src/main.py``:

``DataGenerationPipeline``  generate the synthetic corpus and its annotated questions,
``TrainPipeline``           chunk, index, monitor on the validation questions, persist artefacts,
``EvaluationPipeline``      reload the artefacts, measure the test questions, write the report,
``InferencePipeline``       answer new questions and write the predictions.

Each one returns a :class:`PipelineResult`, which keeps the outcome machine readable.
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
