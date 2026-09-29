"""Preprocessing: custom transformers and their composition into a pipeline."""

from src.preprocessing.pipelines import (
    PreprocessingPipeline,
    PreprocessingReport,
    infer_column_kinds,
)
from src.preprocessing.transformers import (
    ColumnSelector,
    DataFrameScaler,
    Log1pTransformer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)

__all__ = [
    "ColumnSelector",
    "DataFrameScaler",
    "Log1pTransformer",
    "OutlierClipper",
    "PreprocessingPipeline",
    "PreprocessingReport",
    "RareCategoryGrouper",
    "TypeCaster",
    "infer_column_kinds",
]
