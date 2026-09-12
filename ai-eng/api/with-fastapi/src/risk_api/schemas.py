import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, Field


class RiskFeatureSchema(pa.DataFrameModel):
    income: Series[float] = pa.Field(gt=0)
    loan_amount: Series[float] = pa.Field(gt=0)
    credit_score: Series[int] = pa.Field(ge=300, le=850)

    class Config:
        strict = True
        coerce = True


class PredictionRequest(BaseModel):
    income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)


class PredictionResponse(BaseModel):
    model_version: str
    risk_score: float
    decision: str
