import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, Field


class ChurnFeatures(pa.DataFrameModel):
    tenure: Series[int] = pa.Field(ge=0)
    monthly_spend: Series[float] = pa.Field(ge=0)
    support_tickets: Series[int] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True


class ChurnRequest(BaseModel):
    tenure: int = Field(ge=0)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class ChurnResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    churn_probability: float
    action: str
