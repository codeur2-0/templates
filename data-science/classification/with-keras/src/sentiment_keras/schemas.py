import pandera.pandas as pa
from pandera.typing import Series


class ReviewSchema(pa.DataFrameModel):
    review_id: Series[str] = pa.Field(str_matches=r"^R-\d{3}$")
    text: Series[str] = pa.Field(str_length={"min_value": 5})
    sentiment: Series[str] = pa.Field(isin=["positive", "neutral", "negative"])

    class Config:
        strict = True
        coerce = True
