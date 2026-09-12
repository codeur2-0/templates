import pandera.pandas as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(str_matches=r"^C-\d{3}$")
    tenure: Series[int] = pa.Field(ge=0)
    monthly_spend: Series[float] = pa.Field(ge=0)
    support_tickets: Series[int] = pa.Field(ge=0)
    churned: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True
