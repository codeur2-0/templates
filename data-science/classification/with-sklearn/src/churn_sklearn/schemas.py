import pandera.pandas as pa
from pandera.typing import Series


class CustomerSchema(pa.DataFrameModel):
    """Input contract for the customer snapshot."""

    customer_id: Series[str] = pa.Field(str_matches=r"^C-\d{3}$")
    age: Series[int] = pa.Field(ge=18, le=100)
    monthly_spend: Series[float] = pa.Field(ge=0)
    tenure_months: Series[int] = pa.Field(ge=0)
    contract: Series[str] = pa.Field(isin=["monthly", "annual"])
    support_tickets: Series[int] = pa.Field(ge=0)
    churned: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True
