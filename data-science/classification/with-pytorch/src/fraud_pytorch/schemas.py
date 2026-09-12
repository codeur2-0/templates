import pandera.pandas as pa
from pandera.typing import Series


class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[str] = pa.Field(str_matches=r"^T-\d{3}$")
    amount: Series[float] = pa.Field(ge=0)
    transactions_last_hour: Series[int] = pa.Field(ge=0)
    device_age_days: Series[int] = pa.Field(ge=0)
    country_risk: Series[float] = pa.Field(ge=0, le=1)
    is_fraud: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True
