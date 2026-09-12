import pandera.pandas as pa
from pandera.typing import Series


class SaleSchema(pa.DataFrameModel):
    sale_id: Series[str] = pa.Field(str_matches=r"^V-\d{3}$")
    sale_date: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    country: Series[str] = pa.Field(isin=["FR", "BE", "DE", "ES"])
    amount: Series[float] = pa.Field(ge=0)
    status: Series[str] = pa.Field(isin=["paid", "refunded"])

    class Config:
        strict = True
        coerce = True
