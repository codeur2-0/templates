import pandera.pandas as pa
from pandera.typing import Series


class OrderSchema(pa.DataFrameModel):
    order_id: Series[str] = pa.Field(str_matches=r"^O-\d{3}$")
    order_date: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    customer_id: Series[str] = pa.Field(str_matches=r"^C-\d{3}$")
    country: Series[str] = pa.Field(isin=["FR", "BE", "DE", "ES"])
    quantity: Series[int] = pa.Field(gt=0)
    unit_price: Series[float] = pa.Field(ge=0)
    status: Series[str] = pa.Field(isin=["paid", "cancelled", "refunded"])

    class Config:
        strict = True
        coerce = True
