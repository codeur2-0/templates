import pandera.pandas as pa
from pandera.typing import Series


class SalesSchema(pa.DataFrameModel):
    sale_id: Series[str] = pa.Field(str_matches=r"^V-\d{3}$", unique=True)
    sale_date: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    product_category: Series[str] = pa.Field(isin=["software", "hardware", "subscription"])
    quantity: Series[int] = pa.Field(gt=0)
    amount: Series[float] = pa.Field(ge=0)
    country: Series[str] = pa.Field(isin=["FR", "BE", "DE", "ES"])

    class Config:
        strict = True
        coerce = True
