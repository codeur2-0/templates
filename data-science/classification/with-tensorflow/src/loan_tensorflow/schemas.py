import pandera.pandas as pa
from pandera.typing import Series


class LoanSchema(pa.DataFrameModel):
    application_id: Series[str] = pa.Field(str_matches=r"^L-\d{3}$")
    income: Series[float] = pa.Field(gt=0)
    loan_amount: Series[float] = pa.Field(gt=0)
    years_employed: Series[int] = pa.Field(ge=0)
    credit_score: Series[int] = pa.Field(ge=300, le=850)
    defaulted: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True
