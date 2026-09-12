import pandera.pandas as pa
from pandera.typing import Series


class TicketSchema(pa.DataFrameModel):
    ticket_id: Series[str] = pa.Field(str_matches=r"^S-\d{3}$")
    text: Series[str] = pa.Field(str_length={"min_value": 5})
    label: Series[str] = pa.Field(isin=["billing", "technical", "account"])

    class Config:
        strict = True
        coerce = True
