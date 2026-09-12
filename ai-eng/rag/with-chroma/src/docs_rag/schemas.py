import pandera.pandas as pa
from pandera.typing import Series


class DocumentSchema(pa.DataFrameModel):
    document_id: Series[str] = pa.Field(str_matches=r"^[a-z_]+$")
    title: Series[str] = pa.Field(str_length={"min_value": 2})
    text: Series[str] = pa.Field(str_length={"min_value": 20})
    source: Series[str] = pa.Field(str_length={"min_value": 1})

    class Config:
        strict = True
        coerce = True
