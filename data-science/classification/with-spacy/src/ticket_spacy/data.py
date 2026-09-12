from pathlib import Path

import pandas as pd

from .schemas import TicketSchema


class TicketDataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return TicketSchema.validate(pd.read_csv(self.path), lazy=True)
