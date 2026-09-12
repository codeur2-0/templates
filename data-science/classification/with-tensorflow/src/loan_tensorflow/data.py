from pathlib import Path

import pandas as pd

from .schemas import LoanSchema


class LoanDataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return LoanSchema.validate(pd.read_csv(self.path), lazy=True)
