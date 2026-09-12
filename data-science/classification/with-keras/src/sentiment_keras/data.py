from pathlib import Path

import pandas as pd

from .schemas import ReviewSchema


class ReviewDataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return ReviewSchema.validate(pd.read_csv(self.path), lazy=True)
