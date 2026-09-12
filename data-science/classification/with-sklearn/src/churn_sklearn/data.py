from pathlib import Path

import pandas as pd

from .schemas import CustomerSchema


class CustomerDataLoader:
    """Reads an immutable CSV and enforces the data contract."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        frame = pd.read_csv(self.path)
        return CustomerSchema.validate(frame, lazy=True)
