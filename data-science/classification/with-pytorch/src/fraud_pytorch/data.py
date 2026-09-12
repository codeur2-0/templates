from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .schemas import TransactionSchema

FEATURES = ["amount", "transactions_last_hour", "device_age_days", "country_risk"]


class TransactionDataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return TransactionSchema.validate(pd.read_csv(self.path), lazy=True)


class TransactionDataset(Dataset):
    """Converts an already validated frame into tensors."""

    def __init__(self, frame: pd.DataFrame, target: str = "is_fraud") -> None:
        self.features = torch.tensor(frame[FEATURES].to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(frame[target].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]
