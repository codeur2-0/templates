from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .data import TransactionDataLoader, TransactionDataset
from .model import FraudMLP, TorchTrainer


class FraudTrainingPipeline:
    def __init__(self, data_path: str | Path, output_dir: str | Path, seed: int = 42) -> None:
        self.data_path, self.output_dir, self.seed = Path(data_path), Path(output_dir), seed

    def run(
        self, test_size: float, epochs: int, batch_size: int, learning_rate: float, hidden_dim: int
    ) -> dict:
        torch.manual_seed(self.seed)
        frame = TransactionDataLoader(self.data_path).load()
        train, test = train_test_split(
            frame, test_size=test_size, random_state=self.seed, stratify=frame["is_fraud"]
        )
        model = FraudMLP(hidden_dim=hidden_dim)
        trainer = TorchTrainer(model, learning_rate=learning_rate, epochs=epochs)
        history = trainer.fit(
            DataLoader(TransactionDataset(train), batch_size=batch_size, shuffle=True)
        )
        model.eval()
        with torch.no_grad():
            logits = model(TransactionDataset(test).features)
            predictions = (torch.sigmoid(logits) >= 0.5).int().numpy()
        accuracy = float((predictions == test["is_fraud"].to_numpy()).mean())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output_dir / "fraud_mlp.pt")
        metrics = {"accuracy": accuracy, "final_loss": history[-1]}
        pd.Series(metrics).to_json(self.output_dir / "metrics.json", indent=2)
        return metrics
