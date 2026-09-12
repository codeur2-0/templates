from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .data import LoanDataLoader
from .model import LoanClassifier


class LoanTrainingPipeline:
    def __init__(self, data_path: str | Path, output_dir: str | Path, seed: int = 42) -> None:
        self.data_path, self.output_dir, self.seed = Path(data_path), Path(output_dir), seed

    def run(
        self,
        test_size: float,
        epochs: int,
        batch_size: int,
        hidden_units: int,
        learning_rate: float,
    ) -> dict:
        tf.random.set_seed(self.seed)
        frame = LoanDataLoader(self.data_path).load()
        train, test = train_test_split(
            frame, test_size=test_size, random_state=self.seed, stratify=frame["defaulted"]
        )
        model = LoanClassifier(hidden_units, learning_rate)
        history = model.fit(train, "defaulted", epochs, batch_size, (test, test["defaulted"]))
        loss, accuracy = model.network.evaluate(test[model.FEATURES], test["defaulted"], verbose=0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output_dir / "loan_classifier.keras")
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "epochs": len(history.history["loss"]),
        }
        pd.Series(metrics).to_json(self.output_dir / "metrics.json", indent=2)
        return metrics
