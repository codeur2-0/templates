from pathlib import Path

import pandas as pd

from .data import ReviewDataLoader
from .model import SentimentClassifier


class SentimentTrainingPipeline:
    def __init__(self, data_path: str | Path, output_dir: str | Path, seed: int = 42) -> None:
        self.data_path, self.output_dir, self.seed = Path(data_path), Path(output_dir), seed

    def run(
        self,
        epochs: int,
        batch_size: int,
        vocabulary_size: int,
        sequence_length: int,
        embedding_dim: int,
    ) -> dict:
        frame = ReviewDataLoader(self.data_path).load()
        model = SentimentClassifier(vocabulary_size, sequence_length, embedding_dim)
        history = model.fit(frame, "sentiment", epochs, batch_size)
        predictions = model.predict(frame["text"].tolist())
        accuracy = sum(
            pred == actual for pred, actual in zip(predictions, frame["sentiment"])
        ) / len(frame)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output_dir / "sentiment.keras")
        metrics = {
            "train_accuracy": float(accuracy),
            "final_loss": float(history.history["loss"][-1]),
        }
        pd.Series(metrics).to_json(self.output_dir / "metrics.json", indent=2)
        return metrics
