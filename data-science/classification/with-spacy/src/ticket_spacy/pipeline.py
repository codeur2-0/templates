from pathlib import Path

import pandas as pd

from .data import TicketDataLoader
from .model import SpacyTicketClassifier


class TicketTrainingPipeline:
    def __init__(self, data_path: str | Path, output_dir: str | Path) -> None:
        self.data_path, self.output_dir = Path(data_path), Path(output_dir)

    def run(self, epochs: int, dropout: float) -> dict[str, float]:
        frame = TicketDataLoader(self.data_path).load()
        labels = sorted(frame["label"].unique().tolist())
        model = SpacyTicketClassifier(labels, dropout=dropout)
        losses = model.fit(frame, epochs=epochs)
        predictions = model.predict(frame["text"].tolist())
        accuracy = sum(
            max(item, key=item.get) == label for item, label in zip(predictions, frame["label"])
        ) / len(frame)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output_dir / "ticket_textcat")
        metrics = {"train_accuracy": float(accuracy), "final_loss": losses[-1]}
        pd.Series(metrics).to_json(self.output_dir / "metrics.json", indent=2)
        return metrics
