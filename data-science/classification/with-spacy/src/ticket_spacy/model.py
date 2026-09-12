from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example


class SpacyTicketClassifier:
    """Owns the spaCy pipeline and exposes framework-neutral operations."""

    def __init__(self, labels: list[str], dropout: float = 0.2) -> None:
        self.labels, self.dropout = labels, dropout
        self.nlp = spacy.blank("en")
        self.textcat = self.nlp.add_pipe("textcat")
        for label in labels:
            self.textcat.add_label(label)

    def fit(self, frame: pd.DataFrame, epochs: int = 10) -> list[float]:
        examples = [
            Example.from_dict(
                self.nlp.make_doc(row.text),
                {"cats": {label: label == row.label for label in self.labels}},
            )
            for row in frame.itertuples()
        ]
        optimizer = self.nlp.initialize(lambda: examples)
        losses_history = []
        for _ in range(epochs):
            losses = {}
            self.nlp.update(examples, sgd=optimizer, drop=self.dropout, losses=losses)
            losses_history.append(float(losses.get("textcat", 0.0)))
        return losses_history

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.nlp(text).cats for text in texts]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(destination)
