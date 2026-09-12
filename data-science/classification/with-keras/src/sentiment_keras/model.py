from pathlib import Path
from typing import ClassVar

import keras
import pandas as pd


class SentimentClassifier:
    LABELS: ClassVar[list[str]] = ["negative", "neutral", "positive"]

    def __init__(self, vocabulary_size: int, sequence_length: int, embedding_dim: int) -> None:
        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=vocabulary_size, output_mode="int", output_sequence_length=sequence_length
        )
        self.network = keras.Sequential(
            [
                keras.Input(shape=(), dtype="string"),
                self.vectorizer,
                keras.layers.Embedding(vocabulary_size, embedding_dim),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(len(self.LABELS), activation="softmax"),
            ]
        )
        self.network.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def fit(self, frame: pd.DataFrame, target: str, epochs: int, batch_size: int):
        self.vectorizer.adapt(frame["text"].to_numpy())
        labels = (
            frame[target].map({name: index for index, name in enumerate(self.LABELS)}).to_numpy()
        )
        return self.network.fit(
            frame["text"].to_numpy(), labels, epochs=epochs, batch_size=batch_size, verbose=0
        )

    def predict(self, texts: list[str]):
        probabilities = self.network.predict(texts, verbose=0)
        return [self.LABELS[index] for index in probabilities.argmax(axis=1)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.network.save(path)
