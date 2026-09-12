from pathlib import Path
from typing import ClassVar

import pandas as pd
from tensorflow import keras


class LoanClassifier:
    FEATURES: ClassVar[list[str]] = ["income", "loan_amount", "years_employed", "credit_score"]

    def __init__(self, hidden_units: int, learning_rate: float) -> None:
        self.normalizer = keras.layers.Normalization(axis=-1)
        self.network = keras.Sequential(
            [
                keras.Input(shape=(len(self.FEATURES),)),
                self.normalizer,
                keras.layers.Dense(hidden_units, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def fit(
        self, train: pd.DataFrame, target: str, epochs: int, batch_size: int, validation_data=None
    ):
        x = train[self.FEATURES].astype("float32").to_numpy()
        y = train[target].astype("float32").to_numpy()
        self.normalizer.adapt(x)
        valid = None
        if validation_data is not None:
            valid = (
                validation_data[0][self.FEATURES].astype("float32").to_numpy(),
                validation_data[1].to_numpy(),
            )
        return self.network.fit(
            x, y, epochs=epochs, batch_size=batch_size, validation_data=valid, verbose=0
        )

    def predict(self, frame: pd.DataFrame):
        return self.network.predict(
            frame[self.FEATURES].astype("float32").to_numpy(), verbose=0
        ).ravel()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.network.save(path)
