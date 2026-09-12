from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnClassifier:
    """OOP facade around a leakage-safe sklearn pipeline."""

    FEATURES: ClassVar[list[str]] = [
        "age",
        "monthly_spend",
        "tenure_months",
        "contract",
        "support_tickets",
    ]

    def __init__(self, c: float = 1.0, max_iter: int = 300) -> None:
        numeric = ["age", "monthly_spend", "tenure_months", "support_tickets"]
        categorical = ["contract"]
        transformer = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        self.pipeline = Pipeline(
            [("features", transformer), ("classifier", LogisticRegression(C=c, max_iter=max_iter))]
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ChurnClassifier":
        self.pipeline.fit(features[self.FEATURES], target)
        return self

    def predict(self, features: pd.DataFrame):
        return self.pipeline.predict(features[self.FEATURES])

    def predict_proba(self, features: pd.DataFrame):
        return self.pipeline.predict_proba(features[self.FEATURES])[:, 1]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, destination)
