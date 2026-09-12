from pathlib import Path
from typing import ClassVar

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .schemas import ChurnSchema


class ExperimentRunner:
    FEATURES: ClassVar[list[str]] = ["tenure", "monthly_spend", "support_tickets"]

    def __init__(self, tracking_uri: str, experiment: str, seed: int = 42) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.seed = seed

    def run(
        self, data_path: str | Path, target: str, test_size: float, c: float
    ) -> dict[str, float | str]:
        frame = ChurnSchema.validate(pd.read_csv(data_path), lazy=True)
        train, test = train_test_split(
            frame, test_size=test_size, random_state=self.seed, stratify=frame[target]
        )
        model = LogisticRegression(C=c, max_iter=300).fit(train[self.FEATURES], train[target])
        auc = float(roc_auc_score(test[target], model.predict_proba(test[self.FEATURES])[:, 1]))
        with mlflow.start_run() as run:
            mlflow.log_params(
                {"c": c, "test_size": test_size, "seed": self.seed, "schema": "churn-v1"}
            )
            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(model, "model")
            return {"run_id": run.info.run_id, "roc_auc": auc}
