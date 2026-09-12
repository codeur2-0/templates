from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import CustomerDataLoader
from .model import ChurnClassifier


class ChurnTrainingPipeline:
    """Coordinates I/O, split, training and metric persistence."""

    def __init__(self, data_path: str | Path, output_dir: str | Path, seed: int = 42) -> None:
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.seed = seed

    def run(self, target: str, test_size: float, c: float, max_iter: int) -> dict[str, float]:
        frame = CustomerDataLoader(self.data_path).load()
        x = frame.drop(columns=[target])
        y = frame[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        model = ChurnClassifier(c=c, max_iter=max_iter).fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        metrics = {"roc_auc": float(roc_auc_score(y_test, probabilities))}
        report = classification_report(
            y_test, model.predict(x_test), output_dict=True, zero_division=0
        )
        metrics["churn_recall"] = float(report["1"]["recall"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output_dir / "churn_pipeline.joblib")
        pd.Series(metrics).to_json(self.output_dir / "metrics.json", indent=2)
        return metrics
