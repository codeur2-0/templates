import pandas as pd

from .schemas import ChurnFeatures, ChurnRequest


class RuleBasedModel:
    """Drop-in model adapter used only to keep this template self-contained."""

    def predict_probability(self, request: ChurnRequest) -> float:
        frame = pd.DataFrame([request.model_dump()])
        row = ChurnFeatures.validate(frame, lazy=True).iloc[0]
        probability = 0.65 - min(row.tenure, 36) * 0.01 + min(row.support_tickets, 6) * 0.04
        return float(min(0.99, max(0.01, probability)))
