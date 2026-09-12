import pandas as pd

from .schemas import PredictionRequest, PredictionResponse, RiskFeatureSchema


class RiskModelService:
    def __init__(self, model_version: str, threshold: float = 0.5) -> None:
        self.model_version, self.threshold = model_version, threshold

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        frame = pd.DataFrame([request.model_dump()])
        validated = RiskFeatureSchema.validate(frame, lazy=True)
        row = validated.iloc[0]
        utilization = row.loan_amount / row.income
        score = min(0.99, max(0.01, 0.55 * utilization + 0.45 * (700 - row.credit_score) / 400))
        return PredictionResponse(
            model_version=self.model_version,
            risk_score=round(float(score), 4),
            decision="review" if score >= self.threshold else "approve",
        )
