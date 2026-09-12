from uuid import uuid4

from fastapi import FastAPI

from .model import RuleBasedModel
from .schemas import ChurnRequest, ChurnResponse

app = FastAPI(title="Churn model service", version="2026.01.0")
model = RuleBasedModel()
MODEL_NAME, MODEL_VERSION, THRESHOLD = "churn-demo", "2026.01.0", 0.5


@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": MODEL_VERSION}


@app.post("/v1/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    probability = model.predict_probability(request)
    return ChurnResponse(
        request_id=str(uuid4()),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        churn_probability=round(probability, 4),
        action="retain" if probability >= THRESHOLD else "normal",
    )
