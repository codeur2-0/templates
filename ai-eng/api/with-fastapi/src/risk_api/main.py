from fastapi import FastAPI

from .schemas import PredictionRequest, PredictionResponse
from .service import RiskModelService

app = FastAPI(title="Risk scoring API", version="0.1.0")
service = RiskModelService(model_version="demo-risk-v1")


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": service.model_version}


@app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    return service.predict(request)
