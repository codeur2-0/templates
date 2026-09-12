from fastapi.testclient import TestClient

from risk_api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_contract():
    response = client.post(
        "/v1/predict", json={"income": 50000, "loan_amount": 8000, "credit_score": 700}
    )
    assert response.status_code == 200
    assert 0 < response.json()["risk_score"] < 1
