from fastapi.testclient import TestClient

from churn_service.main import app


def test_service_returns_version_and_request_id():
    response = TestClient(app).post(
        "/v1/predict", json={"tenure": 4, "monthly_spend": 29.0, "support_tickets": 3}
    )
    assert response.status_code == 200
    assert response.json()["model_version"] == "2026.01.0"
    assert response.json()["request_id"]
