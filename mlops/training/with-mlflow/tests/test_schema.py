import pandas as pd

from churn_mlflow.schemas import ChurnSchema


def test_schema_accepts_small_frame():
    frame = pd.DataFrame(
        {
            "customer_id": ["C-001"],
            "tenure": [2],
            "monthly_spend": [10.0],
            "support_tickets": [0],
            "churned": [1],
        }
    )
    assert len(ChurnSchema.validate(frame)) == 1
