import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from churn_sklearn.data import CustomerDataLoader
from churn_sklearn.schemas import CustomerSchema


def test_example_dataset_respects_contract():
    frame = CustomerDataLoader("data/raw/customers.csv").load()
    assert set(frame["churned"].unique()) <= {0, 1}


def test_contract_rejects_negative_tickets():
    frame = pd.DataFrame(
        {
            "customer_id": ["C-999"],
            "age": [30],
            "monthly_spend": [10.0],
            "tenure_months": [1],
            "contract": ["monthly"],
            "support_tickets": [-1],
            "churned": [0],
        }
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        CustomerSchema.validate(frame, lazy=True)
