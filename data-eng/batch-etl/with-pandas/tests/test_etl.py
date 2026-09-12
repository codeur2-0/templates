import pandas as pd

from orders_etl.transform import OrderTransformer


def test_transform_is_idempotent_and_deduplicates():
    frame = pd.DataFrame(
        {
            "order_id": ["O-001", "O-001"],
            "order_date": ["2026-01-01"] * 2,
            "customer_id": ["C-001"] * 2,
            "country": ["FR"] * 2,
            "quantity": [2] * 2,
            "unit_price": [10.0] * 2,
            "status": ["paid"] * 2,
        }
    )
    result = OrderTransformer().run(frame)
    assert len(result) == 1
    assert result.loc[0, "net_amount"] == 20.0
