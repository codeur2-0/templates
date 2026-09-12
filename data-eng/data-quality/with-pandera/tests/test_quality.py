import pandas as pd

from sales_quality.report import SalesQualityChecker


def test_quality_checker_accepts_sample():
    frame = pd.DataFrame(
        {
            "sale_id": ["V-001"],
            "sale_date": ["2026-01-01"],
            "product_category": ["software"],
            "quantity": [1],
            "amount": [10.0],
            "country": ["FR"],
        }
    )
    assert SalesQualityChecker().check(frame, "unit.csv").valid
