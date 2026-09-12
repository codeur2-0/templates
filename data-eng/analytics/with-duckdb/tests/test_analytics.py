from sales_duckdb.analytics import RevenueAnalytics


def test_revenue_query_returns_expected_columns():
    result = RevenueAnalytics("data/raw/sales.csv").run()
    assert {"country", "month", "net_revenue", "transactions"} <= set(result.columns)
    assert result["net_revenue"].sum() == 396.0
