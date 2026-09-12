from pathlib import Path

import duckdb
import pandas as pd

from .schemas import SaleSchema


class RevenueAnalytics:
    QUERY = """
        SELECT country, date_trunc('month', CAST(sale_date AS DATE)) AS month,
               SUM(CASE WHEN status = 'paid' THEN amount ELSE -amount END) AS net_revenue,
               COUNT(*) AS transactions
        FROM sales
        GROUP BY country, month
        ORDER BY month, country
    """

    def __init__(self, input_path: str | Path) -> None:
        self.input_path = Path(input_path)

    def run(self) -> pd.DataFrame:
        frame = SaleSchema.validate(pd.read_csv(self.input_path), lazy=True)
        with duckdb.connect() as connection:
            connection.register("sales", frame)
            return connection.sql(self.QUERY).df()
