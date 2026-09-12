import pandas as pd


class OrderTransformer:
    """Pure transformation layer: no file system access and safe to unit test."""

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.drop_duplicates(subset=["order_id"], keep="last").copy()
        result["order_date"] = pd.to_datetime(result["order_date"], utc=True)
        result["gross_amount"] = (result["quantity"] * result["unit_price"]).round(2)
        result["net_amount"] = result["gross_amount"].where(result["status"] == "paid", 0.0)
        result["is_revenue"] = result["status"].eq("paid")
        return result.sort_values("order_id").reset_index(drop=True)
