from pathlib import Path

import pandas as pd

from .schemas import OrderSchema
from .transform import OrderTransformer


class OrderBatchPipeline:
    def __init__(self, input_path: str | Path, output_path: str | Path) -> None:
        self.input_path, self.output_path = Path(input_path), Path(output_path)

    def run(self) -> dict[str, int | float]:
        raw = pd.read_csv(self.input_path)
        validated = OrderSchema.validate(raw, lazy=True)
        clean = OrderTransformer().run(validated)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(self.output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        return {
            "input_rows": len(raw),
            "output_rows": len(clean),
            "revenue": float(clean["net_amount"].sum()),
        }
