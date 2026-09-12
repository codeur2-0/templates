from pathlib import Path

from .analytics import RevenueAnalytics


class AnalyticsPipeline:
    def __init__(self, data_path: str | Path, output_path: str | Path) -> None:
        self.data_path, self.output_path = Path(data_path), Path(output_path)

    def run(self):
        result = RevenueAnalytics(self.data_path).run()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_path, index=False)
        return result
