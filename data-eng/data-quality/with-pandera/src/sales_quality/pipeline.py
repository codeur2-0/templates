from pathlib import Path

import pandas as pd

from .report import QualityReport, SalesQualityChecker


class QualityPipeline:
    def __init__(self, data_path: str | Path, report_path: str | Path) -> None:
        self.data_path, self.report_path = Path(data_path), Path(report_path)

    def run(self) -> QualityReport:
        report = SalesQualityChecker().check(pd.read_csv(self.data_path), self.data_path.name)
        report.write(self.report_path)
        return report
