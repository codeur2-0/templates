import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from .schemas import SalesSchema


@dataclass(frozen=True)
class QualityReport:
    dataset: str
    rows: int
    valid: bool
    errors: list[str]

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class SalesQualityChecker:
    def check(self, frame: pd.DataFrame, dataset: str) -> QualityReport:
        try:
            SalesSchema.validate(frame, lazy=True)
            return QualityReport(dataset, len(frame), True, [])
        except (SchemaError, SchemaErrors) as error:
            # Keep the report serializable while preserving the actionable Pandera message.
            return QualityReport(dataset, len(frame), False, [str(error)])
