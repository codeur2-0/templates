from pathlib import Path

import pandas as pd

from .schemas import DocumentSchema


class MarkdownDocumentLoader:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def load(self) -> pd.DataFrame:
        records = []
        for path in sorted(self.directory.glob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            title = next(
                (line.removeprefix("# ") for line in lines if line.startswith("# ")), path.stem
            )
            records.append(
                {
                    "document_id": path.stem,
                    "title": title,
                    "text": "\n".join(lines),
                    "source": path.name,
                }
            )
        if not records:
            raise ValueError(f"No Markdown document found in {self.directory}")
        return DocumentSchema.validate(pd.DataFrame(records), lazy=True)
