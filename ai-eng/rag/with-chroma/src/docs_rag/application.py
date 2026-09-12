from dataclasses import dataclass

from .loader import MarkdownDocumentLoader
from .retrieval import ChromaRetriever


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]


class GroundedAnswerGenerator:
    """Safe baseline: answer only with retrieved excerpts and cite their sources."""

    def generate(self, query: str, contexts: list[dict]) -> Answer:
        if not contexts:
            return Answer("I could not find an answer in the documentation.", [])
        source_names = [item["metadata"]["source"] for item in contexts]
        excerpt = contexts[0]["text"].replace("\n", " ")
        return Answer(f"Based on {source_names[0]}: {excerpt}", source_names)


class RAGApplication:
    def __init__(self, documents_dir, persist_dir, collection_name, top_k=3) -> None:
        self.loader = MarkdownDocumentLoader(documents_dir)
        self.retriever = ChromaRetriever(persist_dir, collection_name)
        self.top_k = top_k
        self.generator = GroundedAnswerGenerator()

    def ingest(self) -> int:
        documents = self.loader.load()
        self.retriever.index(documents)
        return len(documents)

    def answer(self, query: str) -> Answer:
        return self.generator.generate(query, self.retriever.search(query, self.top_k))
