from hashlib import sha256
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd


class HashEmbeddingProvider:
    """Deterministic local embedding for a runnable example, not a semantic model."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in text.lower().split():
                index = int.from_bytes(sha256(token.encode()).digest()[:4], "big") % self.dimensions
                vector[index] += 1.0
            norm = np.linalg.norm(vector) or 1.0
            vectors.append((vector / norm).tolist())
        return vectors


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder: HashEmbeddingProvider | None = None,
    ) -> None:
        self.embedder = embedder or HashEmbeddingProvider()
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index(self, documents: pd.DataFrame) -> None:
        self.collection.upsert(
            ids=documents.document_id.tolist(),
            documents=documents.text.tolist(),
            metadatas=documents[["title", "source"]].to_dict("records"),
            embeddings=self.embedder.embed(documents.text.tolist()),
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]), n_results=top_k
        )
        return [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
