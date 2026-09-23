from typing import Any

from translate_manager_rag.native import TopKMipsIndex

SparseVector = list[tuple[int, float]]
Document = dict[str, Any]


class SparseMipsRetriever:
    """RAG-facing retriever that keeps document metadata next to native vector search."""

    def __init__(self) -> None:
        self._index = TopKMipsIndex()
        self._documents: list[Document] = []

    def build(self, documents: list[Document], vectors: list[SparseVector], num_features: int) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        self._documents = [dict(document) for document in documents]
        self._index.build(vectors, num_features)

    def search(self, query: SparseVector, top_k: int) -> list[Document]:
        results: list[Document] = []
        for row, score in self._index.search(query, top_k):
            document = dict(self._documents[row])
            document["score"] = score
            document["row"] = row
            results.append(document)
        return results

    def clear(self) -> None:
        self._documents = []
        self._index.clear()
