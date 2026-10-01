import base64
from typing import Any

from translate_manager_rag.native import TopKMipsIndex
from translate_manager_rag.persistence import FORMAT_VERSION, SDK_VERSION, pack_payload, unpack_payload

_PERSISTENCE_MAGIC = b"TMRTR001"

SparseVector = list[tuple[int, float]]
Document = dict[str, Any]


class SparseMipsRetriever:
    """RAG-facing retriever that keeps document metadata next to native vector search."""

    def __init__(self, candidate_threshold: float = 0.0) -> None:
        self._index = TopKMipsIndex(candidate_threshold=candidate_threshold)
        self._documents: list[Document] = []

    def build(self, documents: list[Document], vectors: list[SparseVector], num_features: int) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        self._documents = [dict(document) for document in documents]
        self._index.build(vectors, num_features)

    def search(self, query: SparseVector, top_k: int = 5) -> list[Document]:
        results: list[Document] = []
        for row, score in self._index.search(query, top_k):
            document = dict(self._documents[row])
            document["score"] = score
            document["row"] = row
            results.append(document)
        return results

    def serialize(self) -> bytes:
        if not self._index.is_built:
            raise RuntimeError("build the retriever before serializing it")
        return pack_payload(
            _PERSISTENCE_MAGIC,
            {
                "sdk_version": SDK_VERSION,
                "format_version": FORMAT_VERSION,
                "index": base64.b64encode(self._index.serialize()).decode("ascii"),
                "documents": self._documents,
            },
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> "SparseMipsRetriever":
        value = unpack_payload(payload, _PERSISTENCE_MAGIC)
        encoded_index = value.get("index")
        documents = value.get("documents")
        if not isinstance(encoded_index, str) or not isinstance(documents, list):
            raise ValueError("invalid serialized retriever payload")
        if any(not isinstance(document, dict) for document in documents):
            raise ValueError("serialized documents must be JSON objects")
        try:
            index_payload = base64.b64decode(encoded_index, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("invalid serialized retriever index") from exc
        index = TopKMipsIndex.deserialize(index_payload)
        if index.row_count != len(documents):
            raise ValueError("serialized documents do not match index row count")
        retriever = cls.__new__(cls)
        retriever._index = index
        retriever._documents = [dict(document) for document in documents]
        return retriever

    @property
    def candidate_threshold(self) -> float:
        return self._index.candidate_threshold

    def clear(self) -> None:
        self._documents = []
        self._index.clear()
