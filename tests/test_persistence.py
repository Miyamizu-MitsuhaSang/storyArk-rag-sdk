from __future__ import annotations

import pytest

from translate_manager_rag import SparseMipsRetriever, TopKMipsIndex


def test_topk_index_round_trips_rows_and_search_results():
    original = TopKMipsIndex(candidate_threshold=0.0)
    original.build(
        rows=[[(0, 1.0), (2, 0.5)], [(1, 1.0), (2, 2.0)]],
        num_features=3,
    )

    restored = TopKMipsIndex.deserialize(original.serialize())

    assert restored.is_built
    assert restored.row_count == 2
    assert restored.feature_count == 3
    assert restored.candidate_threshold == original.candidate_threshold
    assert restored.search([(2, 1.0), (1, 0.5)], top_k=2) == original.search(
        [(2, 1.0), (1, 0.5)], top_k=2
    )


def test_sparse_retriever_round_trips_documents_and_search_results():
    original = SparseMipsRetriever(candidate_threshold=0.0)
    original.build(
        documents=[{"id": "a", "text": "alpha"}, {"id": "b", "tags": ["tm"]}],
        vectors=[[(0, 1.0)], [(1, 2.0)]],
        num_features=2,
    )

    restored = SparseMipsRetriever.deserialize(original.serialize())

    assert restored.search([(1, 1.0)], top_k=1) == original.search([(1, 1.0)], top_k=1)


def test_index_deserialization_rejects_corrupted_payload():
    index = TopKMipsIndex()
    index.build(rows=[[(0, 1.0)]], num_features=1)
    payload = bytearray(index.serialize())
    payload[-1] ^= 0x01

    with pytest.raises(ValueError, match="checksum"):
        TopKMipsIndex.deserialize(bytes(payload))


def test_retriever_deserialization_rejects_unsupported_format_version():
    retriever = SparseMipsRetriever()
    retriever.build(documents=[{"id": "a"}], vectors=[[(0, 1.0)]], num_features=1)
    payload = bytearray(retriever.serialize())
    payload[8:10] = b"\x00\x02"

    with pytest.raises(ValueError, match="format version"):
        SparseMipsRetriever.deserialize(bytes(payload))
