from translate_manager_rag import SparseMipsRetriever, TopKMipsIndex


def test_sdk_exports_native_index():
    index = TopKMipsIndex()
    index.build(
        rows=[
            [(0, 1.0), (2, 0.5)],
            [(1, 1.0), (2, 2.0)],
        ],
        num_features=3,
    )

    assert index.search(query=[(2, 1.0), (1, 0.5)], top_k=1) == [(1, 2.5)]


def test_sdk_exports_rag_retriever():
    retriever = SparseMipsRetriever()
    retriever.build(
        documents=[{"id": "doc-a", "text": "alpha"}, {"id": "doc-b", "text": "beta"}],
        vectors=[[(0, 1.0)], [(1, 2.0)]],
        num_features=2,
    )

    assert retriever.search(query=[(1, 1.0)], top_k=1) == [
        {"id": "doc-b", "text": "beta", "score": 2.0, "row": 1}
    ]
