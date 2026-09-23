from translate_manager_rag import SparseMipsRetriever


def test_retriever_returns_metadata_with_scores():
    retriever = SparseMipsRetriever()
    retriever.build(
        documents=[
            {"id": "doc-a", "text": "alpha sparse vector"},
            {"id": "doc-b", "text": "beta sparse vector"},
            {"id": "doc-c", "text": "gamma"},
        ],
        vectors=[
            [(0, 1.0), (2, 0.5)],
            [(1, 1.0), (2, 2.0)],
            [(0, 0.25), (1, 0.25)],
        ],
        num_features=3,
    )

    results = retriever.search(query=[(2, 1.0), (1, 0.5)], top_k=2)

    assert results == [
        {
            "id": "doc-b",
            "text": "beta sparse vector",
            "score": 2.5,
            "row": 1,
        },
        {
            "id": "doc-a",
            "text": "alpha sparse vector",
            "score": 0.5,
            "row": 0,
        },
    ]
