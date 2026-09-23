from translate_manager_rag import TopKMipsIndex


def test_native_index_returns_top_k_inner_products():
    index = TopKMipsIndex()
    index.build(
        rows=[
            [(0, 1.0), (2, 0.5)],
            [(1, 1.0), (2, 2.0)],
            [(0, 0.25), (1, 0.25)],
        ],
        num_features=3,
    )

    results = index.search(query=[(2, 1.0), (1, 0.5)], top_k=2)

    assert results == [(1, 2.5), (0, 0.5)]


def test_native_index_rejects_search_before_build():
    index = TopKMipsIndex()

    try:
        index.search(query=[(0, 1.0)], top_k=1)
    except RuntimeError as exc:
        assert "build" in str(exc)
    else:
        raise AssertionError("Expected search before build to raise RuntimeError")


def test_native_index_uses_inverted_candidates_without_zero_score_padding():
    index = TopKMipsIndex()
    index.build(
        rows=[
            [(0, 1.0)],
            [(1, 1.0)],
            [(2, 1.0)],
        ],
        num_features=3,
    )

    results = index.search(query=[(2, 1.0)], top_k=3)

    assert results == [(2, 1.0)]
