import pytest

from translate_manager_rag import TopKMipsIndex


def test_native_index_accepts_candidate_threshold_and_default_top_k():
    index = TopKMipsIndex(candidate_threshold=0.0)
    index.build(rows=[[(0, 1.0)]], num_features=1)
    assert index.candidate_threshold == 0.0
    assert index.search(query=[(0, 1.0)]) == [(0, 1.0)]


def test_native_index_threshold_controls_candidate_recall():
    index = TopKMipsIndex(candidate_threshold=0.1)
    index.build(rows=[[(0, 0.05)], [(0, 0.5)]], num_features=1)
    assert index.search(query=[(0, 1.0)], top_k=2) == [(1, 0.5)]


def test_native_index_default_threshold_keeps_accumulated_low_weight_match():
    index = TopKMipsIndex()
    index.build(rows=[[(feature, 0.05) for feature in range(10)], [(0, 0.2)]], num_features=10)
    result = index.search(query=[(feature, 1.0) for feature in range(10)], top_k=1)
    assert result[0][0] == 0
    assert result[0][1] == pytest.approx(0.5)


def test_native_index_rejects_negative_vector_weights_for_pruned_mips():
    index = TopKMipsIndex()
    try:
        index.build(rows=[[(0, -1.0)]], num_features=1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative vector weights to be rejected")


def test_native_index_rejects_negative_query_weights_for_pruned_mips():
    index = TopKMipsIndex()
    index.build(rows=[[(0, 1.0)]], num_features=1)
    try:
        index.search(query=[(0, -1.0)], top_k=1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative query weights to be rejected")


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
