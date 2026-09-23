try:
    from translate_manager_rag._topk_mips import TopKMipsIndex as _NativeTopKMipsIndex
except ImportError as exc:  # pragma: no cover - exercised only when extension is not built
    _IMPORT_ERROR = exc
    _NativeTopKMipsIndex = None
else:
    _IMPORT_ERROR = None


class TopKMipsIndex:
    """Python facade over the pybind11 C++ sparse MIPS extension."""

    def __init__(self) -> None:
        if _NativeTopKMipsIndex is None:
            raise ImportError(
                "C++ extension translate_manager_rag._topk_mips is not built. "
                "Run `uv sync --index-url https://pypi.org/simple` or `uv run --index-url https://pypi.org/simple pytest`."
            ) from _IMPORT_ERROR
        self._index = _NativeTopKMipsIndex()

    def build(self, rows: list[list[tuple[int, float]]], num_features: int) -> None:
        self._index.build(rows, num_features)

    def search(self, query: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        return [(result.row, result.score) for result in self._index.search(query, top_k)]

    def clear(self) -> None:
        self._index.clear()

    @property
    def is_built(self) -> bool:
        return self._index.is_built

    @property
    def row_count(self) -> int:
        return self._index.row_count

    @property
    def feature_count(self) -> int:
        return self._index.feature_count
