import math

try:
    from translate_manager_rag._topk_mips import TopKMipsIndex as _NativeTopKMipsIndex
except ImportError as exc:  # pragma: no cover - exercised only when extension is not built
    _IMPORT_ERROR = exc
    _NativeTopKMipsIndex = None
else:
    _IMPORT_ERROR = None

from translate_manager_rag.persistence import FORMAT_VERSION, SDK_VERSION, pack_payload, unpack_payload

_PERSISTENCE_MAGIC = b"TMIDX001"


class TopKMipsIndex:
    """Python facade over the pybind11 C++ sparse MIPS extension."""

    def __init__(self, candidate_threshold: float = 0.0) -> None:
        if _NativeTopKMipsIndex is None:
            raise ImportError(
                "C++ extension translate_manager_rag._topk_mips is not built. "
                "Run `uv sync --index-url https://pypi.org/simple` or `uv run --index-url https://pypi.org/simple pytest`."
            ) from _IMPORT_ERROR
        self._index = _NativeTopKMipsIndex(candidate_threshold)
        self._rows: list[list[tuple[int, float]]] | None = None
        self._num_features: int | None = None

    def build(self, rows: list[list[tuple[int, float]]], num_features: int) -> None:
        self._index.build(rows, num_features)
        self._rows = [[(int(feature), float(weight)) for feature, weight in row] for row in rows]
        self._num_features = num_features

    def serialize(self) -> bytes:
        if self._rows is None or self._num_features is None:
            raise RuntimeError("build the index before serializing it")
        return pack_payload(
            _PERSISTENCE_MAGIC,
            {
                "sdk_version": SDK_VERSION,
                "format_version": FORMAT_VERSION,
                "num_features": self._num_features,
                "row_count": len(self._rows),
                "feature_count": self._index.feature_count,
                "candidate_threshold": self.candidate_threshold,
                "rows": self._rows,
            },
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> "TopKMipsIndex":
        value = unpack_payload(payload, _PERSISTENCE_MAGIC)
        rows = value.get("rows")
        num_features = value.get("num_features")
        threshold = value.get("candidate_threshold")
        if not isinstance(rows, list) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("invalid serialized index dimensions")
        if value.get("row_count") != len(rows) or value.get("feature_count") != num_features:
            raise ValueError("serialized index dimensions do not match payload")
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid serialized candidate threshold")

        parsed_rows: list[list[tuple[int, float]]] = []
        for row in rows:
            if not isinstance(row, list):
                raise ValueError("invalid serialized sparse row")
            parsed_row: list[tuple[int, float]] = []
            for pair in row:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid serialized sparse feature")
                feature, weight = pair
                if type(feature) is not int or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                    raise ValueError("invalid serialized sparse feature")
                parsed_row.append((feature, float(weight)))
            parsed_rows.append(parsed_row)

        index = cls(candidate_threshold=float(threshold))
        index.build(parsed_rows, num_features)
        return index

    def search(self, query: list[tuple[int, float]], top_k: int = 5) -> list[tuple[int, float]]:
        return [(result.row, result.score) for result in self._index.search(query, top_k)]

    def clear(self) -> None:
        self._index.clear()
        self._rows = None
        self._num_features = None

    @property
    def is_built(self) -> bool:
        return self._index.is_built

    @property
    def row_count(self) -> int:
        return self._index.row_count

    @property
    def feature_count(self) -> int:
        return self._index.feature_count

    @property
    def candidate_threshold(self) -> float:
        return self._index.candidate_threshold
