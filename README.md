# storyArk RAG SDK

面向稀疏向量检索的 Python SDK，使用 C++17 实现 Top-K 最大内积搜索（MIPS），并提供可携带文档 metadata 的检索封装。

## 功能

- `TopKMipsIndex`：构建稀疏向量索引，按查询向量与文档向量的内积返回 Top-K 行号和分数。
- `SparseMipsRetriever`：将向量索引与文档 metadata 组合，搜索结果会带回原文档字段、`score` 和 `row`。
- 核心索引由 pybind11 扩展连接到 C++ 实现。

适合已经将文本转换为稀疏特征向量的检索流程；本 SDK 不负责文本切分、embedding 或稀疏特征生成。

## 环境要求

- Python 3.13 或更高版本
- 可用的 C++17 编译器
- 安装源码时需要 Git

## 安装

```bash
python -m pip install "git+https://github.com/Miyamizu-MitsuhaSang/storyArk-rag-sdk.git"
```

安装过程会编译 C++ 扩展。若编译失败，请确认本机已安装适用于当前 Python 版本的 C++ 工具链。

## 使用方式

稀疏向量使用 `(feature_id, weight)` 列表表示；`feature_id` 必须位于 `[0, num_features)`，同一个向量中不能重复。`num_features` 和 `top_k` 都必须大于 0。

### 直接使用索引

```python
from translate_manager_rag import TopKMipsIndex

index = TopKMipsIndex()
index.build(
    rows=[
        [(0, 1.0), (2, 0.5)],
        [(1, 1.0), (2, 2.0)],
    ],
    num_features=3,
)

# 返回按分数降序排列的 (row, score) 列表
results = index.search(query=[(2, 1.0), (1, 0.5)], top_k=2)
print(results)
# [(1, 2.5), (0, 0.5)]
```

### 返回文档 metadata

```python
from translate_manager_rag import SparseMipsRetriever

retriever = SparseMipsRetriever()
retriever.build(
    documents=[
        {"id": "doc-a", "text": "alpha"},
        {"id": "doc-b", "text": "beta"},
    ],
    vectors=[
        [(0, 1.0)],
        [(1, 2.0)],
    ],
    num_features=2,
)

results = retriever.search(query=[(1, 1.0)], top_k=1)
print(results)
# [{"id": "doc-b", "text": "beta", "score": 2.0, "row": 1}]
```

`documents` 与 `vectors` 必须一一对应。每个结果保留对应文档的字段，并附加 `score`（内积得分）和 `row`（索引中的行号）。

## 本地开发与测试

```bash
git clone https://github.com/Miyamizu-MitsuhaSang/storyArk-rag-sdk.git
cd storyArk-rag-sdk
uv sync
uv run pytest -q
```

`uv sync` 会安装开发依赖并构建本地 SDK；测试覆盖 Python 导出、检索封装和原生索引。
