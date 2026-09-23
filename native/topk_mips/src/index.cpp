#include "topk_mips/index.h"

#include <algorithm>
#include <stdexcept>

#include "topk_mips/internal/heap_sort.h"
#include "topk_mips/internal/query_matrix.h"

namespace topk_mips {

SparseMipsIndex::SparseMipsIndex() : num_features_(0), built_(false) {}

void SparseMipsIndex::build(const std::vector<SparseVector>& rows, int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be greater than 0");
    }

    matrix_ = internal::CSRMatrix::from_sparse_rows(rows, num_features);
    inverted_index_ = internal::InvertedIndex(matrix_);
    num_features_ = num_features;
    built_ = true;
}

std::vector<ScoredResult> SparseMipsIndex::search(const SparseVector& query, int top_k) const {
    if (!built_) {
        throw std::runtime_error("SparseMipsIndex.search called before build");
    }
    if (top_k <= 0) {
        throw std::invalid_argument("top_k must be greater than 0");
    }

    const internal::QueryMatrix query_matrix(query, num_features_);
    if (query_matrix.size() == 0) {
        return {};
    }

    internal::SortableList candidates = inverted_index_.candidate_calculator(query_matrix, top_k);
    return sort_results(candidates.get_list());
}

void SparseMipsIndex::clear() {
    matrix_ = internal::CSRMatrix();
    inverted_index_.clear();
    num_features_ = 0;
    built_ = false;
}

bool SparseMipsIndex::is_built() const {
    return built_;
}

int SparseMipsIndex::rows() const {
    return matrix_.rows();
}

int SparseMipsIndex::features() const {
    return num_features_;
}

std::vector<ScoredResult> SparseMipsIndex::sort_results(const std::vector<internal::Matrix>& heap_results) {
    std::vector<ScoredResult> results;
    results.reserve(heap_results.size());
    for (const auto& result : heap_results) {
        results.push_back({result.vec_id, result.product});
    }

    std::sort(results.begin(), results.end(), [](const ScoredResult& left, const ScoredResult& right) {
        if (left.score == right.score) {
            return left.row < right.row;
        }
        return left.score > right.score;
    });
    return results;
}

}  // namespace topk_mips
