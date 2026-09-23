#include "topk_mips/internal/compressed_matrix.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace topk_mips::internal {

CSRMatrix::CSRMatrix() : rows_(0), cols_(0) {}

CSRMatrix::CSRMatrix(std::vector<std::size_t> indptr,
                     std::vector<int> indices,
                     std::vector<double> data,
                     int rows,
                     int cols)
    : indptr_(std::move(indptr)),
      indices_(std::move(indices)),
      data_(std::move(data)),
      rows_(rows),
      cols_(cols) {
    if (rows_ < 0 || cols_ <= 0) {
        throw std::invalid_argument("CSRMatrix dimensions are invalid");
    }
    if (indptr_.size() != static_cast<std::size_t>(rows_) + 1) {
        throw std::invalid_argument("CSRMatrix indptr size must be rows + 1");
    }
    if (indices_.size() != data_.size()) {
        throw std::invalid_argument("CSRMatrix indices and data must have the same size");
    }
    if (!indptr_.empty() && indptr_.back() != indices_.size()) {
        throw std::invalid_argument("CSRMatrix indptr must end at nnz");
    }
}

CSRMatrix CSRMatrix::from_sparse_rows(const std::vector<SparseVector>& rows, int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be greater than 0");
    }

    std::vector<std::size_t> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    indptr.reserve(rows.size() + 1);
    indptr.push_back(0);

    for (const auto& row : rows) {
        SparseVector normalized;
        normalized.reserve(row.size());
        std::unordered_set<int> seen;
        seen.reserve(row.size());

        for (const auto& [feature_id, value] : row) {
            if (feature_id < 0 || feature_id >= num_features) {
                throw std::out_of_range("feature id is outside [0, num_features)");
            }
            if (!seen.insert(feature_id).second) {
                throw std::invalid_argument("sparse vectors must not contain duplicate feature ids");
            }
            if (value != 0.0) {
                normalized.push_back({feature_id, value});
            }
        }

        std::sort(normalized.begin(), normalized.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });

        for (const auto& [feature_id, value] : normalized) {
            indices.push_back(feature_id);
            data.push_back(value);
        }
        indptr.push_back(indices.size());
    }

    return CSRMatrix(std::move(indptr), std::move(indices), std::move(data), static_cast<int>(rows.size()), num_features);
}

int CSRMatrix::rows() const {
    return rows_;
}

int CSRMatrix::cols() const {
    return cols_;
}

std::size_t CSRMatrix::nnz() const {
    return data_.size();
}

const std::vector<std::size_t>& CSRMatrix::indptr() const {
    return indptr_;
}

const std::vector<int>& CSRMatrix::indices() const {
    return indices_;
}

const std::vector<double>& CSRMatrix::data() const {
    return data_;
}

}  // namespace topk_mips::internal
