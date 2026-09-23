#include "topk_mips/internal/query_matrix.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace topk_mips::internal {

QueryMatrix::QueryMatrix() = default;

QueryMatrix::QueryMatrix(const std::vector<std::pair<int, double>>& entries, int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be greater than 0");
    }

    std::unordered_set<int> seen;
    seen.reserve(entries.size());
    query_.reserve(entries.size());

    for (const auto& [term_id, value] : entries) {
        if (term_id < 0 || term_id >= num_features) {
            throw std::out_of_range("query feature id is outside [0, num_features)");
        }
        if (!seen.insert(term_id).second) {
            throw std::invalid_argument("query vectors must not contain duplicate feature ids");
        }
        if (value != 0.0) {
            query_.push_back({term_id, value});
        }
    }

    std::sort(query_.begin(), query_.end(), [](const Query& left, const Query& right) {
        return left.term_id < right.term_id;
    });
}

std::size_t QueryMatrix::size() const {
    return query_.size();
}

const std::vector<Query>& QueryMatrix::getquery() const {
    return query_;
}

}  // namespace topk_mips::internal
