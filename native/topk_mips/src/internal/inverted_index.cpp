#include "topk_mips/internal/inverted_index.h"

#include "topk_mips/internal/calculator.h"
#include "topk_mips/internal/compressed_matrix.h"
#include "topk_mips/internal/heap_sort.h"
#include "topk_mips/internal/query_matrix.h"

#include <algorithm>

namespace topk_mips::internal {

InvertedIndex::InvertedIndex() : num_vectors_(0), num_features_(0) {}

InvertedIndex::InvertedIndex(const CSRMatrix& csr) : num_vectors_(0), num_features_(0) {
    build_index(csr);
}

void InvertedIndex::build_index(const CSRMatrix& matrix) {
    num_vectors_ = matrix.rows();
    num_features_ = matrix.cols();

    const auto& indptr = matrix.indptr();
    const auto& indices = matrix.indices();
    const auto& data = matrix.data();

    for (int vec_id = 0; vec_id < num_vectors_; vec_id++) {
        const std::size_t start = indptr[static_cast<std::size_t>(vec_id)];
        const std::size_t end = indptr[static_cast<std::size_t>(vec_id + 1)];
        for (std::size_t i = start; i < end; i++) {
            const int term_id = indices[i];
            const double value = data[i];
            index_[term_id].emplace_back(vec_id, value);
        }
    }

    for (auto& [term_id, postings] : index_) {
        std::sort(postings.begin(), postings.end(), [](const Posting& left, const Posting& right) {
            return left.vec_id < right.vec_id;
        });

        double max_val = 0.0;
        for (const auto& posting : postings) {
            max_val = std::max(max_val, posting.value);
        }
        max_posting_value_[term_id] = max_val;
    }
}

void InvertedIndex::insert(int term_id, int vec_id, double val) {
    index_[term_id].push_back({vec_id, val});
    std::sort(index_[term_id].begin(), index_[term_id].end(), [](const Posting& left, const Posting& right) {
        return left.vec_id < right.vec_id;
    });
    max_posting_value_[term_id] = std::max(max_posting_value_[term_id], val);
}

std::optional<const std::vector<Posting>*> InvertedIndex::get_postings(int term_id) const {
    const auto it = index_.find(term_id);
    if (it != index_.end()) {
        return &it->second;
    }
    return std::nullopt;
}

int InvertedIndex::get_word_frequency(int term_id) const {
    const auto it = index_.find(term_id);
    if (it != index_.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

int InvertedIndex::get_vectors() const {
    return num_vectors_;
}

int InvertedIndex::get_features() const {
    return num_features_;
}

const std::unordered_map<int, std::vector<Posting>>& InvertedIndex::get_index() const {
    return index_;
}

void InvertedIndex::clear() {
    index_.clear();
    max_posting_value_.clear();
    num_vectors_ = 0;
    num_features_ = 0;
}

std::unordered_set<int> InvertedIndex::filter_by_threshold(const QueryMatrix& query, double value_threshold) const {
    const std::vector<Query>& query_list = query.getquery();
    std::unordered_set<int> candidates;
    candidates.reserve(query_list.size() * 8);
    candidates.max_load_factor(0.6f);

    for (const auto& pair : query_list) {
        const auto it = index_.find(pair.term_id);
        if (it != index_.end()) {
            for (const auto& posting : it->second) {
                if (posting.value > value_threshold) {
                    candidates.insert(posting.vec_id);
                }
            }
        }
    }
    return candidates;
}

SortableList InvertedIndex::candidate_calculator(const QueryMatrix& query, int topk) const {
    std::vector<Query> sorted_query = query.getquery();
    std::sort(sorted_query.begin(), sorted_query.end(), [](const Query& a, const Query& b) {
        return a.value > b.value;
    });

    std::vector<double> max_contributions(sorted_query.size());
    double total_max = 0.0;
    for (std::size_t i = 0; i < sorted_query.size(); ++i) {
        const double max_val = max_posting_value_.count(sorted_query[i].term_id) ? max_posting_value_.at(sorted_query[i].term_id) : 0.0;
        max_contributions[i] = sorted_query[i].value * max_val;
        total_max += max_contributions[i];
    }

    std::unordered_set<int> candidates = filter_by_threshold(query, VALUE_THRESHOLD);

    SortableList candidate_heap{topk};
    double min_product = -1.0;
    for (const auto& candidate : candidates) {
        const double product = Calculator::inner_product(index_,
                                                         sorted_query,
                                                         max_contributions,
                                                         candidate,
                                                         min_product,
                                                         total_max);
        if (candidate_heap.size() < static_cast<std::size_t>(topk) || product > min_product) {
            candidate_heap.heap_insert(Matrix(candidate, product));
            if (candidate_heap.size() == static_cast<std::size_t>(topk)) {
                min_product = candidate_heap.get_list()[0].product;
            }
        }
    }

    return candidate_heap;
}

}  // namespace topk_mips::internal
