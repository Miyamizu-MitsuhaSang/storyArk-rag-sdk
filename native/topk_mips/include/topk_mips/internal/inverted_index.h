#ifndef TOPK_MIPS_INTERNAL_INVERTED_INDEX_H
#define TOPK_MIPS_INTERNAL_INVERTED_INDEX_H

#include "topk_mips/internal/library.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace topk_mips::internal {

class CSRMatrix;
class QueryMatrix;
class SortableList;

class InvertedIndex {
public:
    InvertedIndex();
    explicit InvertedIndex(const CSRMatrix& csr);

    void insert(int term_id, int vec_id, double val);
    std::optional<const std::vector<Posting>*> get_postings(int term_id) const;

    int get_word_frequency(int feature_id) const;
    int get_vectors() const;
    int get_features() const;
    const std::unordered_map<int, std::vector<Posting>>& get_index() const;

    SortableList candidate_calculator(const QueryMatrix& query, int topk) const;
    void clear();

private:
    static constexpr double VALUE_THRESHOLD = 0.1;

    std::unordered_map<int, std::vector<Posting>> index_;
    std::unordered_map<int, double> max_posting_value_;
    int num_vectors_;
    int num_features_;

    void build_index(const CSRMatrix& matrix);
    std::unordered_set<int> filter_by_threshold(const QueryMatrix& query, double value_threshold) const;
};

}  // namespace topk_mips::internal

#endif
