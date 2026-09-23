#ifndef TOPK_MIPS_INDEX_H
#define TOPK_MIPS_INDEX_H

#include <utility>
#include <vector>

#include "topk_mips/internal/compressed_matrix.h"
#include "topk_mips/internal/heap_sort.h"
#include "topk_mips/internal/inverted_index.h"

namespace topk_mips {

struct ScoredResult {
    int row;
    double score;
};

using SparseVector = std::vector<std::pair<int, double>>;

class SparseMipsIndex {
public:
    SparseMipsIndex();

    void build(const std::vector<SparseVector>& rows, int num_features);
    std::vector<ScoredResult> search(const SparseVector& query, int top_k) const;
    void clear();

    bool is_built() const;
    int rows() const;
    int features() const;

private:
    internal::CSRMatrix matrix_;
    internal::InvertedIndex inverted_index_;
    int num_features_;
    bool built_;

    static std::vector<ScoredResult> sort_results(const std::vector<internal::Matrix>& heap_results);
};

}  // namespace topk_mips

#endif
