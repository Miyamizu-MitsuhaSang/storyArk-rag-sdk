#ifndef TOPK_MIPS_INTERNAL_CALCULATOR_H
#define TOPK_MIPS_INTERNAL_CALCULATOR_H

#include "topk_mips/internal/library.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace topk_mips::internal {

class Calculator {
public:
    static double inner_product(const std::unordered_map<int, std::vector<Posting>>& index,
                                const std::vector<Query>& query,
                                const std::vector<double>& max_contributions,
                                int candidate_vec_id,
                                double min_product,
                                double total_max);

    static bool index_binary_search(const std::vector<Posting>& postings, int target, std::size_t& vec_id);
};

}  // namespace topk_mips::internal

#endif
