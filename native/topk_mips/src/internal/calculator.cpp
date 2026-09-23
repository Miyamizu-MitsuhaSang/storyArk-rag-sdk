#include "topk_mips/internal/calculator.h"

namespace topk_mips::internal {

double Calculator::inner_product(const std::unordered_map<int, std::vector<Posting>>& index,
                                 const std::vector<Query>& query,
                                 const std::vector<double>& max_contributions,
                                 int candidate_vec_id,
                                 double min_product,
                                 double total_max) {
    double current_product = 0.0;

    for (std::size_t i = 0; i < query.size(); i++) {
        const auto& pair = query[i];
        total_max -= max_contributions[i];

        const auto it = index.find(pair.term_id);
        if (it != index.end()) {
            std::size_t vec_index = 0;
            if (index_binary_search(it->second, candidate_vec_id, vec_index)) {
                current_product += it->second[vec_index].value * pair.value;
            }
        }

        if (current_product + total_max < min_product) {
            break;
        }
    }

    return current_product;
}

bool Calculator::index_binary_search(const std::vector<Posting>& postings, int target, std::size_t& vec_id) {
    int top = static_cast<int>(postings.size()) - 1;
    int bottom = 0;

    while (bottom <= top) {
        const int mid = (bottom + top) / 2;
        const int id = postings[static_cast<std::size_t>(mid)].vec_id;
        if (target == id) {
            vec_id = static_cast<std::size_t>(mid);
            return true;
        }
        if (target < id) {
            top = mid - 1;
        } else {
            bottom = mid + 1;
        }
    }

    return false;
}

}  // namespace topk_mips::internal
