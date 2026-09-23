#ifndef TOPK_MIPS_INTERNAL_QUERY_MATRIX_H
#define TOPK_MIPS_INTERNAL_QUERY_MATRIX_H

#include "topk_mips/internal/library.h"

#include <utility>
#include <vector>

namespace topk_mips::internal {

class QueryMatrix {
public:
    QueryMatrix();
    QueryMatrix(const std::vector<std::pair<int, double>>& entries, int num_features);

    std::size_t size() const;
    const std::vector<Query>& getquery() const;

private:
    std::vector<Query> query_;
};

}  // namespace topk_mips::internal

#endif
