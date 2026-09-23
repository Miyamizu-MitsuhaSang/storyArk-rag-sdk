#ifndef TOPK_MIPS_INTERNAL_COMPRESSED_MATRIX_H
#define TOPK_MIPS_INTERNAL_COMPRESSED_MATRIX_H

#include <cstddef>
#include <utility>
#include <vector>

namespace topk_mips::internal {

using SparseVector = std::vector<std::pair<int, double>>;

class CSRMatrix {
public:
    CSRMatrix();
    CSRMatrix(std::vector<std::size_t> indptr, std::vector<int> indices, std::vector<double> data, int rows, int cols);

    static CSRMatrix from_sparse_rows(const std::vector<SparseVector>& rows, int num_features);

    int rows() const;
    int cols() const;
    std::size_t nnz() const;

    const std::vector<std::size_t>& indptr() const;
    const std::vector<int>& indices() const;
    const std::vector<double>& data() const;

private:
    std::vector<std::size_t> indptr_;
    std::vector<int> indices_;
    std::vector<double> data_;
    int rows_;
    int cols_;
};

}  // namespace topk_mips::internal

#endif
