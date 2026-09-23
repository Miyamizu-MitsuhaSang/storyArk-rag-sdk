#include "topk_mips/internal/dataset_reader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace topk_mips::internal {

CSRMatrix read_csr_dataset(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int64_t sizes[3];
    infile.read(reinterpret_cast<char*>(sizes), 3 * sizeof(int64_t));
    if (!infile) {
        throw std::runtime_error("Failed to read CSR header");
    }

    const int64_t num_rows = sizes[0];
    const int64_t num_cols = sizes[1];
    const int64_t nnz = sizes[2];
    if (num_rows < 0 || num_cols <= 0 || nnz < 0) {
        throw std::runtime_error("CSR header contains invalid dimensions");
    }

    std::vector<std::size_t> indptr(static_cast<std::size_t>(num_rows + 1));
    std::vector<int64_t> raw_indptr(static_cast<std::size_t>(num_rows + 1));
    infile.read(reinterpret_cast<char*>(raw_indptr.data()), (num_rows + 1) * sizeof(int64_t));
    if (!infile) {
        throw std::runtime_error("Failed to read CSR indptr");
    }
    for (std::size_t i = 0; i < raw_indptr.size(); ++i) {
        if (raw_indptr[i] < 0) {
            throw std::runtime_error("CSR indptr contains negative offsets");
        }
        indptr[i] = static_cast<std::size_t>(raw_indptr[i]);
    }

    std::vector<int> indices(static_cast<std::size_t>(nnz));
    infile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));
    if (!infile) {
        throw std::runtime_error("Failed to read CSR indices");
    }

    std::vector<float> raw_data(static_cast<std::size_t>(nnz));
    infile.read(reinterpret_cast<char*>(raw_data.data()), nnz * sizeof(float));
    if (!infile) {
        throw std::runtime_error("Failed to read CSR data");
    }

    std::vector<double> data;
    data.reserve(raw_data.size());
    for (const float value : raw_data) {
        data.push_back(static_cast<double>(value));
    }

    return CSRMatrix(std::move(indptr), std::move(indices), std::move(data), static_cast<int>(num_rows), static_cast<int>(num_cols));
}

}  // namespace topk_mips::internal
