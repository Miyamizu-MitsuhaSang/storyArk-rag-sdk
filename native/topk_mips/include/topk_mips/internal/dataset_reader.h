#ifndef TOPK_MIPS_INTERNAL_DATASET_READER_H
#define TOPK_MIPS_INTERNAL_DATASET_READER_H

#include "topk_mips/internal/compressed_matrix.h"

#include <string>

namespace topk_mips::internal {

CSRMatrix read_csr_dataset(const std::string& filename);

}  // namespace topk_mips::internal

#endif
