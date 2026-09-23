#ifndef TOPK_MIPS_INTERNAL_LIBRARY_H
#define TOPK_MIPS_INTERNAL_LIBRARY_H

namespace topk_mips::internal {

enum class ErrorCode {
    success,
    underflow,
    overflow,
};

struct Posting {
    int vec_id;
    double value;

    Posting(int id, double val) : vec_id(id), value(val) {}
};

struct Query {
    int term_id;
    double value;

    Query(int id, double val) : term_id(id), value(val) {}
};

}  // namespace topk_mips::internal

#endif
