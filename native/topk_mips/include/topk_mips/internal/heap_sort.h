#ifndef TOPK_MIPS_INTERNAL_HEAP_SORT_H
#define TOPK_MIPS_INTERNAL_HEAP_SORT_H

#include <vector>

namespace topk_mips::internal {

struct Matrix {
    int vec_id;
    double product;

    Matrix();
    Matrix(int id, double product);
};

bool operator<(const Matrix& x, const Matrix& y);
bool operator>(const Matrix& x, const Matrix& y);
bool operator==(const Matrix& x, const Matrix& y);
bool operator<=(const Matrix& x, const Matrix& y);
bool operator>=(const Matrix& x, const Matrix& y);

class SortableList {
public:
    explicit SortableList(int topk);
    SortableList(const std::vector<Matrix>& original, int topk);

    void heap_insert(const Matrix& current);
    const std::vector<Matrix>& get_list() const;
    std::size_t size() const;

private:
    void insert_heap(const Matrix& current, std::size_t low, std::size_t high);
    void insert_new(int index);

    std::vector<Matrix> list_;
    int count_;
    int topk_;
};

}  // namespace topk_mips::internal

#endif
