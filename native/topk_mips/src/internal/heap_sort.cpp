#include "topk_mips/internal/heap_sort.h"

#include <algorithm>
#include <stdexcept>

namespace topk_mips::internal {

Matrix::Matrix() : vec_id(0), product(0.0) {}

Matrix::Matrix(int id, double product) : vec_id(id), product(product) {}

bool operator<(const Matrix& x, const Matrix& y) {
    return x.product < y.product;
}

bool operator>(const Matrix& x, const Matrix& y) {
    return x.product > y.product;
}

bool operator==(const Matrix& x, const Matrix& y) {
    return x.product == y.product;
}

bool operator<=(const Matrix& x, const Matrix& y) {
    return x.product <= y.product;
}

bool operator>=(const Matrix& x, const Matrix& y) {
    return x.product >= y.product;
}

SortableList::SortableList(int topk) : count_(0), topk_(topk) {
    if (topk <= 0) {
        throw std::invalid_argument("topk must be greater than 0");
    }
    list_.reserve(topk);
}

SortableList::SortableList(const std::vector<Matrix>& original, int topk)
    : list_(original), count_(static_cast<int>(original.size())), topk_(topk) {
    if (topk <= 0) {
        throw std::invalid_argument("topk must be greater than 0");
    }
}

const std::vector<Matrix>& SortableList::get_list() const {
    return list_;
}

std::size_t SortableList::size() const {
    return static_cast<std::size_t>(count_);
}

void SortableList::insert_heap(const Matrix& current, std::size_t low, std::size_t high) {
    std::size_t small = 2 * low + 1;
    while (small <= high) {
        if (small < high && list_[small] > list_[small + 1]) {
            small++;
        }
        if (current <= list_[small]) {
            break;
        }
        list_[low] = list_[small];
        low = small;
        small = 2 * low + 1;
    }
    list_[low] = current;
}

void SortableList::heap_insert(const Matrix& current) {
    if (count_ < topk_) {
        list_.push_back(current);
        count_++;
        insert_new(count_ - 1);
        return;
    }

    if (current > list_[0]) {
        insert_heap(current, 0, static_cast<std::size_t>(count_ - 1));
    }
}

void SortableList::insert_new(int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (list_[parent] > list_[index]) {
            std::swap(list_[parent], list_[index]);
        }
        index = parent;
    }
}

}  // namespace topk_mips::internal
