#include "topk_mips/index.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_topk_mips, module) {
    module.doc() = "Sparse top-k maximum inner product search extension";

    py::class_<topk_mips::SparseMipsIndex>(module, "TopKMipsIndex")
        .def(py::init<>())
        .def("build", &topk_mips::SparseMipsIndex::build, py::arg("rows"), py::arg("num_features"))
        .def("search", &topk_mips::SparseMipsIndex::search, py::arg("query"), py::arg("top_k"))
        .def("clear", &topk_mips::SparseMipsIndex::clear)
        .def_property_readonly("is_built", &topk_mips::SparseMipsIndex::is_built)
        .def_property_readonly("row_count", &topk_mips::SparseMipsIndex::rows)
        .def_property_readonly("feature_count", &topk_mips::SparseMipsIndex::features);

    py::class_<topk_mips::ScoredResult>(module, "ScoredResult")
        .def_readonly("row", &topk_mips::ScoredResult::row)
        .def_readonly("score", &topk_mips::ScoredResult::score);
}
