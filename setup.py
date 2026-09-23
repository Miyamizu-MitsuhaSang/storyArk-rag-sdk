from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "translate_manager_rag._topk_mips",
        [
            "native/topk_mips/src/index.cpp",
            "native/topk_mips/src/internal/calculator.cpp",
            "native/topk_mips/src/internal/compressed_matrix.cpp",
            "native/topk_mips/src/internal/dataset_reader.cpp",
            "native/topk_mips/src/internal/heap_sort.cpp",
            "native/topk_mips/src/internal/inverted_index.cpp",
            "native/topk_mips/src/internal/query_matrix.cpp",
            "native/topk_mips/bindings.cpp",
        ],
        include_dirs=["native/topk_mips/include"],
        cxx_std=17,
    )
]


setup(
    packages=find_packages(include=["translate_manager_rag", "translate_manager_rag.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
