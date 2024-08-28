cmake_minimum_required(VERSION 3.27)
project(tetrahedral_interpolation_cuda_kernels LANGUAGES CXX CUDA)

enable_language(CUDA)

find_c_and_cpp_files("include" tetrahedral_interpolation_cuda_kernels_headers)
find_c_and_cpp_files("src" tetrahedral_interpolation_cuda_kernels_sources)
find_cuda_files("src" tetrahedral_interpolation_cuda_kernels_cuda_sources)

add_library(tetrahedral_interpolation_cuda_kernels OBJECT
        ${tetrahedral_interpolation_cuda_kernels_headers}
        ${tetrahedral_interpolation_cuda_kernels_sources}
        ${tetrahedral_interpolation_cuda_kernels_cuda_sources}
)
target_include_directories(tetrahedral_interpolation_cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/"include"
        ${CMAKE_CURRENT_SOURCE_DIR}/"src"
)
set_target_properties(tetrahedral_interpolation_cuda_kernels
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_PTX_COMPILATION YES
)