cmake_minimum_required(VERSION 3.27)
project(tetrahedral_interpolation_cuda LANGUAGES CXX)

add_subdirectory(kernels)

find_package(CUDAToolkit REQUIRED)

find_c_and_cpp_files("include" tetrahedral_interpolation_cuda_headers)
find_c_and_cpp_files("src" tetrahedral_interpolation_cuda_sources)

add_library(tetrahedral_interpolation_cuda STATIC
        ${tetrahedral_interpolation_cuda_headers}
        ${tetrahedral_interpolation_cuda_sources}
        ${tetrahedral_interpolation_cuda_cuda_sources}
)
target_include_directories(tetrahedral_interpolation_cuda PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(tetrahedral_interpolation_cuda PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(tetrahedral_interpolation_cuda PRIVATE
        tetrahedral_interpolation_core
        CUDA::cuda_driver
)
add_dependencies(tetrahedral_interpolation_cuda
        tetrahedral_interpolation_cuda_kernels
)
set_target_properties(tetrahedral_interpolation_cuda
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)