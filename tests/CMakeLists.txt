cmake_minimum_required(VERSION 3.27)
project(cuda_tests LANGUAGES CXX CUDA)

enable_language(CUDA)

set(cuda_tests_sources
    src/main.cu
)

add_executable(cuda_tests
        ${cuda_tests_sources}
)
target_include_directories(cuda_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_options(cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        >)
set_target_properties(cuda_tests
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)