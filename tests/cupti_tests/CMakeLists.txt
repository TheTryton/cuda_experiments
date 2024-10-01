cmake_minimum_required(VERSION 3.30)
project(cupti_tests LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

set(cupti_tests_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu"
)

add_executable(cupti_tests
        ${cupti_tests_sources}
)
target_include_directories(cupti_tests PRIVATE
        "include"
        "src"
)
target_link_libraries(cupti_tests PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cupti
)
target_compile_options(cupti_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        >)
set_target_properties(cupti_tests
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)