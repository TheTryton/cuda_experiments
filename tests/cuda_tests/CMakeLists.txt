cmake_minimum_required(VERSION 3.30)
project(cuda_tests LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

set(cuda_tests_sources
        "src/main.cu"
)

add_executable(cuda_tests
        ${cuda_tests_sources}
)
target_include_directories(cuda_tests PRIVATE
        "include"
        "src"
)
target_link_libraries(cuda_tests PRIVATE
        CUDA::cudart
        #CUDA::cuda_driver
        #CUDA::cupti
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