cmake_minimum_required(VERSION 3.30)
project(ram_gpuram_tests LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

set(cuda_tests_sources
        "src/main.cu"
)

add_executable(ram_gpuram_tests
        ${cuda_tests_sources}
)
target_include_directories(ram_gpuram_tests PRIVATE
        "include"
        "src"
)
target_link_libraries(ram_gpuram_tests PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cupti
)
target_compile_options(ram_gpuram_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        >)
set_target_properties(ram_gpuram_tests
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)