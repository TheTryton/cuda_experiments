cmake_minimum_required(VERSION 3.30)
project(tetrahedral_interpolation LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(common)

add_subdirectory(core)
add_subdirectory(cuda)
add_subdirectory(tests/cuda_tests)
add_subdirectory(tests/cupti_tests)
add_subdirectory(tests/ram_gpuram_tests)

find_package(Stb REQUIRED)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" tetrahedral_interpolation_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" tetrahedral_interpolation_sources)

add_executable(tetrahedral_interpolation
        ${tetrahedral_interpolation_headers}
        ${tetrahedral_interpolation_sources}
)

target_include_directories(tetrahedral_interpolation PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${Stb_INCLUDE_DIR}
)
target_link_libraries(tetrahedral_interpolation
        PRIVATE
        tetrahedral_interpolation_core
        tetrahedral_interpolation_cuda
)
set_target_properties(tetrahedral_interpolation
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)