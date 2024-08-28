cmake_minimum_required(VERSION 3.27)
project(tetrahedral_interpolation_core LANGUAGES CXX)

#enable_language(CUDA)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" tetrahedral_interpolation_core_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" tetrahedral_interpolation_core_sources)

add_library(tetrahedral_interpolation_core STATIC
        ${tetrahedral_interpolation_core_headers}
        ${tetrahedral_interpolation_core_sources}
)
target_include_directories(tetrahedral_interpolation_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(tetrahedral_interpolation_core PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set_target_properties(tetrahedral_interpolation_core
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)