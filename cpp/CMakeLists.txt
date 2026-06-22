# NOTE:
# VSCode/CMakeTools may report pybind11 as "package not found".
# The extension still builds correctly via:
#   python -m pip install -e .
# This is an editor/CMakeTools configuration issue, not a real build failure.

cmake_minimum_required(VERSION 3.15)
project(fast_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_module
    bindings.cpp
    fast_module.cpp
)

target_link_libraries(
    fast_module
    PRIVATE
    OpenMP::OpenMP_CXX
)

install(TARGETS fast_module DESTINATION .)