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

if(OpenMP_CXX_FOUND)
    target_compile_options(fast_module PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_options(fast_module PRIVATE ${OpenMP_CXX_FLAGS})
endif()

install(TARGETS fast_module DESTINATION .)