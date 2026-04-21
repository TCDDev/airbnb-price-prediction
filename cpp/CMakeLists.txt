cmake_minimum_required(VERSION 3.15)
project(fast_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_module
    bindings.cpp
    fast_module.cpp
)

install(TARGETS fast_module DESTINATION .)