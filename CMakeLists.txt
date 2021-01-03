cmake_minimum_required(VERSION 3.17)
project(PerformanceModelling CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_executable(PerformanceModelling main.cpp perf.cu)

set_target_properties(
        PerformanceModelling
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
