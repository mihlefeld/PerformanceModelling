cmake_minimum_required(VERSION 3.1)

project(PerformanceModelling)


link_directories(
		/usr/local/cuda-11.1/lib64
		/usr/local/cuda-10.1/lib64
		/usr/local/cuda-10.0/lib64
)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

find_package(CUDA REQUIRED)

# # Fermi
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_21")

# Kepler
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

# # Maxwell
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

# Turing
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")

cuda_add_executable(PerformanceModelling
		main.cpp perf.cu matrix.cpp tests.cpp)

cuda_add_library(cuMppm SHARED
		perf.cu matrix.cpp)

set_target_properties(cuMppm PROPERTIES
    VERSION 1
    SOVERSION 0)

# Activate (and require) C++11 support
set_property(TARGET PerformanceModelling PROPERTY CXX_STANDARD 17)
set_property(TARGET PerformanceModelling PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET cuMppm PROPERTY CXX_STANDARD 17)
set_property(TARGET cuMppm PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PerformanceModelling -lcublas -lcurand)
target_link_libraries(cuMppm -lcublas -lcurand)

install(TARGETS PerformanceModelling RUNTIME DESTINATION bin)
