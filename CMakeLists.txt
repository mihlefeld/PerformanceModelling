cmake_minimum_required(VERSION 3.1)

project(PerformanceModelling)

if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
	link_directories(
			/usr/local/cuda-11.1/lib64
			/usr/local/cuda-10.1/lib64
			/usr/local/cuda-10.0/lib64
	)
endif ()

# Set the default build type
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

find_package(CUDA REQUIRED)

cuda_add_executable(PerformanceModelling
		src/main.cpp
		src/perf.cu
		src/matrix.cpp
		test/tests.cpp
)

cuda_add_library(cuMPPM SHARED
		src/perf.cu
		src/matrix.cpp
		exports.DEF
)

set_target_properties(cuMPPM PROPERTIES
    VERSION 1
    SOVERSION 0)

# Activate (and require) C++11 support
set_property(TARGET PerformanceModelling PROPERTY CXX_STANDARD 17)
set_property(TARGET PerformanceModelling PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET cuMPPM PROPERTY CXX_STANDARD 17)
set_property(TARGET cuMPPM PROPERTY CXX_STANDARD_REQUIRED ON)

if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
	target_link_libraries(PerformanceModelling -lcublas -lcurand)
	target_link_libraries(cuMPPM -lcublas -lcurand)
else ()
	target_link_libraries(PerformanceModelling ${CUDA_CUBLAS_LIBRARIES})
	target_link_libraries(cuMPPM ${CUDA_CUBLAS_LIBRARIES})
endif ()


install(TARGETS PerformanceModelling RUNTIME DESTINATION bin)
