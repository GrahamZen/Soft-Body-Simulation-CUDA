cmake_minimum_required(VERSION 3.18)
project(tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

include(catch2)
include(finite_diff)

file(GLOB_RECURSE TEST_SOURCES "*.cpp")
file(GLOB_RECURSE TEST_DISTANCE_SOURCES
    "${CMAKE_SOURCE_DIR}/src/simulation/distance/*.cu"
)

add_executable(tests
    ${TEST_SOURCES}
    ${TEST_DISTANCE_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/collision/intersections.cu
    ${CMAKE_SOURCE_DIR}/src/simulation/solver/linear/cg.cu
    ${CMAKE_SOURCE_DIR}/src/simulation/solver/linear/jacobi.cu
    ${CMAKE_SOURCE_DIR}/src/simulation/solver/linear/linear.cu
)

target_link_libraries(tests
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusolver
    Catch2::Catch2
    finitediff::finitediff
)

set_target_properties(tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
    -lineinfo
  >
)
