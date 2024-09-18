cmake_minimum_required(VERSION 3.18)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(CUDAToolkit REQUIRED)

include(catch2)
include(finite_diff)

########################################
# Catch2 Tests
########################################
file(GLOB_RECURSE TEST_SOURCES "./*.cpp")
file(GLOB_RECURSE TEST_DISTANCE_SOURCES "${CMAKE_SOURCE_DIR}/src/simulation/distance/*.cu")
add_executable(tests
${TEST_SOURCES}
${TEST_DISTANCE_SOURCES}
${CMAKE_SOURCE_DIR}/src/collision/intersections.cu
${CMAKE_SOURCE_DIR}/src/simulation/solver/linear/cg.cu
${CMAKE_SOURCE_DIR}/src/simulation/solver/linear/linear.cu
)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/src/simulation/solver/linear)
target_link_libraries(tests PUBLIC
Eigen3::Eigen
CUDA::cudart
Catch2::Catch2
CUDA::cusolver
finitediff::finitediff
)
set_target_properties(tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --expt-relaxed-constexpr -Xcompiler "/wd 4819 /wd 4068 /wd 4661" -Xcudafe "--display_error_number --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=20014">
)
########################################
