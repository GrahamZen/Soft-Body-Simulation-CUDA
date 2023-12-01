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

########################################
# Catch2 Setup
########################################
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)
########################################

########################################
# Catch2 Tests
########################################
file(GLOB_RECURSE TEST_SOURCES "./*.cpp")
add_executable(tests ${TEST_SOURCES} ${CMAKE_SOURCE_DIR}/src/collision/intersections.cu)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/src/collision/intersections)
target_link_libraries(tests PUBLIC CUDA::cudart Catch2::Catch2)
########################################
