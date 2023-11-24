cmake_minimum_required(VERSION 3.18)

project(softbody_simulation LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDAToolkit REQUIRED)
########################################


find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

file(GLOB IMGUI_SOURCES "external/ImGui/*.cpp")
include_directories(${EXTERNAL}/include)

find_package(OpenMP)

set(headers
    src/main.h
    src/simulation/simulationContext.h
    src/simulation/softBody.h
    src/simulation/utilities.cuh
    src/collision/intersections.h
    src/collision/bvh.cuh
    src/collision/bvh.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/context.h
    src/openglcontext/surfaceshader.h
    src/openglcontext/shaderprogram.h
    src/openglcontext/drawable.h
    src/openglcontext/texture.h
    src/mesh.h

    src/svd3_cuda.h
    src/svd3.h

    src/constraint.h
    src/solver.h
    src/deformable_mesh.h
    src/deformation_gradient_constraint.h
    src/positional_constraint.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/simulation/softBody.cpp
    src/simulation/softBody.cu
    src/simulation/pdController.cu
    src/simulation/utilities.cu
    src/simulation/simulationContext.cu
    src/simulation/simulationContext.cpp
    src/collision/bvh.cu
    src/preview.cpp
    src/utilities.cpp
    src/context.cpp
    src/openglcontext/surfaceshader.cpp
    src/openglcontext/shaderprogram.cpp
    src/openglcontext/drawable.cpp
    src/openglcontext/texture.cpp
    src/mesh.cpp

    src/solver.cpp
    src/deformable_mesh.cpp
    src/deformation_gradient_constraint.cpp
    src/positional_constraint.cpp
)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

#add_subdirectory(src/ImGui)
set(PROJECT_INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/src 
${PROJECT_SOURCE_DIR}/src/ImGui 
${PROJECT_SOURCE_DIR}/src/openglcontext 
${PROJECT_SOURCE_DIR}/src/simulation 
${PROJECT_SOURCE_DIR}/src/utilities 
${PROJECT_SOURCE_DIR}/src/sceneStructs 
${PROJECT_SOURCE_DIR}/src/openglcontext 
${PROJECT_SOURCE_DIR}/src/collision 
${PROJECT_SOURCE_DIR}/src/simulation)

include_directories(
    ${PROJECT_INCLUDE_DIR} 
    ${GLM_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIR} 
    ${GLFW_INCLUDE_DIR})


add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${IMGUI_SOURCES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${EXTERNAL}/ImGui")
#"${EXTERNAL}/svd3_cuda"
target_link_libraries(${CMAKE_PROJECT_NAME}
${LIBRARIES}
CUDA::cudart
Eigen3::Eigen
OpenMP::OpenMP_CXX
CUDA::cusolver
#stream_compaction  # TODO: uncomment if using your stream compaction
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)