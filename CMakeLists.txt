cmake_minimum_required(VERSION 3.1)

project(softbody_simulation)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
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

include_directories(${EXTERNAL}/include)

find_package(OpenMP)

set(headers
    src/main.h
    src/intersections.h
    src/simulation/simulationContext.h
    src/simulation/softBody.h
    src/simulation/utilities.cuh
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/openglcontext/surfaceshader.h
    src/openglcontext/shaderprogram.h
    src/openglcontext/drawable.h
    src/openglcontext/texture.h
    src/mesh.h

    src/ImGui/imconfig.h
	src/ImGui/imgui.h
    src/ImGui/imconfig.h
    src/ImGui/imgui_impl_glfw.h
     src/ImGui/imgui_impl_opengl3.h 
     src/ImGui/imgui_impl_opengl3_loader.h 
     src/ImGui/imgui_internal.h 
     src/ImGui/imstb_rectpack.h 
     src/ImGui/imstb_textedit.h 
     src/ImGui/imstb_truetype.h 
    )

set(sources
    src/main.cpp
    src/stb.cpp
    src/simulation/softBody.cpp
    src/simulation/utilities.cu
    src/simulation/simulationContext.cu
    src/preview.cpp
    src/utilities.cpp
    src/openglcontext/surfaceshader.cpp
    src/openglcontext/shaderprogram.cpp
    src/openglcontext/drawable.cpp
    src/openglcontext/texture.cpp
    src/mesh.cpp
    
    src/ImGui/imgui.cpp 
      src/ImGui/imgui_demo.cpp 
      src/ImGui/imgui_draw.cpp 
     src/ImGui/imgui_impl_glfw.cpp 
     src/ImGui/imgui_impl_opengl3.cpp 
     src/ImGui/imgui_tables.cpp 
     src/ImGui/imgui_widgets.cpp 
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/ImGui ${PROJECT_SOURCE_DIR}/src/openglcontext ${PROJECT_SOURCE_DIR}/src/simulation ${PROJECT_SOURCE_DIR}/src/utilities ${PROJECT_SOURCE_DIR}/src/sceneStructs ${PROJECT_SOURCE_DIR}/src/openglcontext ${PROJECT_SOURCE_DIR}/src/simulation)
include_directories(
    ${PROJECT_INCLUDE_DIR} 
    ${GLM_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIR} 
    ${GLFW_INCLUDE_DIR})


cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    #stream_compaction  # TODO: uncomment if using your stream compaction
)
