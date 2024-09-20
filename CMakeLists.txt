cmake_minimum_required(VERSION 3.18)

project(softbody_simulation LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G -g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

include(eigen)
include(spdlog)

file(GLOB_RECURSE HEADER_FILES 
     "${PROJECT_SOURCE_DIR}/src/*.h"
     "${PROJECT_SOURCE_DIR}/src/*.cuh")
file(GLOB_RECURSE SOURCE_FILES 
     "${PROJECT_SOURCE_DIR}/src/*.cpp"
     "${PROJECT_SOURCE_DIR}/src/*.cu")

foreach(FILE ${HEADER_FILES} ${SOURCE_FILES}) 
    get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)

    string(REPLACE "${PROJECT_SOURCE_DIR}/src" "" GROUP "${PARENT_DIR}")

    string(REPLACE "/" "\\" GROUP "${GROUP}")

    source_group("${GROUP}" FILES "${FILE}")
endforeach()


#add_subdirectory(src/ImGui)
set(PROJECT_INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/src 
${PROJECT_SOURCE_DIR}/src/openglcontext 
${PROJECT_SOURCE_DIR}/src/simulation
${PROJECT_SOURCE_DIR}/src/simulation/solver
${PROJECT_SOURCE_DIR}/src/simulation/distance
${PROJECT_SOURCE_DIR}/src/simulation/energy
${PROJECT_SOURCE_DIR}/src/simulation/solver/projective
${PROJECT_SOURCE_DIR}/src/collision 
${PROJECT_SOURCE_DIR}/src/collision/rigid
)

include_directories(
    ${PROJECT_INCLUDE_DIR}
    ${GLM_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIR} 
    ${GLFW_INCLUDE_DIR})

########################################
# Catch2 Tests
########################################
include(CTest)
enable_testing()
add_subdirectory(tests)
########################################
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${IMGUI_SOURCES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${EXTERNAL}/ImGui" "${EXTERNAL}/svd3_cuda")
target_link_libraries(${CMAKE_PROJECT_NAME}
${LIBRARIES}
CUDA::cudart
CUDA::cusolver
OpenMP::OpenMP_CXX
spdlog::spdlog
#stream_compaction  # TODO: uncomment if using your stream compaction
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --expt-relaxed-constexpr -Xcompiler "/wd 4819 /wd 4068 /wd 4661" -Xcudafe "--display_error_number --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=20014">
)