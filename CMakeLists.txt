cmake_minimum_required(VERSION 3.16)

project(VolumeViser LANGUAGES CXX C CUDA)

option(BUILD_TEST "" OFF)
option(BUILD_VolAnnotater "" ON)
option(BUILD_VolViewer "" ON)
option(BUILD_VolRenderer "" ON)
option(BUILD_VolEditor "" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(cuda)

add_subdirectory(util)

if(BUILD_TEST)
    add_subdirectory(tests)
endif()

add_subdirectory(deps)

file(GLOB_RECURSE
        CORE_SRCS
        "include/*.hpp"
        "src/*.cpp")

set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/Renderer/CRTVolumeRenderer.cpp
        ${PROJECT_SOURCE_DIR}/src/Renderer/RTVolumeRenderer.cpp
        ${PROJECT_SOURCE_DIR}/src/Renderer/PBVolumeRenderer.cpp
        ${PROJECT_SOURCE_DIR}/src/Renderer/LVolumeRenderer.cpp
        ${PROJECT_SOURCE_DIR}/src/Algorithm/MarchingCube.cpp
        ${PROJECT_SOURCE_DIR}/src/Algorithm/Voxelization.cpp
        ${PROJECT_SOURCE_DIR}/src/Algorithm/MeshSmoother.cpp
        PROPERTIES LANGUAGE CUDA
)

add_library(VolumeViserLib ${CORE_SRCS})

set_target_properties(
        VolumeViserLib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 80
)

target_include_directories(
        VolumeViserLib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_INCLUDE_DIR}
)

target_link_libraries(VolumeViserLib PUBLIC
        culib vutil_lib VolumeUtils
        ${MPI_LIBS}
        )


if(BUILD_VolAnnotater)
    add_subdirectory(applications/VolViewer)
endif()

if(BUILD_VolViewer)
    add_subdirectory(applications/VolAnnotater)
endif()

if(BUILD_VolRenderer)
    add_subdirectory(applications/VolRenderer)
endif()

if(BUILD_VolEditor)
    add_subdirectory(applications/VolEditor)
endif()