cmake_minimum_required(VERSION 3.16)

project(VolumeViser LANGUAGES CXX C CUDA)

add_subdirectory(cuda)

add_subdirectory(util)

add_subdirectory(tests)

add_subdirectory(deps)

file(GLOB_RECURSE
        CORE_SRCS
        "include/*.hpp"
        "src/*.cpp")

set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/Renderer/CRTVolumeRenderer.cpp
        ${PROJECT_SOURCE_DIR}/src/Renderer/RTVolumeRenderer.cpp
        PROPERTIES LANGUAGE CUDA
)

add_library(VolumeViserLib ${CORE_SRCS})



target_include_directories(
        VolumeViserLib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(VolumeViserLib PUBLIC
        culib vutil_lib)

target_link_libraries(VolumeViserLib PRIVATE
        VolumeUtils)


add_subdirectory(applications/VolViewer)

add_subdirectory(applications/VolAnnotater)