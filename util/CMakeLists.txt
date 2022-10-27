

option(ENABLE_OPENGL "" ON)

file(GLOB_RECURSE
        UTIL_SRCS
        "*.hpp"
        "*.cpp"
        ${PROJECT_SOURCE_DIR}/deps/glad/src/glad.c
        )

add_library(
        vutil_lib

        ${UTIL_SRCS}
)

target_compile_features(
        vutil_lib
        PUBLIC
        cxx_std_20
)

if(ENABLE_OPENGL)
target_compile_definitions(
        vutil_lib
        PUBLIC
        UTIL_ENABLE_OPENGL
)
endif()

target_include_directories(
        vutil_lib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/util
        ${PROJECT_SOURCE_DIR}/deps/glad/include
)

target_link_libraries(
        vutil_lib
        PUBLIC
        spdlog::spdlog
        imgui
)