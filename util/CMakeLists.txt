
file(GLOB_RECURSE
        UTIL_SRCS
        "*.hpp"
        "*.cpp")

add_library(
        vutil_lib

        ${UTIL_SRCS}
)

target_compile_features(
        vutil_lib
        PUBLIC
        cxx_std_20
)

target_include_directories(
        vutil_lib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/util
)