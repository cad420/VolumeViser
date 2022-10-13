
find_package(CUDA REQUIRED)
enable_language (CUDA)
file(GLOB_RECURSE
        CUDA_SRCS
        "*.cpp"
        "*.hpp")

add_library(culib ${CUDA_SRCS})

target_link_libraries(culib
        PUBLIC ${CUDA_CUDA_LIBRARY}
        ${CUDA_LIBRARIES}
        )

target_include_directories(culib
        PUBLIC ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/cuda
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_features(culib PUBLIC cxx_std_20)
set_target_properties(
        culib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
)

target_compile_options(
        culib
        PUBLIC
        "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
)

