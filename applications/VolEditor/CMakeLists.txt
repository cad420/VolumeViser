set(ENABLE_OPENGL ON)

file(
        GLOB_RECURSE
        VolEditor_SRCS
        "src/*.hpp"
        "src/*.cpp"
)

add_executable(VolEditor ${VolEditor_SRCS})

target_link_libraries(
        VolEditor PRIVATE VolumeViserLib
)

target_compile_definitions(VolEditor PRIVATE cxx_std_20)
