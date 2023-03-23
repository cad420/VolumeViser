set(ENABLE_OPENGL ON)

file(
        GLOB_RECURSE
        VolAnnotater_SRCS
        "src/*.hpp"
        "src/*.cpp"
)

add_executable(VolAnnotater ${VolAnnotater_SRCS})

target_link_libraries(
        VolAnnotater PRIVATE VolumeViserLib
)

target_compile_definitions(VolAnnotater PRIVATE cxx_std_20)

add_subdirectory(tool)