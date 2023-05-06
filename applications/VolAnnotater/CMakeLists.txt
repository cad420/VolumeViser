set(ENABLE_OPENGL ON)

option(BUILD_Neuron3DReconstruct "NOTE: build will cause volume render error" OFF)

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


if(BUILD_Neuron3DReconstruct)
    message(STATUS "Warning: use another project Neuron3DReconstruct(https://github.com/cad420/Neuron3DReconstruct)")
    add_subdirectory(tool)
endif()