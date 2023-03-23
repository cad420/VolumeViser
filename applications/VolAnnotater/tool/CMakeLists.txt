
file(
        GLOB_RECURSE
        Neuron3DReconstruct_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(Neuron3DReconstruct ${Neuron3DReconstruct_SRCS})

target_link_libraries(
        Neuron3DReconstruct PRIVATE VolumeViserLib
)

target_compile_definitions(Neuron3DReconstruct PRIVATE cxx_std_20)