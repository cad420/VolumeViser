

add_executable(VolRendererUI main.cpp)

target_link_libraries(VolRendererUI PRIVATE VolumeViserLib)

target_compile_features(VolRendererUI PRIVATE cxx_std_20)