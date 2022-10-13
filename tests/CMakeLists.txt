
add_executable(culib_test test_cuda.cpp)

set_source_files_properties(
        test_cuda.cpp
        PROPERTIES LANGUAGE CUDA
)

target_link_libraries(culib_test PRIVATE culib)

target_compile_features(culib_test PRIVATE cxx_std_20)




add_executable(test_rt_vol_renderer test_rt_vol_renderer.cpp)

target_link_libraries(test_rt_vol_renderer
        PRIVATE
        VolumeViserLib)

target_compile_features(test_rt_vol_renderer PRIVATE cxx_std_20)