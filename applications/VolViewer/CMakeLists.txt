
file(GLOB_RECURSE
        VolViewer_SRCS
        "*.hpp"
        "*.cpp")

add_executable(VolViewer ${VolViewer_SRCS})

target_link_libraries(VolViewer
        PRIVATE
        VolumeViserLib)

target_compile_features(VolViewer PRIVATE cxx_std_20)

target_link_libraries(VolViewer PRIVATE  glfw)

set(DX11_LIBS d3d11.lib dxgi.lib d3dcompiler.lib dxguid.lib)

target_link_libraries(VolViewer PRIVATE ${DX11_LIBS})