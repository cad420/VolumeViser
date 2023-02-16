

file(GLOB_RECURSE
        VolViewer_SRCS
        "*.hpp"
        "*.cpp")

add_executable(VolViewer ${VolViewer_SRCS})

target_link_libraries(VolViewer
        PRIVATE
        VolumeViserLib)

target_compile_features(VolViewer PRIVATE cxx_std_20)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/SDL2)

target_include_directories(VolViewer PRIVATE ${SDL_INCLUDE_DIR})

target_link_libraries(VolViewer PRIVATE ${SDL_LIBS})
message(STATUS ${SDL_BIN})
file(COPY ${SDL_BIN} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})