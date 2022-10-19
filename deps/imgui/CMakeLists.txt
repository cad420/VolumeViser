file(GLOB
        IMGUI_SRCS
        "*.cpp"
        "*.h")

add_library(imgui ${IMGUI_SRCS})

target_include_directories(
        imgui
        PUBLIC
        ${PROJECT_SOURCE_DIR}/deps/imgui
)

target_link_libraries(
        imgui
        PUBLIC
        glfw
)