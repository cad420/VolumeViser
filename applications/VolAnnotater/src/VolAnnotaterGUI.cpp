#include "VolAnnotaterGUI.hpp"
#include <imgui_internal.h>
void VolAnnotaterGUI::Initialize() {

}

// 初始化opengl相关
void VolAnnotaterGUI::initialize() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("EditorFont.TTF",  16, nullptr, nullptr);
    io.Fonts->Build();



}

void VolAnnotaterGUI::frame() {
    framebuffer_t::bind_to_default();
    framebuffer_t::clear_color_depth_buffer();

    show_editor_ui();

}

void VolAnnotaterGUI::destroy() {

}

void VolAnnotaterGUI::handle_events() {

    gl_app_t::handle_events();
}

void VolAnnotaterGUI::show_editor_ui() {

    show_editor_menu(&editor_menu_window_open);

    show_editor_vol_info_window(&vol_info_window_open);

    show_editor_vol_render_window(&vol_render_window_open);

    show_editor_swc_window(&swc_info_window_open);

    show_editor_swc_tree_window(&swc_tree_window_open);

    show_editor_neuron_mesh(&neuron_mesh_window_open);
}

void VolAnnotaterGUI::show_editor_menu(bool* p_open) {

    ImGuiDockNodeFlags dock_flags = ImGuiDockNodeFlags_DockSpace;
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground |
                                    ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_NoBringToFrontOnFocus;

    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(window->get_window_width(), window->get_window_height()), ImGuiCond_Always);

    ImGui::SetNextWindowViewport(main_viewport->ID);

    ImGui::Begin("Editor menu", p_open, window_flags);

    ImGuiID main_docking_id = ImGui::GetID("Main Docking");

    if(ImGui::DockBuilderGetNode(main_docking_id) == nullptr){

        ImGui::DockBuilderRemoveNode(main_docking_id);

        ImGui::DockBuilderAddNode(main_docking_id, dock_flags);

        ImGui::DockBuilderSetNodePos(main_docking_id,
                                     ImVec2(main_viewport->WorkPos.x, main_viewport->WorkPos.y + 18.0f));
        ImGui::DockBuilderSetNodeSize(main_docking_id,
                                      ImVec2(window->get_window_width(), window->get_window_height() - 18.0f));


        ImGui::DockBuilderFinish(main_docking_id);
    }

    ImGui::DockSpace(main_docking_id);

    if(ImGui::BeginMenuBar()){
        if(ImGui::BeginMenu("Menu")){
            if(ImGui::MenuItem("Load Project File")){

            }
            if(ImGui::MenuItem("New Project File")){

            }
            if(ImGui::MenuItem("Save Project File")){

            }
            if(ImGui::MenuItem("Exit")){

            }

            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Window")){
            ImGui::MenuItem("Vol Info", nullptr, &vol_info_window_open);
            ImGui::MenuItem("Vol Render", nullptr, &vol_render_window_open);
            ImGui::MenuItem("SWC Info", nullptr, &swc_info_window_open);
            ImGui::MenuItem("SWC Tree", nullptr, &swc_tree_window_open);
            ImGui::MenuItem("Neuron Info", nullptr, &neuron_mesh_window_open);

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }


    ImGui::End();
}

void VolAnnotaterGUI::show_editor_vol_info_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(!*p_open) return;

    if(ImGui::Begin("Vol Info", p_open, window_flags)){

    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_vol_render_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(!*p_open) return;

    if(ImGui::Begin("Vol Render", p_open, window_flags)){

    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_swc_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(!*p_open) return;

    if(ImGui::Begin("SWC Info", p_open, window_flags)){

    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_swc_tree_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(!*p_open) return;

    if(ImGui::Begin("SWC Tree View", p_open, window_flags)){

    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_neuron_mesh(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(!*p_open) return;

    if(ImGui::Begin("Neuron Mesh Info", p_open, window_flags)){

    }

    ImGui::End();
}




