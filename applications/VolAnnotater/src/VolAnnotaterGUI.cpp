#include <Algorithm/Voxelization.hpp>
#include <imgui_internal.h>
#include "VolAnnotaterGUI.hpp"
#include "SWCRenderer.hpp"
#include "NeuronRenderer.hpp"
#include "Common.hpp"
#include <unordered_set>

#define FOV 40.f

void VolAnnotaterGUI::Initialize() {
    viser_resc = std::make_unique<ViserRescPack>();
    viser_resc->Initialize();

    vol_render_resc = std::make_unique<VolRenderRescPack>();
    vol_render_resc->Initialize(*viser_resc);

    swc_resc = std::make_unique<SWCRescPack>();


    swc2mesh_resc = std::make_unique<SWC2MeshRescPack>();


    vol_file_dialog.SetTitle("Volume Lod Desc Json File");
    vol_file_dialog.SetTypeFilters({".json"});
    swc_file_dialog.SetTitle("Volume SWC File");
    swc_file_dialog.SetTypeFilters({SWCFile::SWC_FILENAME_EXT_TXT, SWCFile::SWC_FILENAME_EXT_BIN});
    mesh_file_dialog.SetTitle("Neuron Mesh File");
    mesh_file_dialog.SetTypeFilters({MeshFile::MESH_FILENAME_EXT_OBJ});

    swc2mesh_group.start(2);
    render_group.start(2);

    status_flags |= VOL_DRAW_VOLUME | VOL_DRAW_SWC;
    if(vol_swc_blend_with_depth) status_flags |= VOL_SWC_VOLUME_BLEND_WITH_DEPTH;
}

// 初始化opengl相关
void VolAnnotaterGUI::initialize() {
    GL_EXPR(glEnable(GL_DEPTH_TEST));
    GL_EXPR(glEnable(GL_CULL_FACE));
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("EditorFont.TTF",  16, nullptr, nullptr);
    io.Fonts->Build();

//    Float3 default_pos = {4.1, 6.21, 7.f};
    Float3 default_pos = {2.1, 2.577f, 5.312f};
//    Float3 default_pos = {5.025f, 3.028f, 6.334f};
//    Float3 default_pos = {5.047, 5.859, 6.668};
    camera.set_position(default_pos);
    camera.set_perspective(FOV, 0.001f, 10.f);
    camera.set_direction(vutil::deg2rad(-90.f), 0.f);
    camera.set_move_speed(0.01);
    camera.set_view_rotation_speed(0.001f);

    quad_vao.initialize_handle();
    v2p_priv_data.v2p_shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("glsl/quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("glsl/view_to_proj_depth.frag"));
    v2p_priv_data.v2p_params_buffer.initialize_handle();
    v2p_priv_data.v2p_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    v2p_priv_data.v2p_params.fov = vutil::deg2rad(FOV);

    //需要创建OpenGL上下文后才能初始化
    swc_resc->Initialize();
    swc2mesh_resc->Initialize(*viser_resc);

    swc_resc->on_swc_selected = [this](MeshUID uid){
        swc2mesh_resc->Select(uid);
    };

//    viser_resc->render_gpu_mem_mgr_ref->_get_cuda_context()->set_ctx();

    init_ui_func();
}

void VolAnnotaterGUI::frame() {
    pre_render();

    handle_events();

    check_window_resize_event();

    framebuffer_t::bind_to_default();
    framebuffer_t::clear_color_depth_buffer();

    update_per_frame();

    show_editor_ui();

}

void VolAnnotaterGUI::destroy() {

}

void VolAnnotaterGUI::pre_render() {
    status_flags &= ~VOL_CAMERA_CHANGED;

    //vol需要最后画
    status_flags &= ~VOL_RENDER_PARAMS_CHANGED;

    vol_render_resc->vol_query_priv_data.clicked = false;

    vol_render_save_frame = false;
}

void VolAnnotaterGUI::handle_events() {
    //只在鼠标交互vol mesh render时更新相机
    if(!window_priv_data.vol_mesh_render_hovered) return;
    bool moved = keyboard->is_pressed('W') | keyboard->is_pressed('A')
                          | keyboard->is_pressed('D') | keyboard->is_pressed('S')
                          | keyboard->is_pressed(KEY_SPACE) | keyboard->is_pressed(KEY_LSHIFT);
    mouse->show_cursor(!moved);
    bool vol_camera_changed = false;
    if(moved || mouse->is_pressed(mouse_button_t::Mouse_Button_Right)){
        camera.update(
                {
                        .front = keyboard->is_pressed('W'),
                        .left  = keyboard->is_pressed('A'),
                        .right = keyboard->is_pressed('D'),
                        .back  = keyboard->is_pressed('S'),
                        .up    = keyboard->is_pressed(KEY_SPACE),
                        .down  = keyboard->is_pressed(KEY_LSHIFT),
                        .cursor_rel_x = static_cast<float>(mouse->get_delta_cursor_x()),
                        .cursor_rel_y = static_cast<float>(mouse->get_delta_cursor_y())
                });
        vol_camera_changed |= keyboard->is_pressed('W') | keyboard->is_pressed('A')
                              | keyboard->is_pressed('D') | keyboard->is_pressed('S')
                              | keyboard->is_pressed(KEY_SPACE) | keyboard->is_pressed(KEY_LSHIFT)
                              | (mouse->get_delta_cursor_x() != 0.0) | (mouse->get_delta_cursor_y() != 0.0);
    }
    camera.recalculate_matrics();

    vol_camera_changed |= window_priv_data.vol_render_resize;


    if(
//        is_annotating() &&
        mouse->is_pressed(mouse_button_t::Mouse_Button_Left)){
        auto [x, y] = ImGui::GetIO().MouseClickedPos[0];
        x = x - window_priv_data.vol_render_window_pos.x - 1;
        y = y - window_priv_data.vol_render_window_pos.y - 22;
        if(vol_render_resc->vol_query_priv_data.query_pos != vec2i(x, y)){
            vol_render_resc->vol_query_priv_data.query_pos = vec2i(x, y);
            vol_render_resc->vol_query_priv_data.clicked = true;
        }
    }
    if(vol_camera_changed)
        status_flags |= VOL_CAMERA_CHANGED;

    if(keyboard->is_pressed('P'))
        vol_render_save_frame = true;
}

void VolAnnotaterGUI::show_editor_ui() {
    Timer timer;
    timer.start();
    show_editor_menu(&editor_menu_window_open);
    timer.stop();
//    timer.print_duration("editor menu");
    timer.start();
    show_editor_vol_render_info_window(&vol_render_info_window_open);
    timer.stop();
//    timer.print_duration("editor vol render info window");
    timer.start();
    show_editor_vol_info_window(&vol_info_window_open);
    timer.stop();
//    timer.print_duration("editor vol info window");
    timer.start();
    show_editor_vol_render_window(&vol_render_window_open);
    timer.stop();
//    timer.print_duration("editor vol render window");
    timer.start();
    show_editor_mesh_render_info_window(&mesh_render_info_window_open);
    timer.stop();
//    timer.print_duration("editor mesh render info window");
    timer.start();
    show_editor_mesh_render_window(&mesh_render_window_open);
    timer.stop();
//    timer.print_duration("editor mesh render window");
    timer.start();
    show_editor_swc_window(&swc_info_window_open);
    timer.stop();
//    timer.print_duration("editor swc window");
    timer.start();

    show_editor_swc_op_window(&swc_op_window_open);

    show_editor_swc_tree_window(&swc_tree_window_open);
    timer.stop();
//    timer.print_duration("editor swc tree window");
    timer.start();
    show_editor_neuron_mesh_window(&neuron_mesh_window_open);
    timer.stop();
//    timer.print_duration("editor neuron mesh window");
    timer.start();
    show_smooth_mesh_window(&smooth_mesh_window_open);
    timer.stop();
//    timer.print_duration("smooth mesh window");

    show_swc_load_window(&swc_load_window_open);

    timer.start();
    show_debug_window(nullptr);
    timer.stop();
//    timer.print_duration("debug window");
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
        if(ImGui::BeginMenu("Edit")){
            if(ImGui::MenuItem(" Annotate ", nullptr, is_annotating(), can_start_annotating())){
                if(!is_annotating()){
                    if(check_and_start_annotating()){
                        status_flags |= VOL_ANNOTATING;
                    }
                }
                else{
                    if(stop_and_save_annotating()){
                        status_flags &= ~VOL_ANNOTATING;
                    }
                }
            }

            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Window")){
            ImGui::MenuItem("Vol Info", nullptr, &vol_info_window_open);
            ImGui::MenuItem("Render Info", nullptr, &vol_render_info_window_open);
            ImGui::MenuItem("Vol Render", nullptr, &vol_render_window_open);
            ImGui::MenuItem("Mesh Render", nullptr, &mesh_render_window_open);
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
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None | ImGuiWindowFlags_HorizontalScrollbar;

    if(p_open && !*p_open) return;

    static std::string buffer(256, '\0');
    if(ImGui::Begin("Vol Info", p_open, window_flags)){
        ImGui::NewLine();
        ImGui::InputText("Volume File", buffer.data(), buffer.length());
        if(ImGui::Button("Select", ImVec2(120, 18))){
            vol_file_dialog.Open();
        }
        ImGui::SameLine();
        if(ImGui::Button("Load", ImVec2(120, 18))){
            load_volume(buffer);
        }
        ImGui::NewLine();
        ImGui::BulletText("Volume Lod Levels: %d", viser_resc->vol_priv_data.max_lod + 1);
        ImGui::BulletText("Volume Space Ratio: %.2f %.2f %.2f",
                          viser_resc->vol_priv_data.volume_space_ratio.x,
                          viser_resc->vol_priv_data.volume_space_ratio.y,
                          viser_resc->vol_priv_data.volume_space_ratio.z);
        if(viser_resc->vol_priv_data.volume.IsValid()){
            auto vol_desc = viser_resc->vol_priv_data.volume->GetDesc();
            if(ImGui::TreeNode("Volume Desc")){
                ImGui::BulletText("Volume Name: %s", vol_desc.volume_name.c_str());
                ImGui::BulletText("Volume Dim: (%d, %d, %d)",
                                  vol_desc.shape.x, vol_desc.shape.y, vol_desc.shape.z);
                ImGui::BulletText("Voxel Space: (%.5f, %.5f, %.5f)",
                                  vol_desc.voxel_space.x, vol_desc.voxel_space.y, vol_desc.voxel_space.z);
                ImGui::BulletText("Samples Per Voxel: %d", vol_desc.samples_per_voxel);
                ImGui::BulletText("Bits Per Sample: %d", vol_desc.bits_per_sample);
                ImGui::BulletText("Voxel Is Float: %s", vol_desc.is_float ? "yes" : "no");
                ImGui::BulletText("Block Length: %d", vol_desc.block_length);
                ImGui::BulletText("Block Padding: %d", vol_desc.padding);
                ImGui::TreePop();
            }
        }
    }

    ImGui::End();

    vol_file_dialog.Display();
    if(vol_file_dialog.HasSelected()){
        std::cout << vol_file_dialog.GetSelected().string() << std::endl;
        buffer = vol_file_dialog.GetSelected().string();
        buffer.resize(256, '\0');
        vol_file_dialog.ClearSelected();
    }
}

void VolAnnotaterGUI::show_editor_vol_render_info_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Vol Render Info", p_open, window_flags)){
        ImGui::BulletText("Vol Render Frame Time: %s", vol_render_timer.duration_str().c_str());


        if(ImGui::TreeNode("Camera Setting")){
            auto pos = camera.get_position();
            ImGui::BulletText("Camera Pos %.3f %.3f %.3f", pos.x, pos.y, pos.z);
            auto dir = camera.get_xyz_direction();
            ImGui::BulletText("Camera Dir %.3f %.3f %.3f", dir.x, dir.y, dir.z);
            bool update = false;
            update |= ImGui::InputFloat("Move Speed", &vol_camera_move_speed, 0.f, 0.f, "%.5f");
            update |= ImGui::InputFloat("Rotate Speed", &vol_camera_view_ratation_speed, 0.f, 0.f, "%.5f");
            if(update){
                update_vol_camera_setting(false);
            }
            ImGui::TreePop();
        }
        if(ImGui::TreeNode("Vol Render Setting")){
            if(ImGui::Checkbox("Render Volume", &vol_render_volume)){
                if(vol_render_volume) status_flags |= VOL_DRAW_VOLUME;
                else status_flags &= ~VOL_DRAW_VOLUME;
            }

            if(ImGui::TreeNode("TransferFunc")){
                bool tf_update = false;

                static std::map<int, Float4> pt_mp;

                static Float3 color;
                static bool selected_pt = false;
                static int sel_pos;
                if(selected_pt){
                    color = pt_mp.at(sel_pos).xyz();
                }
                if(ImGui::ColorEdit3("Point Color(RGBA)", &color.x)){
                    if(selected_pt){
                        auto& c = pt_mp.at(sel_pos);
                        c.x = color.x;
                        c.y = color.y;
                        c.z = color.z;

                        tf_update = true;
                    }
                }



                ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
                const int ysize = 255;
                canvas_sz.y = ysize;
                ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
                ImGui::InvisibleButton("tf", canvas_sz);


                ImGuiIO& io = ImGui::GetIO();
                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
                draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(200, 200, 200, 255));

                const bool is_hovered = ImGui::IsItemHovered(); // Hovered
                const bool is_active = ImGui::IsItemActive();   // Held
                const ImVec2 origin(canvas_p0.x, canvas_p0.y); // Lock scrolled origin
                const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);
                const ImVec2 tf_origin(canvas_p0.x, canvas_p0.y + canvas_sz.y);

                bool check_add = false;
                if(is_active && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)){
                    check_add = true;
                }

                auto canvas_y_to_alpha = [&](float y){
                    return (ysize - y) / float(ysize);
                };
                auto alpha_to_canvas_y = [&](float alpha){
                    return ysize - alpha * ysize;
                };



                if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                    bool pick = false;
                    for(auto& [x_pos, color] : pt_mp){
                        if(std::abs(x_pos - mouse_pos_in_canvas.x) < 5
                            && std::abs(alpha_to_canvas_y(color.w) - mouse_pos_in_canvas.y) < 5){
                            selected_pt = true;
                            sel_pos = x_pos;
                            pick = true;
                            break;
                        }
                    }
                    if(!pick) selected_pt = false;
                }



                if(!selected_pt && check_add){
                    auto it = pt_mp.upper_bound(mouse_pos_in_canvas.x);
                    Float4 rgba;
                    rgba.w = canvas_y_to_alpha(mouse_pos_in_canvas.y);
                    if(it == pt_mp.end()){
                        auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                        if(itt == pt_mp.begin()){
                            rgba.x = rgba.y = rgba.z = 0.f;
                        }
                        else{
                            itt = std::prev(itt);
                            rgba.x = itt->second.x;
                            rgba.y = itt->second.y;
                            rgba.z = itt->second.z;
                        }
                    }
                    else{
                        auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                        if(itt == pt_mp.begin()){
                            rgba.x = it->second.x;
                            rgba.y = it->second.y;
                            rgba.z = it->second.z;
                        }
                        else{
                            itt = std::prev(itt);
                            float u = (mouse_pos_in_canvas.x - itt->first) / (float)(it->first - itt->first);
                            rgba.x = itt->second.x * (1.f - u) + it->second.x * u;
                            rgba.y = itt->second.y * (1.f - u) + it->second.y * u;
                            rgba.z = itt->second.z * (1.f - u) + it->second.z * u;
                        }
                    }
                    pt_mp[mouse_pos_in_canvas.x] = rgba;
                    selected_pt = true;
                    sel_pos = mouse_pos_in_canvas.x;
                    tf_update = true;
                }


                //add
                if(is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)
                    && selected_pt){
                    int nx = sel_pos + io.MouseDelta.x;
                    auto c = pt_mp.at(sel_pos);
                    int ny = alpha_to_canvas_y(c.w) + io.MouseDelta.y;
//                    LOG_DEBUG("ny : {}, delta y: {}", ny, io.MouseDelta.y);
                    ny = (std::min)(ny, ysize);
                    ny = (std::max)(ny, 0);
                    if(nx == sel_pos || pt_mp.count(nx) == 0){
                        c.w = canvas_y_to_alpha(ny);
                        pt_mp.erase(sel_pos);
                        sel_pos = nx;
                        pt_mp[nx] = c;
                        tf_update = true;
                    }
                }

                //delete
                if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Right)
                    && selected_pt){
                    selected_pt = false;
                    pt_mp.erase(sel_pos);
                    tf_update = true;
                }

                draw_list->PushClipRect(canvas_p0, canvas_p1, true);
                bool first = true;
                ImVec2 prev;
                if(!pt_mp.empty()){
                    auto it = pt_mp.begin();
                    ImVec2 p = ImVec2(it->first + origin.x, alpha_to_canvas_y(it->second.w) + origin.y);
                    draw_list->AddLine(ImVec2(origin.x, p.y), p, IM_COL32(0, 0, 0, 255));
                    auto itt = std::prev(pt_mp.end());
                    p = ImVec2(itt->first + origin.x, alpha_to_canvas_y(itt->second.w) + origin.y);
                    draw_list->AddLine(p, ImVec2(origin.x + canvas_sz.x, p.y), IM_COL32(0, 0, 0, 255));
                }
                for(auto& [x, c] : pt_mp){
                    ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                    if(first){
                        first = false;
                    }
                    else{
                        draw_list->AddLine(prev, cur, IM_COL32(0, 0, 0, 255));
                    }
                    prev = cur;
                }
                for(auto& [x, c] : pt_mp){
                    ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                    draw_list->AddCircleFilled(cur, 5.f,
                                               IM_COL32(int(c.x * 255), int(c.y * 255), int(c.z * 255), 255));
                    if(x == sel_pos && selected_pt){
                        draw_list->AddCircle(cur, 6.f, IM_COL32(255, 127, 0, 255), 0, 2.f);
                    }
                }
                draw_list->PopClipRect();

                ImGui::TreePop();

                if(tf_update){
                    std::vector<std::pair<float, Float4>> pts;
                    for(auto& [x, c] : pt_mp){
                        pts.emplace_back((float)x / (float)canvas_sz.x, c);
                    }
                    vol_render_resc->UpdateTransferFunc(pts);

                    status_flags |= VOL_RENDER_PARAMS_CHANGED;
                }
            }

            ImGui::TreePop();
        }
        if(ImGui::TreeNode("SWC Render Setting")){
            if(ImGui::Checkbox("Render SWC", &vol_render_swc)){
                if(vol_render_swc) status_flags |= VOL_DRAW_SWC;
                else status_flags &= ~VOL_DRAW_SWC;
            }

            if(ImGui::Checkbox("Blend With Depth", &vol_swc_blend_with_depth)){
                if(vol_swc_blend_with_depth) status_flags |= VOL_SWC_VOLUME_BLEND_WITH_DEPTH;
                else status_flags &= ~VOL_SWC_VOLUME_BLEND_WITH_DEPTH;
            }
            if(ImGui::Checkbox("Render SWC Point Tag", &vol_render_swc_point_tag)){
                if(vol_render_swc_point_tag) status_flags |= VOL_DRAW_SWC_TAG;
                else status_flags &= ~VOL_DRAW_SWC_TAG;
            }


            ImGui::TreePop();
        }
    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_vol_render_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;


    if(ImGui::Begin("Vol Render", p_open, window_flags)){
        auto [px, py] = ImGui::GetWindowPos();
        window_priv_data.vol_render_window_pos = vec2i(px, py);
        auto [x, y] = ImGui::GetWindowSize();
        y -= 20;
        if(window_priv_data.vol_render_window_size != vec2i(x, y)){
            window_priv_data.vol_render_resize = true;
            window_priv_data.vol_render_window_size = vec2i(x, y);
            ImGui::End();
            return;
        }
        ImGui::InvisibleButton("vol-render", ImVec2(x, y));
        window_priv_data.vol_mesh_render_hovered = ImGui::IsItemHovered();

         frame_vol_render();


    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_mesh_render_info_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Neuron Render Info", p_open, window_flags)){
        ImGui::NewLine();

        ImGui::Checkbox("Sync With Vol Render", &vol_mesh_render_sync);

        if(ImGui::TreeNode("Render Select")){

            if(ImGui::RadioButton("None", swc2mesh_resc->mesh_status == SWC2MeshRescPack::MeshStatus::None))
                swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::MeshStatus::None);
            if(ImGui::RadioButton("Merged", swc2mesh_resc->mesh_status == SWC2MeshRescPack::MeshStatus::Merged))
                swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::MeshStatus::Merged);
            if(ImGui::RadioButton("Blocked", swc2mesh_resc->mesh_status == SWC2MeshRescPack::MeshStatus::Blocked))
                swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::MeshStatus::Blocked);

            ImGui::TreePop();
        }

        bool update = false;
        update |= ImGui::SliderFloat("Light H Degree", &light_h_degree, 0.f, 360.f);
        update |= ImGui::SliderFloat("Light V Degree", &light_v_degree, -90.f, 90.f);
        update |= ImGui::ColorEdit3("Light Color", &light_color.x);
        update |= ImGui::InputFloat("Light Intensity", &light_intensity);
        if(update){
            Float3 light_dir;
            light_dir.y = std::sin(vutil::deg2rad(light_v_degree));
            light_dir.x = std::cos(vutil::deg2rad(light_h_degree)) * std::cos(vutil::deg2rad(light_v_degree));
            light_dir.z = std::sin(vutil::deg2rad(light_h_degree)) * std::cos(vutil::deg2rad(light_v_degree));
            swc2mesh_resc->neuron_renderer->Set(light_dir, light_color * light_intensity);
        }
    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_mesh_render_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Neuron Render", p_open, window_flags)){
        auto [px, py] = ImGui::GetWindowPos();
        window_priv_data.mesh_render_window_pos = vec2i(px, py);
        auto [x, y] = ImGui::GetWindowSize();
        y -= 20;
        if(window_priv_data.mesh_render_window_size != vec2i(x, y)){
            window_priv_data.mesh_render_resize = true;
            window_priv_data.mesh_render_window_size = vec2i(x, y);
            ImGui::End();
            return;
        }
//        AutoTimer t("frame mesh render");
        ImGui::InvisibleButton("mesh-render", ImVec2(x, y));
        window_priv_data.vol_mesh_render_hovered |= ImGui::IsItemHovered();
        frame_mesh_render();

    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_swc_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("SWC Info", p_open, window_flags)){
        ImGui::NewLine();

        //只可以在没有标注时进行swc的选择和切换

        std::string sel_swc_str;
        if (!swc_resc->loaded_swc.empty())
            sel_swc_str = swc_resc->loaded_swc.at(swc_resc->selected_swc_uid).name;
        else sel_swc_str = "None";
        if (ImGui::BeginCombo("Selected SWC", sel_swc_str.c_str())) {
            if(!is_annotating()){
                for (auto &[id, _]: swc_resc->loaded_swc) {
                    bool select = id == swc_resc->selected_swc_uid;

                    if (ImGui::Selectable(_.name.c_str(), select)) {
                        swc_resc->SelectSWC(id);
                        LOG_TRACE("Select SWC: {}, {}", id, _.name);
                        if(swc_resc->swc_mesh_mp.count(id))
                            swc2mesh_resc->Select(swc_resc->swc_mesh_mp.at(id));
                        update_swc_influenced_blocks();
                    }
                }

            }
            ImGui::EndCombo();
        }



        if(ImGui::Button("New SWC", ImVec2(120, 18))){
            swc_resc->CreateSWC();

            //新建一个swc的时候也创建对应的mesh
            //这里的代价是很低的 并没有体素化mc生成mesh 只用于记录状态
            std::string neuron_name = swc_resc->loaded_swc.at(swc_resc->selected_swc_uid).name
                    + "_neuron";
            swc2mesh_resc->CreateMesh(neuron_name);

            swc_resc->BindMeshToSWC(swc2mesh_resc->selected_mesh_uid);

            update_swc_influenced_blocks();
        }

        ImGui::SameLine();

        if(ImGui::Button("Delete SWC", ImVec2(120, 18))){
            swc_resc->DeleteSelSWC();
        }

        if(ImGui::Button("Load", ImVec2(120, 18))){
            swc_load_window_open = true;
//            swc_file_dialog.Open();
        }

        ImGui::SameLine();

        if(ImGui::Button("Export", ImVec2(120, 18))){
            if(swc_resc->Selected()){
                auto name = swc_resc->GetSelected().name;
                name += SWCFile::SWC_FILENAME_EXT_TXT;
                swc_resc->ExportSWCToFile(name);
            }
        }

        static ImGuiTableFlags table_flags =
                ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
                | ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti
                | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_NoBordersInBody
                | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
                | ImGuiTableFlags_SizingFixedFit;
        if(ImGui::BeginTable("SWC Points", 2, table_flags)){
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_NoSort |
            ImGuiTableColumnFlags_WidthFixed, 30);
            ImGui::TableSetupColumn("Tag PID X Y Z Radius");
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            if(!swc_resc->loaded_swc.empty()){
                auto& sel_swc = swc_resc->loaded_swc.at(swc_resc->selected_swc_uid).swc;
                auto sel_swc_pt_id = swc_resc->swc_priv_data.last_picked_swc_pt_id;
                static char buffer[64] = {'\0'};
//                const int max_display_item = 32;
//                int display_items = 0;
                for(auto it = sel_swc->begin(); it != sel_swc->end(); ++it){
//                    if(display_items++ > max_display_item) break;
                    ImGui::TableNextRow();
                    bool sel = it->first == sel_swc_pt_id;
                    ImGui::TableSetColumnIndex(0);
                    if(ImGui::Selectable(std::to_string(it->first).c_str(), sel)){
//                        swc_resc->swc_priv_data.last_picked_swc_pt_id = it->first;
                        swc_resc->AddPickedSWCPoint(it->first);
                    }
                    sprintf(buffer, "%d %d %.5f %.5f %.5f %.5f", it->second.tag, it->second.pid,
                            it->second.x, it->second.y, it->second.z, it->second.radius);
                    ImGui::TableSetColumnIndex(1);

                    ImGui::TextColored(sel ? ImVec4{1.f, 0.f, 0.f, 1.f} : ImVec4{1.f, 1.f ,1.f ,1.f}, "%s", buffer);
                }
            }

            ImGui::EndTable();
        }
    }

    ImGui::End();

//    swc_file_dialog.Display();
//    if(swc_file_dialog.HasSelected()){
//        swc_resc->LoadSWCFile(swc_file_dialog.GetSelected().string());
//        //第一次从文件中加载swc后 因为所有的swc点都是新插入的 因此需要更新受影响的block
//        swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::Blocked);
//
//        std::string neuron_name = swc_resc->loaded_swc.at(swc_resc->selected_swc_uid).name
//                                  + "_neuron";
//        swc2mesh_resc->CreateMesh(neuron_name);
//
//        swc_resc->BindMeshToSWC(swc2mesh_resc->selected_mesh_uid);
//
//        update_swc_influenced_blocks();
//
//        swc_file_dialog.ClearSelected();
//    }
}

void VolAnnotaterGUI::show_editor_swc_op_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("SWC OP", p_open, window_flags)){
        ImGui::NewLine();
        auto draw_swc_pt_ui = [&](SWCPointKey id){
            if(id > 0){
                auto& swc = swc_resc->GetSelected().swc;
                auto& pt = swc->GetNode(id);
                ImGui::BulletText("Selected SWC Point ID: %d", id);
                ImGui::BulletText("Parent ID: %d", pt.pid);
                ImGui::BulletText("Pos: %.5f %.5f %.5f", pt.x, pt.y, pt.z);
                ImGui::BulletText("Radius: %.5f", pt.radius);
            }
            else{
                ImGui::BulletText("No Selected SWC Point");
            }
        };

        if(swc_op == SWC_OP_Point){
            auto id = swc_resc->swc_priv_data.last_picked_swc_pt_id;
            draw_swc_pt_ui(id);
        }
        else{
            if(!swc_resc->swc_priv_data.picked_swc_pt_q.empty()){
                auto id1 = swc_resc->swc_priv_data.picked_swc_pt_q.back();
                auto id2 = swc_resc->swc_priv_data.picked_swc_pt_q.front();
                draw_swc_pt_ui(id1);
                ImGui::NewLine();
                draw_swc_pt_ui(id2);
            }
        }

        ImGui::NewLine();

        if(ImGui::BeginTabBar("SWC-OPs")){

            if(ImGui::BeginTabItem("Point")){
                swc_op = SWC_OP_Point;

                swc_resc->swc_renderer->ClearVertex();

                swc_resc->ClearPickedSWCSegmentPoints();

                swc_resc->SetSWCPointPickSize(1);

                ImGui::NewLine();

                static const char* PointOperations[] = {
                  "UpdateR", "Delete"
                };
                static int pt_op_idx = 0;
                constexpr int pt_op_cnt = sizeof(PointOperations) / sizeof(PointOperations[0]);
                if(ImGui::BeginCombo("Operations", PointOperations[pt_op_idx])){

                    for(int i = 0; i < pt_op_cnt; i++){
                        bool select = i == pt_op_idx;
                        if(ImGui::Selectable(PointOperations[i], select)){
                            pt_op_idx = i;
                        }
                    }

                    ImGui::EndCombo();
                }

                if(pt_op_idx == 0){
                    //UpdateR
                    swc_op_ui.at(SWC_OP_Point_UpdateR)();
                }
                else if(pt_op_idx == 1){
                    //Delete : delete point and connect two sides or delete subtree
                    swc_op_ui.at(SWC_OP_Point_Delete)();
                }



                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Segment")){
                swc_op = SWC_OP_Segment;

                swc_resc->SetSWCPointPickSize(2);

                ImGui::NewLine();

                static const char* SegmentOperations[] = {
                  "InterpR", "Delete", "Add"
                };
                static int seg_op_idx = 0;
                constexpr int seg_op_cnt = sizeof(SegmentOperations) / sizeof(SegmentOperations[0]);
                if(ImGui::BeginCombo("Operations", SegmentOperations[seg_op_idx])){

                    for(int i = 0; i < seg_op_cnt; i++){
                        bool select = i == seg_op_idx;
                        if(ImGui::Selectable(SegmentOperations[i], select)){
                            seg_op_idx = i;
                        }
                    }

                    ImGui::EndCombo();
                }
                if(seg_op_idx == 0){
                    swc_op_ui.at(SWC_OP_Seg_InterpR)();
                }
                else if(seg_op_idx == 1){
                    swc_op_ui.at(SWC_OP_Seg_Delete)();
                }
                else if(seg_op_idx == 2){
                    swc_op_ui.at(SWC_OP_Seg_Add)();
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_swc_tree_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("SWC Tree View", p_open, window_flags)){
        //todo use GetCursorScreenPos and GetContentRegionAvail
        auto [px, py] = ImGui::GetWindowPos();
        window_priv_data.swc_view_window_pos = vec2i(px, py);
        auto [x, y] = ImGui::GetWindowSize();
        y -= 20;
        if(window_priv_data.swc_view_window_size != vec2i(x, y)){
            window_priv_data.swc_view_resize = true;
            window_priv_data.swc_view_window_size = vec2i(x, y);
            ImGui::End();
            return;
        }
        static ImVec2 scrolling(0.0f, 0.0f);
        static bool opt_enable_grid = true;
        static bool opt_enable_context_menu = true;
        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
        ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
        ImGui::InvisibleButton("swc-tree-view",
                               ImVec2(x,y),
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
//        ImGui::SetItemUsingMouseWheel();
        ImGuiIO& io = ImGui::GetIO();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
//        draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

        const bool is_hovered = ImGui::IsItemHovered(); // Hovered
        const bool is_active = ImGui::IsItemActive();   // Held
        const ImVec2 origin(canvas_p0.x + scrolling.x, canvas_p0.y + scrolling.y); // Lock scrolled origin
        const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);


        const float mouse_threshold_for_pan = opt_enable_context_menu ? -1.0f : 0.0f;
        if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
        {
            scrolling.x += io.MouseDelta.x;
            scrolling.y += io.MouseDelta.y;
        }
        bool selecting = false;
        if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
            LOG_DEBUG("mouse left clicked: {} {}", io.MouseClickedPos[0].x, io.MouseClickedPos[0].y);
            selecting = true;
        }

        draw_list->PushClipRect(canvas_p0, canvas_p1, true);

        if (opt_enable_grid)
        {
            const float GRID_STEP = 64.0f;
            for (float x = fmodf(scrolling.x, GRID_STEP); x < canvas_sz.x; x += GRID_STEP){
                draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
            }
            for (float y = fmodf(scrolling.y, GRID_STEP); y < canvas_sz.y; y += GRID_STEP){
                draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
            }
        }

        const auto& draw_lines = swc_resc->swc_priv_data.swc_draw_tree.draw_lines;
        static float scale = 10000.f;
        static float level_height = 50.f;

        if(is_hovered && io.MouseWheel != 0.f){
            scale += io.MouseWheel < 0.f ? 1000.f : -1000.f;
            scale = (std::max)(1000.f, scale);
            scale = (std::min)(100000.f, scale);
            level_height += io.MouseWheel < 0.f ? 5.f : -5.f;
            level_height = (std::max)(5.f, level_height);
            level_height = (std::min)(500.f, level_height);
        }

        for(const auto& line : draw_lines){
            draw_list->AddLine(ImVec2(origin.x + line.first.x * scale, origin.y + line.first.y * level_height + canvas_sz.y * 0.5f),
                               ImVec2(origin.x + line.second.x * scale, origin.y + line.second.y * level_height + canvas_sz.y * 0.5f),
                               IM_COL32(255, 0, 0, 255),
                               2.f);

        }
        const auto& draw_pts = swc_resc->swc_priv_data.swc_draw_tree.draw_points;

        for(const auto& pt : draw_pts){
            auto px = origin.x + pt.pos.x * scale;
            auto py = origin.y + pt.pos.y * level_height + canvas_sz.y * 0.5f;
            if(selecting && std::abs(px - io.MouseClickedPos[0].x) < 3 && std::abs(py - io.MouseClickedPos[0].y) < 3){

               LOG_DEBUG("select swc pt : {}", pt.pt_id);
               swc_resc->AddPickedSWCPoint(pt.pt_id);

               if(swc_op == SWC_OP_Segment){
                   swc_resc->UpdatePickedSWCSegmentPoints();

                   swc_resc->AddPickedSWCSegPtsToRenderer();

               }
            }
            if(std::abs(px - io.MousePos.x) < 3 && std::abs(py - io.MousePos.y) < 3){
                ImGui::SetTooltip("ID %d, Radius %.5f", pt.pt_id, swc_resc->GetSelected().swc->GetNode(pt.pt_id).radius);
            }
            bool selected = pt.pt_id == swc_resc->swc_priv_data.picked_swc_pt_q.front()
                || pt.pt_id == swc_resc->swc_priv_data.picked_swc_pt_q.back();

            bool last_selected = pt.pt_id == swc_resc->swc_priv_data.picked_swc_pt_q.back();

            if(swc_op == SWC_OP_Segment && !selected){
                bool inside_seg = swc_resc->swc_priv_data.swc_draw_tree.PointInsideSeg(pt.pt_id);
                draw_list->AddCircleFilled(ImVec2(px, py),
                                           scale <= 3000.f ? 3.f : 5.f,
                                           inside_seg ? IM_COL32(127, 127, 127, 255) : IM_COL32(255, 255, 255, 255));
            }
            else
                draw_list->AddCircleFilled(ImVec2(px, py),
                                       scale <= 3000.f ? 3.f : 5.f,
                                       selected ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 255, 255, 255));
            if(last_selected)
                draw_list->AddCircle(ImVec2(px, py),
                                           (scale <= 3000.f ? 3.f : 5.f) + 1.f,
                                           IM_COL32(255, 0, 0, 255));

        }

        draw_list->PopClipRect();
    }

    ImGui::End();
}

void VolAnnotaterGUI::show_editor_neuron_mesh_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Neuron Mesh Info", p_open, window_flags)){
        ImGui::NewLine();

        std::string sel_mesh_str = "None";
        if(swc2mesh_resc->Selected())
            sel_mesh_str = swc2mesh_resc->loaded_mesh.at(swc2mesh_resc->selected_mesh_uid).name;
        if(ImGui::BeginCombo("Selected Neuron", sel_mesh_str.c_str())){
            for(auto& [id, _] : swc2mesh_resc->loaded_mesh){
                bool select = id == swc2mesh_resc->selected_mesh_uid;
                if(ImGui::Selectable(_.name.c_str(), select)){
                    swc2mesh_resc->Select(id);

                    LOG_TRACE("Select Neuron Mesh : {}, {}", id, _.name);
                }
            }

            ImGui::EndCombo();
        }
        if(ImGui::Button("New Neuron", ImVec2(120, 18))){
            swc2mesh_resc->CreateMesh();
        }
        ImGui::SameLine();
        if(ImGui::Button("Delete Neuron", ImVec2(120, 18))){

        }

        if(ImGui::Button("Load", ImVec2(120, 18))){
            mesh_file_dialog.Open();
        }
        ImGui::SameLine();
        if(ImGui::Button("Export", ImVec2(120, 18))){
            static int export_count = 0;
            std::string export_filename;
            auto name = swc2mesh_resc->GetSelected().name;
            if(name.empty())
                export_filename = "neuron_mesh_" + std::to_string(++export_count) + MeshFile::MESH_FILENAME_EXT_OBJ;
            else
                export_filename = name + MeshFile::MESH_FILENAME_EXT_OBJ;
            swc2mesh_resc->ExportMeshToFile(export_filename);
        }
        if(ImGui::Button("Re-Generate-Modified", ImVec2(160, 18))){
            generate_modified_mesh();

        }
        ImGui::SameLine();
        //这个用于全部swc点的mesh生成 但是生成的还是blocked 之后整体的使用异步队列实现
        if(ImGui::Button("Re-Generate-All", ImVec2(160, 18))){
            //把当前的block mesh全部设置为Modified
            update_swc_influenced_blocks(true);
            generate_modified_mesh();
        }

        if(ImGui::Button("Merge All", ImVec2(120, 18))){

            swc2mesh_resc->MergeAllBlockMesh();

        }
        ImGui::SameLine();
        if(ImGui::Button("Smooth Mesh", ImVec2(120, 18))){
            smooth_mesh_window_open = !smooth_mesh_window_open;
        }



        ImGuiTableFlags table_flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
                                      | ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti
                                      | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_NoBordersInBody
                                      | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
                                      | ImGuiTableFlags_SizingFixedFit;
        if(ImGui::BeginTable("Neuron Meshes", 3, table_flags)){
            ImGui::TableSetupColumn("Visible", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, 0.f);
            ImGui::TableSetupColumn("BlockUID");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            int idx = 0;
            for(auto& [uid, block_mesh] : swc2mesh_resc->s2m_priv_data.patch_mesh_mp){
                ImGui::PushID(idx++);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Checkbox("", &block_mesh.visible);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored({0.f, 1.f, 0.f, 1.f}, "%d %d %d", uid.x, uid.y, uid.z);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", ToString(block_mesh.status).c_str());
                ImGui::PopID();
            }

            ImGui::EndTable();
        }
    }

    ImGui::End();

    mesh_file_dialog.Display();
    if(mesh_file_dialog.HasSelected()){
        auto filename = mesh_file_dialog.GetSelected().string();

        swc2mesh_resc->LoadMeshFile(filename);

        mesh_file_dialog.ClearSelected();
    }
}

void VolAnnotaterGUI::show_smooth_mesh_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Mesh Smoothing Settings", p_open, window_flags)){
        ImGui::SliderFloat("lambda", &mesh_smooth_priv_data.lambda, 0.f, 1.f);
        ImGui::SliderFloat("mu", &mesh_smooth_priv_data.mu, -1.f, 0.f);
        ImGui::SliderInt("iterations", &mesh_smooth_priv_data.iterations, 1, 1000);
        if(ImGui::Button("Smoothing", ImVec2(120, 18))){
            *p_open = false;
            //这里不用多线程了 否则需要切换OpenGL Context
            swc2mesh_resc->SmoothMesh(mesh_smooth_priv_data.lambda,
                                      mesh_smooth_priv_data.mu,
                                      mesh_smooth_priv_data.iterations);
        }
    }
    ImGui::End();
}

void VolAnnotaterGUI::show_swc_load_window(bool *p_open)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if(p_open && !*p_open) return;
    static char swc_filename[256] = {'\0'};
    if(ImGui::Begin("SWC Load", p_open, window_flags)){

        ImGui::InputText("SWC File", swc_filename, 256);
        static Float3 s = Float3(1.f);
        ImGui::InputFloat3("Transform ratio", &s.x, "%.5f");
        if(ImGui::Button("Select", ImVec2(120, 18))){
            swc_file_dialog.Open();
        }
        ImGui::SameLine();
        if(ImGui::Button("Load")){
            swc_resc->LoadSWCFile(std::string(swc_filename), s);
            //第一次从文件中加载swc后 因为所有的swc点都是新插入的 因此需要更新受影响的block
            swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::Blocked);

            std::string neuron_name = swc_resc->loaded_swc.at(swc_resc->selected_swc_uid).name
                                      + "_neuron";
            swc2mesh_resc->CreateMesh(neuron_name);

            swc_resc->BindMeshToSWC(swc2mesh_resc->selected_mesh_uid);

            update_swc_influenced_blocks();

            swc_file_dialog.ClearSelected();

            *p_open = false;
        }
    }
    ImGui::End();

    swc_file_dialog.Display();
    if(swc_file_dialog.HasSelected()){
//        swc_resc->LoadSWCFile(swc_file_dialog.GetSelected().string());

        std::memcpy(swc_filename, swc_file_dialog.GetSelected().string().c_str(),
                    swc_file_dialog.GetSelected().string().length());

    }
}

void VolAnnotaterGUI::show_debug_window(bool *p_open) {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;

    if(p_open && !*p_open) return;

    if(ImGui::Begin("Debug Info", p_open, window_flags)){
        if(ImGui::TreeNode("App Settings")){
            ImGui::BulletText("MaxHostMem: %.2f GB", AppSettings::MaxHostMemBytes / double(1ull << 30));
            ImGui::BulletText("MaxRenderGPUMem: %.2f GB", AppSettings::MaxRenderGPUMemBytes / double(1ull << 30));
            ImGui::BulletText("MaxRenderGPUMem: %.2f GB", AppSettings::MaxComputeGPUMemBytes / double(1ull << 30));
            ImGui::BulletText("RenderGPUIndex: %d", AppSettings::RenderGPUIndex);
            ImGui::BulletText("ComputeGPUIndex: %d", AppSettings::ComputeGPUIndex);
            ImGui::BulletText("MaxFixedHostMem: %.2f", AppSettings::MaxFixedHostMemBytes / double(1ull << 30));
            ImGui::BulletText("ThreadsGroupWorkerCount: %d", AppSettings::ThreadsGroupWorkerCount);
            ImGui::BulletText("VTexture Count: %d", AppSettings::VTexCount);
            ImGui::BulletText("VTexture Shape: (%d, %d, %d)", AppSettings::VTexShape.x, AppSettings::VTexShape.y, AppSettings::VTexShape.z);


            ImGui::TreePop();
        }
        if(ImGui::TreeNode("Timer")){
            ImGui::BulletText("App FPS: %.2f", ImGui::GetIO().Framerate);

            ImGui::TreePop();
        }

        if(ImGui::TreeNode("Mesh Render")){
            ImGui::Checkbox("Line Mode", &debug_priv_data.mesh_render_line_mode);
            ImGui::TreePop();
        }

    }

    ImGui::End();
}

void VolAnnotaterGUI::frame_vol_render() {
    auto w = vol_render_resc->framebuffer->frame_width;
    auto h = vol_render_resc->framebuffer->frame_height;
    cudaGraphicsResource_t rescs[2] = {vol_render_cudaGL_interop.cuda_frame_color_resc,
                                       vol_render_cudaGL_interop.cuda_frame_depth_resc};
    CUB_CHECK(cudaGraphicsMapResources(2, rescs));
    void *color_mapping_ptr = nullptr;
    size_t color_mapping_size = 0;
    CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&color_mapping_ptr, &color_mapping_size,
                                                   vol_render_cudaGL_interop.cuda_frame_color_resc));
    assert(color_mapping_ptr && color_mapping_size == w* h * sizeof(uint32_t));
    void *depth_mapping_ptr = nullptr;
    size_t depth_mapping_size = 0;
    CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&depth_mapping_ptr, &depth_mapping_size,
                                                   vol_render_cudaGL_interop.cuda_frame_depth_resc));
    assert(depth_mapping_ptr &&
                   depth_mapping_size == w * h * sizeof(float));

    vol_render_resc->framebuffer->color = CUDABufferView2D<uint32_t>(color_mapping_ptr,
                                                                     {.pitch = w * sizeof(uint32_t),
                                                                      .xsize = (size_t) w, .ysize = (size_t) h});
    vol_render_resc->framebuffer->depth = CUDABufferView2D<float>(depth_mapping_ptr,
                                                                  {.pitch = w * sizeof(float),
                                                                          .xsize = (size_t) w, .ysize = (size_t) h});


    auto render_vol_frame = [&]()->bool{
        return viser_resc->vol_priv_data.volume.IsValid()
               && ((status_flags & VOL_CAMERA_CHANGED) || (status_flags & VOL_RENDER_PARAMS_CHANGED))
               && (status_flags & VOL_DRAW_VOLUME);
    };
    auto query_vol = [&]()->bool{
        return viser_resc->vol_priv_data.volume.IsValid()
               && (status_flags & VOL_DRAW_VOLUME) && is_annotating() && vol_render_resc->vol_query_priv_data.clicked;
    };
    auto pick_swc_pt = [&]()->bool{
        if(!(vol_render_resc->vol_query_priv_data.clicked)) return false;

        return pick_swc_point();
    };
    if(render_vol_frame()) {
        render_volume();
    }
    else if(pick_swc_pt()){
        // deal with picked swc point
        process_picked_swc_point();
    }
    else if(query_vol()){
        if(query_volume()){
            check_and_add_swc_pt();
        }
    }

    CUB_CHECK(cudaGraphicsUnmapResources(2, rescs));

    //将cudaGL interop资源拷贝到OpenGL纹理中
    vol_render_cudaGL_interop.color_pbo.bind();
    vol_swc_render_framebuffer.color.set_texture_data<color4b>(w, h, nullptr);
    vol_render_cudaGL_interop.color_pbo.unbind();

    vol_render_cudaGL_interop.depth_pbo.bind();
    vol_swc_render_framebuffer.depth.set_texture_data<GLfloat>(w, h, nullptr);
    vol_render_cudaGL_interop.depth_pbo.unbind();

    GL_EXPR(glFinish());


    if(status_flags & VOL_DRAW_SWC){
//        AutoTimer timer("render swc");
        render_swc();
    }

    auto draw_list = ImGui::GetWindowDrawList();

    draw_list->AddImage((void*)(intptr_t)(vol_swc_render_framebuffer.color.handle()),
                 ImVec2(window_priv_data.vol_render_window_pos.x + 1, window_priv_data.vol_render_window_pos.y + 22),
                 ImVec2(window_priv_data.vol_render_window_pos.x + w + 1, window_priv_data.vol_render_window_pos.y + h + 22));

    if(vol_render_save_frame){
        auto w = vol_swc_render_framebuffer.frame_width;
        auto h = vol_swc_render_framebuffer.frame_height;
        vutil::image2d_t<vutil::color4b> ret(w, h);
        GL_EXPR(glGetTextureImage(vol_swc_render_framebuffer.color.handle(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                                  w * h * sizeof(vutil::color4b),
                                  ret.get_raw_data()));
        GL_EXPR(glFinish());
        vutil::save_rgb_to_png_file("vol_render_snapshot.png", ret.get_data().map([](vutil::color4b c){
            return vutil::color3b(c.r, c.g, c.b);
        }));
    }
}

void VolAnnotaterGUI::frame_mesh_render() {
    mat4 view, proj;
    auto w = mesh_render_framebuffer.frame_width;
    auto h = mesh_render_framebuffer.frame_height;
    if(vol_mesh_render_sync){
        auto old = camera.get_w_over_h();
        auto wh = (float)mesh_render_framebuffer.frame_width / (float)mesh_render_framebuffer.frame_height;
        camera.set_w_over_h(wh);
        view = camera.get_view();
        proj = camera.get_proj();
        camera.set_w_over_h(old);
    }

    if(swc2mesh_resc->DrawMesh()){
//        AutoTimer t("draw mesh");
        mesh_render_framebuffer.fbo.bind();
        GL_EXPR(glViewport(0, 0, w, h));
        mesh_render_framebuffer.fbo.attach(GL_COLOR_ATTACHMENT0, mesh_render_framebuffer.color);
        vutil::gl::framebuffer_t::clear_color_depth_buffer();



        if(debug_priv_data.mesh_render_line_mode){ GL_EXPR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)); }

        swc2mesh_resc->neuron_renderer->Begin(view, proj, camera.get_position());
        if(swc2mesh_resc->mesh_status == SWC2MeshRescPack::Merged)
            swc2mesh_resc->neuron_renderer->Draw(swc2mesh_resc->loaded_mesh.at(swc2mesh_resc->selected_mesh_uid).mesh->GetUID());
        else
            for(auto& [id, _] : swc2mesh_resc->s2m_priv_data.patch_mesh_mp){
                if(_.visible && _.status != SWC2MeshRescPack::Empty)
                    swc2mesh_resc->neuron_renderer->Draw(id.ToUnifiedRescUID());
            }
        swc2mesh_resc->neuron_renderer->End();

        if(debug_priv_data.mesh_render_line_mode) { GL_EXPR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); }

        mesh_render_framebuffer.fbo.unbind();

        auto draw_list = ImGui::GetWindowDrawList();
        draw_list->AddImage((void*)(intptr_t)(mesh_render_framebuffer.color.handle()),
                            ImVec2(window_priv_data.mesh_render_window_pos.x + 1, window_priv_data.mesh_render_window_pos.y + 22),
                            ImVec2(window_priv_data.mesh_render_window_pos.x + w + 1, window_priv_data.mesh_render_window_pos.y + h + 22));

    }
}

void VolAnnotaterGUI::register_cuda_gl_interop_resource() {
    // 体绘制的framebuffer改变大小后重新注册
    auto w = window_priv_data.vol_render_window_size.x;
    auto h = window_priv_data.vol_render_window_size.y;
    vol_render_resc->framebuffer->frame_width = w;
    vol_render_resc->framebuffer->frame_height = h;
    size_t pixel_count = w * h;
    vol_render_cudaGL_interop.color_pbo.destroy();
    vol_render_cudaGL_interop.color_pbo.initialize_handle();
    vol_render_cudaGL_interop.color_pbo.reinitialize_buffer_data(nullptr, pixel_count * sizeof(uint32_t), GL_DYNAMIC_COPY);
    if(vol_render_cudaGL_interop.cuda_frame_color_resc != nullptr){
        CUB_CHECK(cudaGraphicsUnregisterResource(vol_render_cudaGL_interop.cuda_frame_color_resc));
        vol_render_cudaGL_interop.cuda_frame_color_resc = nullptr;
    }
    CUB_CHECK(cudaGraphicsGLRegisterBuffer(&vol_render_cudaGL_interop.cuda_frame_color_resc, vol_render_cudaGL_interop.color_pbo.handle(),
                                           cudaGraphicsRegisterFlagsWriteDiscard));

    vol_render_cudaGL_interop.depth_pbo.destroy();
    vol_render_cudaGL_interop.depth_pbo.initialize_handle();
    vol_render_cudaGL_interop.depth_pbo.reinitialize_buffer_data(nullptr, pixel_count * sizeof(float), GL_DYNAMIC_COPY);
    if(vol_render_cudaGL_interop.cuda_frame_depth_resc != nullptr){
        CUB_CHECK(cudaGraphicsUnregisterResource(vol_render_cudaGL_interop.cuda_frame_depth_resc));
        vol_render_cudaGL_interop.cuda_frame_depth_resc = nullptr;
    }
    CUB_CHECK(cudaGraphicsGLRegisterBuffer(&vol_render_cudaGL_interop.cuda_frame_depth_resc, vol_render_cudaGL_interop.depth_pbo.handle(),
                                           cudaGraphicsRegisterFlagsWriteDiscard));

}

void VolAnnotaterGUI::render_volume() {

    auto camera_proj_view = camera.get_view_proj();
    static Frustum camera_view_frustum;
    ExtractFrustumFromMatrix(camera_proj_view, camera_view_frustum);
    auto& intersect_blocks = vol_render_resc->vol_render_priv_data.intersect_blocks;
    intersect_blocks.clear();

    ComputeIntersectedBlocksWithViewFrustum(intersect_blocks,
                                            vol_render_resc->render_vol.lod0_block_length_space,
                                            vol_render_resc->render_vol.lod0_block_dim,
                                            vol_render_resc->render_vol.volume_bound,
                                            camera_view_frustum,
                                            [max_lod = viser_resc->vol_priv_data.max_lod,
                                             pos = camera.get_position(),
                                             this]
                                                    (const BoundingBox3D& box)->int{
                                                auto center = (box.low + box.high) * 0.5f;
                                                float dist = (center - pos).length();
                                                for(int i = 0; i <= max_lod; i++){
                                                    if(dist < this->vol_render_resc->lod.LOD[i])
                                                        return i;
                                                }
                                                return max_lod;
                                            });

    // 加载缺失的数据块到虚拟纹理中
    auto& blocks_info = vol_render_resc->vol_render_priv_data.blocks_info;
    blocks_info.clear();
    viser_resc->gpu_pt_mgr_ref->GetAndLock(intersect_blocks, blocks_info);

    //因为是单机同步的，不需要加任何锁
    // 暂时使用同步等待加载完数据块
    auto& host_blocks = vol_render_resc->vol_render_priv_data.host_blocks;//在循环结束会释放Handle
    host_blocks.clear();
    auto& missed_host_blocks = vol_render_resc->vol_render_priv_data.missed_host_blocks;
    missed_host_blocks.clear();
    for(auto& block : blocks_info){
        if(!block.second.Missed()) continue;
        auto block_hd = viser_resc->host_block_pool_ref->GetBlock(block.first.ToUnifiedRescUID());
        LOG_DEBUG("ok111 {}", block_hd.GetUID());
        if(block.first.IsSame(block_hd.GetUID())){
            LOG_DEBUG("ok222");
            host_blocks[block.first] = std::move(block_hd);
            LOG_DEBUG("okok222");
        }
        else{
            LOG_DEBUG("ok333");
            block_hd.SetUID(block.first.ToUnifiedRescUID());
            missed_host_blocks[block.first] = std::move(block_hd);
            LOG_DEBUG("okok333");
        }
    }

    // 解压数据块
    // 这里先加载lod小的数据块，但是在同步下是没关系的，只有在异步加载的时候有意义
    auto& task_mp = vol_render_resc->vol_render_priv_data.task_mp;
    task_mp.clear();
    for(auto& missed_block : missed_host_blocks){
        int lod = missed_block.first.GetLOD();
        task_mp[lod].emplace_back(
                [this, block = missed_block.first,
                        block_handle = std::move(missed_block.second)
                ]()mutable{
                    this->viser_resc->vol_priv_data.volume->ReadBlock(block, *block_handle);
                    if constexpr (false){
                        std::vector<int> table(256, 0);
                        auto v = block_handle->view_1d<uint8_t>(256 * 256 * 256);
                        for(int i = 0; i < 1 << 24; i++){
                            table[v.at(i)]++;
                        }
//                        for(int i = 0; i < 256; i++)
//                            std::cout << "(" << i << "," << table[i] << ")  ";
//                        std::cout << std::endl;
                    }
                    block_handle.SetUID(block.ToUnifiedRescUID());
                    this->vol_render_resc->vol_render_priv_data.host_blocks[block] = std::move(block_handle);
                    LOG_DEBUG("finish lod {} block ({}, {}, {}) loading...",
                              block.GetLOD(), block.x, block.y, block.z);
                });
    }
    auto& task_groups = vol_render_resc->vol_render_priv_data.task_groups;
    task_groups.clear();
    auto& lods = vol_render_resc->vol_render_priv_data.lods;
    lods.clear();
    for(auto& task : task_mp){
        int count = task.second.size();
        int lod = task.first;
        auto& tasks = task.second;
        assert(count > 0);
        lods.emplace_back(lod);
        auto task_group = viser_resc->thread_group.create_task(std::move(tasks.front()));
        for(int i = 1; i < count; i++){
            viser_resc->thread_group.enqueue_task(*task_group, std::move(tasks[i]));
        }
        task_groups[lod] = std::move(task_group);
    }
    int lod_count = lods.size();
    for(int i = 0; i < lod_count - 1; i++){
        int first = lods[i], second = lods[i + 1];
        viser_resc->thread_group.add_dependency(*task_groups[second], *task_groups[first]);
    }
    for(auto& [_, task_group] : task_groups){
        viser_resc->thread_group.submit(task_group);
    }
    //同步，等待所有解压任务完成
    viser_resc->thread_group.wait_idle();


    //将数据块上传到虚拟纹理
    for(auto& missed_block : blocks_info){
        if(!missed_block.second.Missed()) continue;
        auto& handle = host_blocks[missed_block.first];
        //这部分已经在CPU的数据块，调用异步memcpy到GPU
        viser_resc->gpu_vtex_mgr_ref->UploadBlockToGPUTexAsync(handle, missed_block.second);
    }

    viser_resc->gpu_vtex_mgr_ref->Flush();


    vol_render_resc->crt_vol_renderer->BindPTBuffer(viser_resc->gpu_pt_mgr_ref->GetPageTable().GetHandle());


    //更新每一帧绘制的参数
    auto& per_frame_params = vol_render_resc->vol_render_priv_data.per_frame_params;
    update_vol_render_per_frame_params(per_frame_params);
    vol_render_resc->crt_vol_renderer->SetPerFrameParams(per_frame_params);

    vol_render_timer.start();
    vol_render_resc->crt_vol_renderer->Render(vol_render_resc->framebuffer);
    vol_render_timer.stop();

    viser_resc->gpu_pt_mgr_ref->Release(intersect_blocks);
}

bool VolAnnotaterGUI::query_volume() {
    auto w = window_priv_data.vol_render_window_size.x;
    auto h = window_priv_data.vol_render_window_size.y;
    auto [x, y] = vol_render_resc->vol_query_priv_data.query_pos;
    if(x < 0 || x >= w || y < 0 || y > h){
        LOG_TRACE("query pos out of frame: {} {} for frame size {} {}", x, y, w, h);
        return false;
    }

    vol_render_resc->crt_vol_renderer->Query(x, y, vol_render_resc->vol_query_priv_data.query_info_view);


    return true;
}

void VolAnnotaterGUI::load_volume(const std::string &filename) {
    LOG_TRACE("load_volume start...");
    LOG_TRACE("load volume filename : {}", filename.c_str());

    viser_resc->LoadVolume(filename);

    vol_render_resc->OnVolumeLoaded(*viser_resc);

    swc2mesh_resc->OnVolumeLoaded(*viser_resc, *vol_render_resc);

    status_flags |= VOL_CAMERA_CHANGED;

    status_flags |= VOL_RENDER_PARAMS_CHANGED;

    status_flags |= VOL_DATA_LOADED;

    update_vol_camera_setting(true);

    LOG_TRACE("load_volume finish...");
}

void VolAnnotaterGUI::update_per_frame() {


}

void VolAnnotaterGUI::update_vol_render_per_frame_params(PerFrameParams& params) {
    params.frame_width = vol_swc_render_framebuffer.frame_width;
    params.frame_height = vol_swc_render_framebuffer.frame_height;
    params.frame_w_over_h = (float)params.frame_width / (float)params.frame_height;
    params.fov = vutil::deg2rad(camera.get_fov_deg());
    params.cam_pos = camera.get_position();
    params.cam_dir = camera.get_xyz_direction();
    static Float3 WorldUp = {0.f, 1.f, 0.f};
    params.cam_right = vutil::cross(camera.get_xyz_direction(), WorldUp).normalized();
    params.cam_up = vutil::cross(params.cam_right, params.cam_dir);
}

void VolAnnotaterGUI::on_vol_render_window_resize() {
    create_vol_render_gl_framebuffer_resource();
    register_cuda_gl_interop_resource();
    auto w_over_h = (float)window_priv_data.vol_render_window_size.x / (float)window_priv_data.vol_render_window_size.y;
    camera.set_w_over_h(w_over_h);
    v2p_priv_data.v2p_params.w_over_h = w_over_h;
}

void VolAnnotaterGUI::on_mesh_render_window_resize() {
    create_mesh_render_gl_framebuffer_resource();

}

void VolAnnotaterGUI::on_swc_view_window_resize() {

}

void VolAnnotaterGUI::check_window_resize_event() {
    if(window_priv_data.vol_render_resize){
        on_vol_render_window_resize();
        window_priv_data.vol_render_resize = false;
    }
    if(window_priv_data.mesh_render_resize){
        on_mesh_render_window_resize();
        window_priv_data.mesh_render_resize = false;
    }
    if(window_priv_data.swc_view_resize){
        on_swc_view_window_resize();
        window_priv_data.swc_view_resize = false;
    }
}

void VolAnnotaterGUI::create_mesh_render_gl_framebuffer_resource() {
    auto w = window_priv_data.mesh_render_window_size.x;
    auto h = window_priv_data.mesh_render_window_size.y;
    mesh_render_framebuffer.frame_width = w;
    mesh_render_framebuffer.frame_height = h;

    mesh_render_framebuffer.fbo.destroy();
    mesh_render_framebuffer.rbo.destroy();
    mesh_render_framebuffer.color.destroy();

    mesh_render_framebuffer.fbo.initialize_handle();
    mesh_render_framebuffer.rbo.initialize_handle();
    mesh_render_framebuffer.rbo.set_format(GL_DEPTH32F_STENCIL8, w, h);
    mesh_render_framebuffer.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, mesh_render_framebuffer.rbo);

    mesh_render_framebuffer.color.initialize_handle();
    mesh_render_framebuffer.color.initialize_texture(1, GL_RGBA8, w, h);
}

void VolAnnotaterGUI::create_vol_render_gl_framebuffer_resource() {
    auto w = window_priv_data.vol_render_window_size.x;
    auto h = window_priv_data.vol_render_window_size.y;
    vol_swc_render_framebuffer.frame_width = w;
    vol_swc_render_framebuffer.frame_height = h;

    vol_swc_render_framebuffer.fbo.destroy();
    vol_swc_render_framebuffer.rbo.destroy();
    vol_swc_render_framebuffer.color.destroy();
    vol_swc_render_framebuffer.depth.destroy();
    vol_swc_render_framebuffer.hash_id.destroy();
    vol_swc_render_framebuffer.m_hash_id.destroy();

    vol_swc_render_framebuffer.fbo.initialize_handle();
    vol_swc_render_framebuffer.rbo.initialize_handle();
    vol_swc_render_framebuffer.rbo.set_format(GL_DEPTH32F_STENCIL8, w, h);
    vol_swc_render_framebuffer.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, vol_swc_render_framebuffer.rbo);
    assert(vol_swc_render_framebuffer.fbo.is_complete());

    vol_swc_render_framebuffer.color.initialize_handle();
    vol_swc_render_framebuffer.color.initialize_texture(1, GL_RGBA8, w, h);
    vol_swc_render_framebuffer.depth.initialize_handle();
    vol_swc_render_framebuffer.depth.initialize_texture(1, GL_R32F, w, h);
    vol_swc_render_framebuffer.hash_id.initialize_handle();
    vol_swc_render_framebuffer.hash_id.initialize_texture(1, GL_R32I, w, h);
    vol_swc_render_framebuffer.m_hash_id.initialize(w, h, 0);
}

bool VolAnnotaterGUI::check_and_start_annotating() {
    //必须在体数据加载了 并且选中了一个swc文件的情况下才能开始标注

    return true;
}

bool VolAnnotaterGUI::stop_and_save_annotating() {


    return true;
}

void VolAnnotaterGUI::check_and_add_swc_pt() {
    assert(is_annotating());
    auto& ret = vol_render_resc->vol_query_priv_data.query_info_view;
    // extract query result to swc point
    SWC::SWCPoint swc_pt;
    swc_pt.x = ret.at(0);
    swc_pt.y = ret.at(1);
    swc_pt.z = ret.at(2);
    swc_pt.radius = ret.at(3);
    swc_pt.tag = 1;
    swc_pt.id = 0;
    swc_pt.pid = 0;

    auto is_valid = [&]()->bool{
        if(swc_pt.radius == 0.f){
            LOG_INFO("swc pt with radius is 0.f, this may be caused for query with block which is not lod0");
            return false;
        }
        return true;
    };

    if(!is_valid()){
        LOG_INFO("Query SWC Point Not Valid: {} {} {} {}", swc_pt.x, swc_pt.y, swc_pt.z, swc_pt.radius);
        return;
    }
    if(swc_op == SWC_OP_Point)
        swc_resc->InsertSWCPoint(swc_pt);
    else if(swc_op == SWC_OP_Segment)
        swc_resc->InsertInternalSWCPoint(swc_pt);

    //更新受影响的block
    update_swc_influenced_blocks();
}

bool VolAnnotaterGUI::is_annotating() {
    return status_flags & VOL_ANNOTATING;
}

void VolAnnotaterGUI::render_swc() {

    vol_swc_render_framebuffer.fbo.bind();
    GL_EXPR(glViewport(0, 0, vol_swc_render_framebuffer.frame_width, vol_swc_render_framebuffer.frame_height));

    vol_swc_render_framebuffer.fbo.attach(GL_COLOR_ATTACHMENT0, vol_swc_render_framebuffer.color);
    vol_swc_render_framebuffer.fbo.attach(GL_COLOR_ATTACHMENT1, vol_swc_render_framebuffer.hash_id);


    if((status_flags & VOL_DRAW_VOLUME) == 0){
        vutil::gl::framebuffer_t::clear_buffer(GL_COLOR_BUFFER_BIT);
    }
    GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT1));
    static GLint clear_color[4] = {0, 0, 0, 0};
    GL_EXPR(glClearBufferiv(GL_COLOR, 0, clear_color));
    vutil::gl::framebuffer_t::clear_buffer(GL_DEPTH_BUFFER_BIT);

    // blend with depth, transfer view depth to proj depth
    if(status_flags & VOL_SWC_VOLUME_BLEND_WITH_DEPTH){
        v2p_priv_data.v2p_params.proj = camera.get_proj();
        v2p_priv_data.v2p_params_buffer.set_buffer_data(&v2p_priv_data.v2p_params);

        v2p_priv_data.v2p_shader.bind();

        v2p_priv_data.v2p_params_buffer.bind(0);
        vol_swc_render_framebuffer.depth.bind(0);

        quad_vao.bind();

        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

        quad_vao.unbind();

        v2p_priv_data.v2p_shader.unbind();
    }

    static GLenum vol_swc_draw_buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    GL_EXPR(glDrawBuffers(2, vol_swc_draw_buffers));

    swc_resc->swc_renderer->Draw(camera.get_view(), camera.get_proj(), status_flags & VOL_DRAW_SWC_TAG);

    GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    vol_swc_render_framebuffer.fbo.unbind();

}

bool VolAnnotaterGUI::can_start_annotating() {
    if((status_flags & VOL_DATA_LOADED) == 0) return false;
    if(swc_resc->loaded_swc.empty() || swc_resc->selected_swc_uid == INVALID_RESC_ID) return false;

    return true;
}

void VolAnnotaterGUI::update_swc_influenced_blocks(bool all) {
    if(!swc2mesh_resc->LocalUpdating()){
        LOG_DEBUG("update_swc_influenced_blocks but MeshStatus is not Blocked");
        return;
    }
    std::vector<BlockUID> blocks;
    if(all){
        blocks = vol_render_resc->ComputeIntersectBlocks(swc_resc->GetSelected().swc->PackAll());
    }
    else
        blocks = vol_render_resc->ComputeIntersectBlocks(swc_resc->GetSelected().swc->GetAllModifiedAndInfluencedPts());

    for(auto& block : blocks){
        swc2mesh_resc->SetBlockMeshStatus(block.SetSWC(), SWC2MeshRescPack::Modified);
    }
    swc2mesh_resc->MeshUpdated();

    swc_resc->GetSelected().swc->Commit();

}

void VolAnnotaterGUI::generate_modified_mesh() {
    AutoTimer timer("generate mesh");
    //首先找到与受影响数据块相交的线段 然后遍历所有线段 如果与受影响区域相交 那么将其加入队列
    //有一个限制 数据块不能超过vtex的容量 不然需要分多次进行
    auto pts = swc_resc->GetSelected().swc->PackAll();

    std::unordered_set<BlockUID> seg_blocks;

    //这部分数据块是 影响线段所涉及的
    std::vector<BoundingBox3D> block_boxes;
    auto block_length_space = vol_render_resc->render_vol.lod0_block_length_space;
    for(auto& [uid, block_mesh] : swc2mesh_resc->s2m_priv_data.patch_mesh_mp){
        if(block_mesh.status != SWC2MeshRescPack::Modified) continue;
        seg_blocks.insert(uid);
        Float3 f_uid = Float3(uid.x, uid.y, uid.z);
        block_boxes.push_back({f_uid * block_length_space,
                               (f_uid + Float3(1)) * block_length_space});
    }
    LOG_DEBUG("seg_blocks count : {}", seg_blocks.size());
    std::vector<SWCSegment> swc_segments;

    std::unordered_map<SWC::SWCPointKey, SWC::SWCPoint> pt_mp;
    for(auto& pt : pts) pt_mp[pt.id] = pt;

    auto get_box = [](const SWC::SWCPoint& pt){
        BoundingBox3D box;
        box |= Float3(pt.x - pt.radius, pt.y - pt.radius, pt.z - pt.radius);
        box |= Float3(pt.x + pt.radius, pt.y + pt.radius, pt.z + pt.radius);
        return box;
    };

    auto intersect = [&](const BoundingBox3D& box){
        for(auto& b : block_boxes){
            if(box.intersect(b))
                return true;
        }
        return false;
    };


    for(auto& pt : pts){
        if(pt_mp.count(pt.pid) == 0) continue;
        auto& prev_pt = pt_mp.at(pt.pid);
        auto box = get_box(prev_pt) | get_box(pt);
        if(intersect(box)){
            swc_segments.emplace_back(Float4(prev_pt.x, prev_pt.y, prev_pt.z, prev_pt.radius),
                                      Float4(pt.x, pt.y, pt.z, pt.radius));
        }
    }


    size_t count = swc_segments.size();
    if(count > MaxSegmentCount){
        throw std::runtime_error("SWC segments' size large than MaxSegmentCount : " + std::to_string(count));
    }

    LOG_TRACE("start voxelize swc segments count : {}", count);

    auto gen_box = [](const Float4& f){
        BoundingBox3D box;
        box |= Float3(f.x - f.w, f.y - f.w, f.z - f.w);
        box |= Float3(f.x + f.w, f.y + f.w, f.z + f.w);
        return box;
    };
    //重新生成真正涉及到的数据数据块 wrong!
    //不需要了 已经拿到了受影响的swc segments和真正受影响的blocks 前者的涉及的blocks可能比后者要多

//    for(auto& seg : swc_segments){
//        auto box = gen_box(seg.first) | gen_box(seg.second);
//        std::vector<BlockUID> tmp;
//        ComputeIntersectedBlocksWithBoundingBox(tmp, vol_render_resc->render_vol.lod0_block_length_space,
//                                                vol_render_resc->render_vol.lod0_block_dim,
//                                                vol_render_resc->render_vol.volume_bound, box);
//        for(auto& b : tmp) seg_blocks.insert(b);
//    }
    //获取页表
    std::vector<BlockUID> seg_blocks_; seg_blocks_.reserve(seg_blocks.size());
    for(auto& b : seg_blocks){
        seg_blocks_.push_back(BlockUID(b).SetSWC());
    }
    std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
    viser_resc->gpu_pt_mgr_ref->GetAndLock(seg_blocks_, blocks_info);
    for(auto& key : seg_blocks_) viser_resc->gpu_pt_mgr_ref->Promote(key);
    //获取页表项后即可 没有从host pool中调度的步骤
    //更新gpu页表并绑定到swc voxelizer
    swc2mesh_resc->swc_voxelizer->BindPTBuffer(viser_resc->gpu_pt_mgr_ref->GetPageTable().GetHandle());

    SWCVoxelizer::SWCVoxelizeAlgoParams params;
    params.ptrs = swc2mesh_resc->s2m_priv_data.segment_buffer->view_1d<SWCSegment>(count);
    for(int i = 0; i < count; i++){
        params.ptrs.at(i) = swc_segments[i];
    }

    //SWC需要的数据块需要添加额外的标记 调用BlockUID的SetSWC 这样子数据块的key会与原来不同 即相同的xyz lod存在于页表对应不同的物理数据块


    swc2mesh_resc->swc_voxelizer->Run(params);

    //voxelize过程中会标记真正受影响的block 需要获取 并且也会同步写入到gpu的pt
    viser_resc->gpu_pt_mgr_ref->GetPageTable(false).DownLoad();
    auto v_blocks = viser_resc->gpu_pt_mgr_ref->GetPageTable(false).GetKeys(TexCoordFlag_IsValid | TexCoordFlag_IsSWC | TexCoordFlag_IsSWCV);
    //note: 这里不需要对所有写入的block重新进行mc 但是这个功能设计还是有用的
    v_blocks = seg_blocks_;

    //对每一个真正体素化影响到的block进行mc
    //这里的pt在体素化的时候被更新了 添加了TexCoordFlag_IsSWCV 这个其实没啥用了
    swc2mesh_resc->mc_algo->BindPTBuffer(viser_resc->gpu_pt_mgr_ref->GetPageTable(false).GetHandle());

    // run marching cube
    MarchingCubeAlgo::MarchingCubeAlgoParams mc_params;
    mc_params.shape = UInt3(viser_resc->vol_priv_data.block_length);
    mc_params.isovalue = 0.5f;
    for(auto& uid : v_blocks){
        assert(uid.IsSWC());
        mc_params.origin = UInt3(uid.x, uid.y, uid.z) * mc_params.shape;
        mc_params.lod = uid.GetLOD();

        int tri_num = swc2mesh_resc->mc_algo->Run(mc_params);
        LOG_DEBUG("Marching Cube for block uid {} {} {} {} gen tri num {}",
                  uid.x, uid.y, uid.z, uid.w, tri_num);

        //从mc_params.gen_host_vertices_ret中拷贝结果 并且生成normal 即转换为MeshData0

        auto mesh = NewHandle<Mesh>(RescAccess::Shared);
        auto view = mc_params.gen_host_vertices_ret;
        mesh->Insert(MeshData0(tri_num, [&](int vert_idx)->const Float3&{
            return view.at(vert_idx);
        }), 0);
        swc2mesh_resc->UpdateBlockMesh(uid, std::move(mesh));
    }
    //有一些被标记为modified的block其实不会被体素化 这部分也更更新为Updated 虽然这部分可能没有mesh
    //因此 将所有存在的block mesh都标记为Updated
    swc2mesh_resc->UpdateAllBlockMesh();

    //通知mesh渲染器更新数据
    swc2mesh_resc->SetMeshStatus(SWC2MeshRescPack::Blocked);

    //跑完mc后再释放pt
    viser_resc->gpu_pt_mgr_ref->Release(seg_blocks_);
    //清除体素化写入的vtex
    for(auto& [uid, tex] : blocks_info){
        viser_resc->gpu_vtex_mgr_ref->Clear(uid.ToUnifiedRescUID(), tex);
    }

}

void VolAnnotaterGUI::update_vol_camera_setting(bool init) {
    if(init){
        vol_camera_move_speed = vol_render_resc->render_base_space * 3.f;
        vol_camera_view_ratation_speed = vol_render_resc->render_base_space * 5.f;
    }

    camera.set_move_speed(vol_camera_move_speed);
    camera.set_view_rotation_speed(vol_camera_view_ratation_speed);
}

bool VolAnnotaterGUI::pick_swc_point(){
    LOG_DEBUG("start pick swc point");
    auto w = vol_swc_render_framebuffer.frame_width;
    auto h = vol_swc_render_framebuffer.frame_height;
    GL_EXPR(glGetTextureImage(vol_swc_render_framebuffer.hash_id.handle(), 0, GL_RED_INTEGER, GL_INT,
                      w * h * sizeof(int),
                      vol_swc_render_framebuffer.m_hash_id.get_raw_data()));
    GL_EXPR(glFinish());
    auto [x, y] = vol_render_resc->vol_query_priv_data.query_pos;
//    x += 1;
//    y += 22;
    auto id = vol_swc_render_framebuffer.m_hash_id.at(x, y);



    if(id <= 0 || swc_resc->GetSelected().swc->QueryNode(id) == false) return false;
    LOG_DEBUG("picked swc point id : {}", id);
    swc_resc->AddPickedSWCPoint(id);

    if(swc_op == SWC_OP_Segment){
        swc_resc->UpdatePickedSWCSegmentPoints();
        swc_resc->AddPickedSWCSegPtsToRenderer();
    }

    //    VISER_WHEN_DEBUG(
    //        vutil::save_rgb_to_png_file("test.png",vol_swc_render_framebuffer.m_hash_id.map([](int x){
    //                                                                                        return color3b(x);
    //                                                                                    }).get_data());)

    return true;
}

void VolAnnotaterGUI::process_picked_swc_point(){

}
void VolAnnotaterGUI::init_ui_func()
{
    auto picked_pt_in_vol_render_frame = [&]()->bool{
        auto pos = vol_render_resc->vol_query_priv_data.clicked;
        auto [x, y] = vol_render_resc->vol_query_priv_data.query_pos;
        if(x < 0 || y < 0
            || x >= window_priv_data.vol_render_window_size.x
            || y >= window_priv_data.vol_render_window_size.y) return false;
        return vol_swc_render_framebuffer.m_hash_id.at(x + 1, y + 22) == swc_resc->swc_priv_data.last_picked_swc_pt_id;
    };
    swc_op_ui[SWC_OP_Point_UpdateR] = [&, picked_pt_in_vol_render_frame = std::move(picked_pt_in_vol_render_frame)](){
        if(!swc_resc->SelectedSWCPoint()) return;
        auto& swc_pt = swc_resc->GetSelectedSWCPoint();
        static float r = 0.f;// =  swc_pt.radius;
        ImGui::NewLine();
        ImGui::InputFloat("New Radius", &r, 0.f, 0.f, "%.5f");
        bool update = false;
        if(ImGui::Button("Update", ImVec2(100, 18))){
            swc_resc->GetSelected().swc->UpdateRadius(swc_pt.id, r);
            update = true;
        }
        ImGui::NewLine();

        if(picked_pt_in_vol_render_frame() && ImGui::Button("Query-Update", ImVec2(150, 18))){
            if(query_volume()){
                auto& ret = vol_render_resc->vol_query_priv_data.query_info_view;
                Float3 pos;
                pos.x = ret.at(0);
                pos.y = ret.at(1);
                pos.z = ret.at(2);
                auto nr = ret.at(3);
                Float3 opos = {swc_pt.x, swc_pt.y, swc_pt.z};
                if((pos - opos).length() > vol_render_resc->render_base_space * 16.f){
                    LOG_ERROR("Query And Update SWC Point Radius: new query pos is not closed to old pos");
                }
                else{
                    swc_resc->GetSelected().swc->UpdateRadius(swc_pt.id, nr);
                    update = true;
                }
            }
        }
        if(update){
            update_swc_influenced_blocks();
        }
    };

    swc_op_ui[SWC_OP_Point_Delete] = [&](){
        if(!swc_resc->SelectedSWCPoint()) return;
        auto& swc_pt = swc_resc->GetSelectedSWCPoint();
        ImGui::NewLine();
        static bool connect = true;
        ImGui::Checkbox("Connect after Delete", &connect);
        bool update = false;
        auto pid = swc_pt.pid;
        if(ImGui::Button("Delete", ImVec2(120, 18))){
            swc_resc->GetSelected().swc->DeleteNode(swc_pt.id, connect);
            update = true;
        }
        if(update){
            update_swc_influenced_blocks();

            swc_resc->UpdateSWCDrawTree();//重新生成地铁图相关

            swc_resc->ResetSWCRenderer();

            swc_resc->AddPickedSWCPoint(pid);
        }
    };

    swc_op_ui[SWC_OP_Seg_InterpR] = [&](){
        auto id1 = swc_resc->swc_priv_data.picked_swc_pt_q.back();
        auto id2 = swc_resc->swc_priv_data.picked_swc_pt_q.front();
        if(id1 <= 0 || id2 <= 0) return;
        auto swc = swc_resc->GetSelected().swc;
        if(!swc->IsRoot(id1, id2) && !swc->IsRoot(id2, id1)){
            ImGui::BulletText("Not on the same axon");
        }
        else{
            bool update = false;
            ImGui::BulletText("Point1 ID: %d, Radius %.5f", id1, swc->GetNode(id1).radius);
            ImGui::BulletText("Point2 ID: %d, Radius %.5f", id2, swc->GetNode(id2).radius);
            if(ImGui::Button("Interp", ImVec2(120, 18))){
                std::vector<SWCPointKey> ids;
                if(swc->IsRoot(id2, id1)) std::swap(id1, id2);
                assert(swc->IsRoot(id1, id2));
                auto id = id2;
                while(id != id1){
                    ids.emplace_back(id);
                    id = swc->GetNode(id).pid;
                }
                ids.emplace_back(id1);
                auto r1 = swc->GetNode(id1).radius;
                auto r2 = swc->GetNode(id2).radius;
                int n = ids.size();
                float total_length = 0.f;
                for(int i = 1; i < n; i++){
                    auto& prev_pt = swc->GetNode(ids[i - 1]);
                    auto& cur_pt = swc->GetNode(ids[i]);
                    total_length += Float3(prev_pt.x - cur_pt.x, prev_pt.y - cur_pt.y, prev_pt.z - cur_pt.z).length();
                }
                float dist = 0.f;
                auto calc_interp_r = [&](float x){
                    float u = x / total_length;
                    return r1 * (1.f - u) + r2 * u;
                };
                for(int i = n - 2; i > 0; i--){
                    auto& prev_pt = swc->GetNode(ids[i + 1]);
                    auto& cur_pt = swc->GetNode(ids[i]);
                    dist += Float3(prev_pt.x - cur_pt.x, prev_pt.y - cur_pt.y, prev_pt.z - cur_pt.z).length();
                    swc->UpdateRadius(ids[i], calc_interp_r(dist));
                }
                update = true;
            }
            if(update){
                update_swc_influenced_blocks();
            }
        }
    };

    swc_op_ui[SWC_OP_Seg_Delete] = [&](){
        auto id1 = swc_resc->swc_priv_data.picked_swc_pt_q.back();
        auto id2 = swc_resc->swc_priv_data.picked_swc_pt_q.front();
        if(id1 <= 0 || id2 <= 0) return;
        auto swc = swc_resc->GetSelected().swc;
        if(!swc->CheckUniquePath(id1, id2)){
            ImGui::BulletText("Not unique path");
            return;
        }
        ImGui::BulletText("Point1 ID: %d, Radius %.5f", id1, swc->GetNode(id1).radius);
        ImGui::BulletText("Point2 ID: %d, Radius %.5f", id2, swc->GetNode(id2).radius);
        if(ImGui::Button("Delete", ImVec2(120, 18))){
            swc->DeleteUniquePath(id1, id2);
            update_swc_influenced_blocks();
            swc_resc->UpdateSWCDrawTree();

            swc_resc->ResetSWCRenderer();

        }
    };
    swc_op_ui[SWC_OP_Seg_Add] = [&](){
        auto id1 = swc_resc->swc_priv_data.picked_swc_pt_q.back();
        auto id2 = swc_resc->swc_priv_data.picked_swc_pt_q.front();
        if(id1 <= 0 || id2 <= 0) return;
        auto swc = swc_resc->GetSelected().swc;
        if(swc->GetNode(id1).pid != id2 && swc->GetNode(id2).pid != id1){
            ImGui::BulletText("Error : Selected two swc points are not neighborhood");
            return;
        }
    };
}

