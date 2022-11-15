#include "Common.hpp"
#include "imfilebrowser.h"

class VolAnnotaterGUI : public gl_app_t{
public:
    using gl_app_t::gl_app_t;

    void Initialize();

private:

    void initialize() override;

    void frame() override;

    void destroy() override;

    void handle_events() override;
private:
    void update_per_frame();

    void show_editor_ui();

    void show_editor_menu(bool* p_open);

    void show_editor_vol_info_window(bool* p_open);

    void show_editor_vol_render_info_window(bool* p_open);

    void show_editor_vol_render_window(bool* p_open);

    void show_editor_mesh_render_window(bool* p_open);

    void show_editor_swc_window(bool* p_open);

    void show_editor_swc_tree_window(bool* p_open);

    void show_editor_neuron_mesh_window(bool *p_open);

    void show_debug_window(bool *p_open);
private:
    void check_window_resize_event();

    void on_vol_render_window_resize();

    void on_mesh_render_window_resize();

    void on_swc_view_window_resize();
private:
    void load_volume(const std::string& filename);

    void load_swc(const std::string& filename);

    void export_swc();

    void load_mesh(const std::string& filename);

    void export_mesh();

private:
    // framebuffer 大小改变时需要调用
    void register_cuda_gl_interop_resource();

    void create_vol_render_gl_framebuffer_resource();

    void create_mesh_render_gl_framebuffer_resource();

private:
    void frame_vol_render();

    // 大规模体绘制一帧
    void render_volume();

    void update_vol_render_per_frame_params(PerFrameParams& params);

    // 点查询 与体绘制(render_volume)串行
    void query_volume();

    void frame_mesh_render();

//    void frame_swc_view_render();

private:
    std::unique_ptr<ViserRescPack> viser_resc;

    std::unique_ptr<VolRenderRescPack> vol_render_resc;

    std::unique_ptr<SWCRescPack> swc_resc;

    std::unique_ptr<SWC2MeshRescPack> swc2mesh_resc;

    // 使得整个程序的fps保持稳定的60帧 其它的绘制任务以异步形式执行 即如果没有在规定时间内完成 保持上一帧结果
    vutil::thread_group_t render_group;

    ImGui::FileBrowser file_dialog;

    struct{
        vec2i vol_render_window_size = {0, 0};
        bool vol_render_resize = false;
        vec2i mesh_render_window_size = {0, 0};
        bool mesh_render_resize = false;
        vec2i swc_view_window_size = {0, 0};
        bool swc_view_resize = false;
    }window_priv_data;

    //体绘制 用于opengl cuda资源交互相关
    struct{
        cudaGraphicsResource_t cuda_frame_color_resc = nullptr;
        cudaGraphicsResource_t cuda_frame_depth_resc = nullptr;
        pixel_unpack_buffer color_pbo;
        pixel_unpack_buffer depth_pbo;
    }vol_render_cudaGL_interop;

    // 体绘制/swc线绘制的OpenGL framebuffer资源
    struct{
        int frame_width = 0, frame_height = 0;
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t color;
        texture2d_t depth;
    }vol_swc_render_framebuffer;


    // 神经元网格绘制的OpenGL framebuffer资源
    struct{
        int frame_width = 0, frame_height = 0;
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t color;
    }mesh_render_framebuffer;

    struct{

    }v2p_priv_data;

    struct{

    }debug_priv_data;

    Timer vol_render_timer;

    struct{
        bool editor_menu_window_open = true;
        bool vol_info_window_open = true;
        bool vol_render_info_window_open = true;
        bool vol_render_window_open = true;
        bool mesh_render_window_open = true;
        bool swc_info_window_open = true;
        bool swc_tree_window_open = true;
        bool neuron_mesh_window_open = true;
    };
};