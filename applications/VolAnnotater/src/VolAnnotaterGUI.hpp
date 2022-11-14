#include "Common.hpp"

class VolAnnotaterGUI : public gl_app_t{
public:
    using gl_app_t::gl_app_t;

    void Initialize();

private:

    void initialize() override;

    void frame() override;

    void destroy() override;
private:
    void show_editor_menu();


private:
    std::unique_ptr<ViserRescPack> viser_resc;

    std::unique_ptr<VolRenderRescPack> vol_render_resc;

    std::unique_ptr<SWCRescPack> swc_resc;

    std::unique_ptr<SWC2MeshRescPack> swc2mesh_resc;

    struct{
        cudaGraphicsResource_t cuda_frame_color_resc;
        cudaGraphicsResource_t cuda_frame_depth_resc;
        pixel_unpack_buffer color_pbo;
        pixel_unpack_buffer depth_pbo;
    }cudaGL_interop;

    struct{

    }v2p_priv_data;

    Timer timer;
};