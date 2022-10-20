#undef UTIL_ENABLE_OPENGL
#include <Core/Renderer.hpp>

VISER_BEGIN

    namespace {

        struct CUDARenderParams{

        };
        __constant__ CUDARenderParams cu_render_params;
    }

    class CRTVolumeRendererPrivate{
    public:
        cub::cu_kernel kernel;



        FrameBuffer Render(bool exclusive){
            cub::cu_kernel_launch_info launch_info;
            kernel.pending(launch_info,
                           CUB_CPU_GPU_LAMBDA_CLS(dim3 block_idx, dim3 thread_idx){
                               blockDim,gridDim;

            });
        }
    };

    CRTVolumeRenderer::CRTVolumeRenderer(const CRTVolumeRenderer::CRTVolumeRendererCreateInfo &info) {

    }

    CRTVolumeRenderer::~CRTVolumeRenderer(){

    }

    void CRTVolumeRenderer::SetVolume(const VolumeInfo& volume_info) {

    }

    void CRTVolumeRenderer::SetRenderParams(const RenderParams& render_params) {

    }

    void CRTVolumeRenderer::SetPerFrameParams(const PerFrameParams &) {

    }

    FrameBuffer CRTVolumeRenderer::GetRenderFrame(bool exclusive) {
        return {};
    }

    void CRTVolumeRenderer::BindVTexture(VTextureHandle handle, TextureUnit unit) {

    }

    void CRTVolumeRenderer::BindPTBuffer(PTBufferHandle handle) {

    }


VISER_END