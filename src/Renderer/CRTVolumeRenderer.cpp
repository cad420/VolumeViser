#include <Core/Renderer.hpp>

VISER_BEGIN

    class CRTVolumeRendererPrivate{
    public:

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

    template<typename T>
    void CRTVolumeRenderer::BindVTexture(VTextureHandle <T> handle, TextureUnit unit) {

    }

    void CRTVolumeRenderer::BindPTBuffer(PTBufferHandle handle) {

    }

    template<> void CRTVolumeRenderer::BindVTexture<uint8_t>(VTextureHandle<uint8_t> handle, TextureUnit unit);
    template<> void CRTVolumeRenderer::BindVTexture<uint16_t>(VTextureHandle<uint16_t> handle, TextureUnit unit);

VISER_END