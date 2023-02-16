#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>

VISER_BEGIN

using BlockUID = GridVolume::BlockUID;

static Int3 DefaultVTexShape{1024, 1024, 1024};

namespace{

}

class RTVolumeRendererPrivate{
  public:
    Ref<HostMemMgr> host_mem_mgr_ref;
    Ref<GPUMemMgr> gpu_mem_mgr_ref;

    // data sts for loading blocks
    struct{
        std::vector<BlockUID> intersect_blocks;
    };

    UnifiedRescUID uid;

    std::mutex g_mtx;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::RTVolRenderer);
    }
};

RTVolumeRenderer::RTVolumeRenderer(const RTVolumeRenderer::RTVolumeRendererCreateInfo &info)
{

}

RTVolumeRenderer::~RTVolumeRenderer()
{

}

void RTVolumeRenderer::Lock()
{

}

void RTVolumeRenderer::UnLock()
{

}

UnifiedRescUID RTVolumeRenderer::GetUID() const
{
    return 0;
}

void RTVolumeRenderer::BindGridVolume(Handle<GridVolume>)
{

}

void RTVolumeRenderer::SetRenderParams(const RenderParams &)
{

}

void RTVolumeRenderer::SetPerFrameParams(const PerFrameParams &)
{

}

void RTVolumeRenderer::SetRenderMode(bool async)
{

}

void RTVolumeRenderer::Render(Handle<FrameBuffer> frame)
{

}

VISER_END