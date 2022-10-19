#pragma once

#include <Core/ResourceMgr.hpp>
#include <Core/RenderParams.hpp>
#include <Model/Volume.hpp>

VISER_BEGIN



struct FrameBuffer{
    using ImageHandle = GPUMemMgr::Handle<CUDAPitchedBuffer>;
    using HostImageHandle = GPUMemMgr::Handle<CUDABuffer>;

    ImageHandle color;
    ImageHandle depth;
    std::vector<ImageHandle> attach;

};
//暂时只考虑体渲染，网格渲染的需求只有标注系统，因此网格渲染的部分完全
//都在标注应用部分编写代码，而不作为核心库里的功能。
//体绘制用CUDA实现


//渲染器本身也是一种资源，它是CPU和GPU资源的结合，因此同一时刻只能有一个调用者
class Renderer{
public:

    enum Type{
        RealTime,
        PhysicalBased
    };

    using VolumeInfo = GridVolume::GridVolumeDesc;

    using RendererUID = size_t;

    virtual ~Renderer() = default;

    virtual void Lock() = 0;

    virtual void UnLock() = 0;

    virtual void SetVolume(const VolumeInfo&) = 0;

    virtual void SetRenderParams(const RenderParams&) = 0;

    virtual void SetPerFrameParams(const PerFrameParams&) = 0;

    //返回Unique还是Shared的资源
    virtual FrameBuffer GetRenderFrame(bool exclusive) = 0;

    template<typename T>
    using VTextureHandle = GPUMemMgr::Handle<CUDAVolumeImage<T>>;
    using TextureUnit = int;
    using PTBufferHandle = GPUMemMgr::Handle<CUDABuffer>;

};

//实时的体渲染，为了最大性能，数据块的计算由CPU完成
// 或者可能的先在GPU一次性计算完
// 切片是特殊的体渲染
class RTVolumeRendererPrivate;
class RTVolumeRenderer: public Renderer{
public:

protected:
    std::unique_ptr<RTVolumeRendererPrivate> _;
};

//分辨率可以动态调整
class CRTVolumeRendererPrivate;
class CRTVolumeRenderer : public Renderer{
public:
    struct CRTVolumeRendererCreateInfo{
        Ref<GPUMemMgr> gpu_mem_mgr;
        Ref<HostMemMgr> host_mem_mgr;
        // other params...

    };
    explicit CRTVolumeRenderer(const CRTVolumeRendererCreateInfo& info);

    ~CRTVolumeRenderer() override;

    void SetVolume(const VolumeInfo&) override;

    void SetRenderParams(const RenderParams&) override;

    void SetPerFrameParams(const PerFrameParams&) override;

    FrameBuffer GetRenderFrame(bool exclusive) override;

    // 需要绑定纹理和页表两项资源
    template<typename T>
    void BindVTexture(VTextureHandle<T> handle, TextureUnit unit);

    void BindPTBuffer(PTBufferHandle handle);

protected:
    std::unique_ptr<CRTVolumeRendererPrivate> _;
};



//离线的真实感体渲染，数据块的计算放在着色器内，
//因为考虑散射后，数据块无法提前计算，而且离线不考虑这一点的性能损失
class PBVolumeRenderer: public Renderer{

};

VISER_END