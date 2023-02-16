#pragma once

#include <Core/ResourceMgr.hpp>
#include <Core/RenderParams.hpp>
#include <Model/Volume.hpp>

VISER_BEGIN

struct FrameBuffer{
    int frame_width = 0;
    int frame_height = 0;

    CUDABufferView2D<uint32_t> color;
    CUDABufferView2D<float> depth;
};
//暂时只考虑体渲染，网格渲染的需求只有标注系统，因此网格渲染的部分完全
//都在标注应用部分编写代码，而不作为核心库里的功能。
//体绘制用CUDA实现

    using VTextureHandle = Handle<CUDATexture>;
    using TextureUnit = int;
    using PTBufferHandle = Handle<CUDABuffer>;

//渲染器本身也是一种资源，它是CPU和GPU资源的结合，因此同一时刻只能有一个调用者
class Renderer : public UnifiedRescBase{
public:

    enum Type{
        RealTime,
        PhysicalBased
    };


    using RendererUID = size_t;

    virtual ~Renderer() = default;

    virtual void SetVolume(const VolumeParams&) = 0;

    virtual void SetRenderParams(const RenderParams&) = 0;

    virtual void SetPerFrameParams(const PerFrameParams&) = 0;

    virtual void Render(Handle<FrameBuffer> frame) = 0;

//    using RenderResult = cub::cu_result;
//    virtual std::future<RenderResult> RenderAsync(FrameBuffer& frame) = 0;



};

//封装
class RTVolumeRendererPrivate;
class RTVolumeRenderer: public Renderer{
public:
    struct RTVolumeRendererCreateInfo{
        Ref<HostMemMgr> host_mem_mgr;
        Ref<GPUMemMgr> gpu_mem_mgr;
        bool async = true;
        bool use_shared_host_mem = false;
        Ref<FixedHostMemMgr> shared_fixed_host_mem_mgr_ref;
    };

    explicit RTVolumeRenderer(const RTVolumeRendererCreateInfo&);

    ~RTVolumeRenderer() override;

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void BindGridVolume(Handle<GridVolume>);

    void SetVolume(const VolumeParams&) override { /* Use BindGridVolume instead */}

    void SetRenderParams(const RenderParams&) override;

    void SetPerFrameParams(const PerFrameParams&) override;

    void SetRenderMode(bool async);

    void Render(Handle<FrameBuffer> frame) override;

protected:
    std::unique_ptr<RTVolumeRendererPrivate> _;
};

//自定义
class CRTVolumeRendererPrivate;
class CRTVolumeRenderer : public Renderer{
public:
    struct CRTVolumeRendererCreateInfo{
        mutable Ref<GPUMemMgr> gpu_mem_mgr;
        mutable Ref<HostMemMgr> host_mem_mgr;
        // other params...

    };
    explicit CRTVolumeRenderer(const CRTVolumeRendererCreateInfo& info);

    ~CRTVolumeRenderer() override;

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void SetVolume(const VolumeParams&) override;

    void SetRenderParams(const RenderParams&) override;

    void SetPerFrameParams(const PerFrameParams&) override;

    void Render(Handle<FrameBuffer> frame) override;

    void Query(int x, int y, CUDABufferView1D<float>& info, int flag = 0);

    // 需要绑定纹理和页表两项资源

    void BindVTexture(VTextureHandle handle, TextureUnit unit);

    void BindPTBuffer(PTBufferHandle handle);

protected:
    std::unique_ptr<CRTVolumeRendererPrivate> _;
};


//离线的真实感体渲染，数据块的计算放在着色器内
class PBVolumeRendererPrivate;
class PBVolumeRenderer: public Renderer{
  public:
    struct PBVolumeRendererCreateInfo{

    };
    explicit PBVolumeRenderer(const PBVolumeRendererCreateInfo&);

    ~PBVolumeRenderer() override;

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void BindGridVolume(Handle<GridVolume>);

    void SetVolume(const VolumeParams&) override { /* Use BindGridVolume instead */}

    void SetRenderParams(const RenderParams&) override;

    void SetPerFrameParams(const PerFrameParams&) override;

    void Render(Handle<FrameBuffer> frame) override;



  private:
    std::unique_ptr<PBVolumeRendererPrivate> _;
};

VISER_END