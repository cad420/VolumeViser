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



class Renderer{
public:

    enum Type{
        RealTime,
        PhysicalBased
    };

    using VolumeInfo = GridVolume::GridVolumeDesc;

    virtual void SetVolume(const VolumeInfo&) = 0;

    virtual void SetRenderParams(const RenderParams&) = 0;

    virtual void SetPerFrameParams(const PerFrameParams&) = 0;

    virtual FrameBuffer GetRenderFrame(int width, int height) = 0;

    using VTextureHandle = GPUMemMgr::Handle<CUDAVolumeImage>;
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

class CRTVolumeRendererPrivate;
class CRTVolumeRenderer : public Renderer{
public:
    CRTVolumeRenderer();

    ~CRTVolumeRenderer();

    void SetVolume(const VolumeInfo&) override;

    void SetRenderParams(const RenderParams&) override;

    void SetPerFrameParams(const PerFrameParams&) override;

    FrameBuffer GetRenderFrame(int width, int height) override;

    // 需要绑定纹理和页表两项资源
    void BindVTexture(VTextureHandle handle, TextureUnit unit);

    void BindPTBuffer(PTBufferHandle handle);

protected:
    std::unique_ptr<CRTVolumeRendererPrivate> _;
};

//离线的真实感体渲染，数据块的计算放在着色器内，
//因为考虑散射后，数据块无法提前计算，而且离线不考虑这一点的性能损失
class PBVolumeRenderer: public Renderer{

};

VISER_END