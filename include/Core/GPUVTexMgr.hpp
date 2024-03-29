#pragma once

#include <Common/Common.hpp>
#include "GPUPageTableMgr.hpp"

VISER_BEGIN

//负责一个GPU上的vtexture的管理，在创建时申请资源，中间无法动态变更资源，在稀释时释放资源
//负责资源上传到GPU

class GPUVTexMgrPrivate;
class GPUVTexMgr : public UnifiedRescBase{
public:
    struct GPUVTexMgrCreateInfo{
        mutable Ref<GPUMemMgr> gpu_mem_mgr;
        mutable Ref<HostMemMgr> host_mem_mgr;
        int vtex_count;
        Int3 vtex_shape;
        int bits_per_sample;
        int samples_per_channel;
        int vtex_block_length;
        bool is_float;
        // ...
        bool exclusive;
    };

    explicit GPUVTexMgr(const GPUVTexMgrCreateInfo& info);

    ~GPUVTexMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    //两者一一对应
    Ref<GPUPageTableMgr> GetGPUPageTableMgrRef();

    using TexCoord = GPUPageTableMgr::TexCoord;

    CUB_CPU_GPU bool UploadBlockToGPUTex(Handle<CUDAHostBuffer> src, TexCoord dst);

    void UploadBlockToGPUTexAsync(Handle<CUDAHostBuffer> src, TexCoord dst);

    void Flush();

    using GPUTexUnit = uint32_t;

    using GPUVTex = std::pair<GPUTexUnit, Handle<CUDATexture>>;

    std::vector<GPUVTex> GetAllTextures();

#ifdef USE_LINEAR_BUFFER_FOR_TEXTURE
    std::vector<std::pair<GPUTexUnit, CUDABufferView3D<uint8_t>>> GetAllTextureBuffers();

#endif

    void Clear(UnifiedRescUID uid, TexCoord dst);



protected:
    std::unique_ptr<GPUVTexMgrPrivate> _;
};

VISER_END