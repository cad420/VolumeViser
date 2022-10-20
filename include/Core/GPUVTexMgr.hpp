#pragma once

#include <Common/Common.hpp>
#include "GPUPageTableMgr.hpp"


VISER_BEGIN

//负责一个GPU上的vtexture的管理，在创建时申请资源，中间无法动态变更资源，在稀释时释放资源
//负责资源上传到GPU
class GPUVTexMgrPrivate;
class GPUVTexMgr{
public:
    struct GPUVTexMgrCreateInfo{
        Ref<GPUMemMgr> gpu_mem_mgr;
        int vtex_count;
        Int3 vtex_shape;
        int bits_per_sample;
        int samples_per_channel;
        // ...
        bool exclusive;
    };

    explicit GPUVTexMgr(const GPUVTexMgrCreateInfo& info);

    ~GPUVTexMgr();

    void Lock();

    void UnLock();

    //两者一一对应
    Ref<GPUPageTableMgr> GetGPUPageTableMgrRef();

    using TexCoord = GPUPageTableMgr::TexCoord;
    void UploadBlockToGPUTex(Handle<CUDAHostBuffer> src, TexCoord dst);

protected:
    std::unique_ptr<GPUVTexMgrPrivate> _;
};




VISER_END