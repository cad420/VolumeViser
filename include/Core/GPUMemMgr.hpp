#pragma once

#include <Common/Common.hpp>
#include "GPUVTexMgr.hpp"

VISER_BEGIN

/**
 * @brief 统一的gpu资源管理，提供渲染器需要的资源，如果一部分的资源被一个渲染器使用，那么
 * 这部分的资源无法被其它渲染器使用，如果一个渲染器无法获取足够的资源，那么需要等待。
 * 即对资源的读写锁粒度为一个完整的GPU资源对象，而不是以前的资源对象的某一区域
 * 因此可以有多个渲染器一起使用不冲突、不重合的资源。
 */

class GPUMemMgrPrivate;
class GPUMemMgr : public UnifiedRescBase{
public:
    struct GPUMemMgrCreateInfo{
        int GPUIndex;
        size_t MaxGPUMemBytes;
    };

    //注意 一个GPUMemMgr代表一个cuda context 一般一个gpu对应一个GPUMemMgr
    //同一个GPU的两个GPUMemMgr之间属于两个不同的context
    explicit GPUMemMgr(const GPUMemMgrCreateInfo& info);

    ~GPUMemMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    // 创建GPU资源会返回一个句柄
    // create unique resource or shared resource
    // 如果是Shared的资源，提供加锁操作，Unique资源总是能加锁成功，而Shared则不一定
    // 通过Alloc得到的资源，不会被记录在内部，需要调用者自己保管，无论是Shared还是Unique
    // 当可用内存不足时，会抛出异常
    Handle<CUDABuffer> AllocBuffer(RescAccess access, size_t bytes);

    Handle<CUDAPitchedBuffer> AllocPitchedBuffer(RescAccess access, size_t width, size_t height, size_t ele_size);

    struct TextureCreateInfo{
        cub::texture_resc_info resc_info;
        cub::texture_view_info view_info;
    };
    Handle<CUDATexture> AllocTexture(RescAccess access, const TextureCreateInfo& info);

    using GPUVTexMgrCreateInfo = GPUVTexMgr::GPUVTexMgrCreateInfo;
    UnifiedRescUID RegisterGPUVTexMgr(const GPUVTexMgrCreateInfo& info);

    Ref<GPUVTexMgr> GetGPUVTexMgrRef(UnifiedRescUID uid);

protected:
    friend class GPUVTexMgr;
    friend class CRTVolumeRenderer;
    friend class HashPageTable;
    //不再更新使用内存
    Handle<CUDATexture> _AllocTexture(RescAccess access, const TextureCreateInfo& info);

    cub::cu_context _get_cuda_context() const ;

    friend class ResourceMgr;
    std::unique_ptr<GPUMemMgrPrivate> _;

};



VISER_END