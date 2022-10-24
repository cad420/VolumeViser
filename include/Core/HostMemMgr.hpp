#pragma once

#include <Common/Common.hpp>
#include "FixedHostMemMgr.hpp"
#include <Model/Volume.hpp>

//使用cuda的host malloc函数分配得到的内存，因为是pinned的，可以更快地由CPU传输到GPU
//自定义优先级的缓存管理
//建立了CPU内存的缓存层，以及适用于transfer from CPU to GPU

VISER_BEGIN
class HostMemMgrPrivate;

using HostBuffer = std::vector<uint8_t>;

class HostMemMgr : public UnifiedRescBase{
public:

    enum RescType{
        Paged = 0,
        Pinned = 1
    };

    struct HostMemMgrCreateInfo{
        size_t MaxCPUMemBytes;
    };

    explicit HostMemMgr(const HostMemMgrCreateInfo& info);

    ~HostMemMgr();

    // 整体的加锁
    void Lock();

    void UnLock();

    UnifiedRescUID GetUID() const;

    template<typename T, RescType type>
    Handle<T> AllocHostMem(RescAccess access, size_t bytes) = delete;

    template<>
    Handle<CUDAHostBuffer> AllocHostMem<CUDAHostBuffer, Pinned>(RescAccess access, size_t bytes);

#define AllocPinnedHostMem(access, bytes) AllocHostMem<CUDAHostBuffer, HostMemMgr::RescType::Pinned>(access,bytes)


    template<>
    Handle<HostBuffer> AllocHostMem<HostBuffer, Paged>(RescAccess access, size_t bytes);

#define AllocPagedHostMem(access, bytes) AllocHostMem<HostBuffer, HostMemMgr::RescType::Paged>(access, bytes)

    using FixedHostMemMgrCreateInfo = FixedHostMemMgr::FixedHostMemMgrCreateInfo;
    UnifiedRescUID RegisterFixedHostMemMgr(const FixedHostMemMgrCreateInfo& info);

    Ref<FixedHostMemMgr> GetFixedHostMemMgrRef(UnifiedRescUID uid);

    using GridVolumeCreateInfo = GridVolume::GridVolumeCreateInfo;
    UnifiedRescUID RegisterGridVolume(const GridVolumeCreateInfo& info);

    Ref<GridVolume> GetGridVolumeRef(UnifiedRescUID uid);


protected:
    friend class ResourceMgr;
    std::unique_ptr<HostMemMgrPrivate> _;
};


VISER_END