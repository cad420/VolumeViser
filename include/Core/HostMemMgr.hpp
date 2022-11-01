#pragma once

#include <Model/Volume.hpp>
#include <Core/FixedHostMemMgr.hpp>

VISER_BEGIN

//使用CUDA的host malloc函数分配得到的内存，因为是pinned的，可以更快地由CPU传输到GPU
//理论上一个进程只需要一个HostMemMgr实例，以后可以扩展到一个体数据对应一个HostMemMgr

class HostMemMgrPrivate;
class HostMemMgr : public UnifiedRescBase{
public:
    enum RescType{
        Paged = 0,
        Pinned = 1
    };

    struct HostMemMgrCreateInfo{
        size_t MaxCPUMemBytes = 0;
    };

    explicit HostMemMgr(const HostMemMgrCreateInfo& info);

    ~HostMemMgr();

    void Lock();

    void UnLock();

    UnifiedRescUID GetUID() const;

    template<typename T, RescType type>
    Handle<T> AllocHostMem(RescAccess access, size_t bytes, bool required = false) = delete;

    template<>
    Handle<CUDAHostBuffer> AllocHostMem<CUDAHostBuffer, Pinned>(RescAccess access, size_t bytes, bool required);

#define AllocPinnedHostMem(access, bytes, required) AllocHostMem<CUDAHostBuffer, HostMemMgr::RescType::Pinned>(access,bytes,required)

    template<>
    Handle<HostBuffer> AllocHostMem<HostBuffer, Paged>(RescAccess access, size_t bytes, bool required);

#define AllocPagedHostMem(access, bytes, required) AllocHostMem<HostBuffer, HostMemMgr::RescType::Paged>(access,bytes,required)

    using FixedHostMemMgrCreateInfo = FixedHostMemMgr::FixedHostMemMgrCreateInfo;
    UnifiedRescUID RegisterFixedHostMemMgr(const FixedHostMemMgrCreateInfo& info);

    void UnRegisterFixedHostMemMgr(UnifiedRescUID uid);

    Ref<FixedHostMemMgr> GetFixedHostMemMgrRef(UnifiedRescUID uid);

protected:
    std::unique_ptr<HostMemMgrPrivate> _;
};

VISER_END