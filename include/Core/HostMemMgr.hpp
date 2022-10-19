#pragma once

#include <Common/Common.hpp>

//使用cuda的host malloc函数分配得到的内存，因为是pinned的，可以更快地由CPU传输到GPU
//自定义优先级的缓存管理
//建立了CPU内存的缓存层，以及适用于transfer from CPU to GPU

VISER_BEGIN
class HostMemMgrPrivate;
class HostMemMgr{
public:
    struct HostMemMgrCreateInfo{
        size_t MaxCPUMemBytes;
    };

    explicit HostMemMgr(const HostMemMgrCreateInfo& info);

    ~HostMemMgr();

    struct Key{
        int DeviceID;
        int HashID;
        std::string HashUID;

    };

    struct Value{
        void* ptr;
        size_t size;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t pitch;

    };

    // 整体的加锁
    void Lock();

    void UnLock();


    Value Get(const Key& key, std::function<void()> f = nullptr);

    //cuMemAllocHost
protected:
    friend class ResourceMgr;
    std::unique_ptr<HostMemMgrPrivate> _;
};


VISER_END