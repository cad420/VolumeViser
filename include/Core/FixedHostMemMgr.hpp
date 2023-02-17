#pragma once

#include <Model/Volume.hpp>

VISER_BEGIN

//pinned和固定大小的内存块管理，启用cache机制
//主要用于管理体数据块
//必须要由HostMemMgr创建
//资源在一开始全部创建后不会再发生变化，不会动态申请新的资源

class FixedHostMemMgrPrivate;
class FixedHostMemMgr : public UnifiedRescBase{
public:

    struct FixedHostMemMgrCreateInfo{
        mutable Ref<HostMemMgr> host_mem_mgr;
        size_t fixed_block_size = 0;
        size_t fixed_block_num = 0;
    };

    explicit FixedHostMemMgr(const FixedHostMemMgrCreateInfo& info);

    ~FixedHostMemMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    //如果uid存在，返回Handle，调用者负责检查锁的状态，以及决定加读锁还是写锁，内部只是一个简单的缓存机制
    //一定会返回一个有效的句柄，会加读锁或者写锁用于表示当前状态
    //返回的句柄在使用过程中，一旦被释放锁，无法保证不会被抢占
    //因此需要在确认不再使用后才释放锁，或者直接将写锁转为读锁
    //如果多线程调用，需要对*this套一个线程安全的Ref
    Handle<CUDAHostBuffer> GetBlock(UnifiedRescUID uid);

    //GetBlockAsync不需要，因为异步的需求在调用者层次处理

protected:
    std::unique_ptr<FixedHostMemMgrPrivate> _;
};

VISER_END