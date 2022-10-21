#pragma once

#include <Common/Common.hpp>
#include <Model/Volume.hpp>

VISER_BEGIN

//pinned和固定大小的内存块管理，启用cache机制
//主要用于管理体数据块
//必须要由HostMemMgr创建
//资源在一开始全部创建后不会再发生变化，不会动态申请新的资源

class FixedHostMemMgrPrivate;
class FixedHostMemMgr{
public:
    enum CachePolicy{

    };

    struct FixedHostMemMgrCreateInfo{
        Ref<HostMemMgr> host_mem_mgr;
        size_t fixed_block_size = 0;
        size_t fixed_block_num = 0;
        CachePolicy cache_policy;
    };

    explicit FixedHostMemMgr(const FixedHostMemMgrCreateInfo& info);

    ~FixedHostMemMgr();

    // 整体的加锁
    void Lock();

    void UnLock();

    //获取一个Block的Handle，如果不存在，则返回空
    //如果emplace，那么必定返回一个空的可以填充的buffer handle，即会淘汰一个block
    //获取的Handle在析构时应该通知Mgr释放锁
    //如果Handle处于异步队列解压中，那么将无法获取
    //同时内部应该记录有那些被锁定的，也就是说，如果想获取一个uid是被锁定的，那么返回Invalid的Handle
    Handle<CUDAHostBuffer> GetBlock(UnifiedRescUID uid, bool emplace = false);

    //GetBlockAsync不需要，因为异步的需求在调用者层次处理

protected:
    std::unique_ptr<FixedHostMemMgrPrivate> _;
};



VISER_END
