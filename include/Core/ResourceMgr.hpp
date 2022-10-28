#pragma once

#include <Core/GPUMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN

//一个进程只需要有一个该实例，管理注册的host和device资源

class ResourceMgrPrivate;
class ResourceMgr final : public UnifiedRescBase{
public:
    enum ResourceType{
        Host,
        Device
    };
    struct ResourceDesc{
        ResourceType type;
        size_t MaxMemBytes = 0;
        int DeviceIndex;
    };

    static Ref<ResourceMgr> GetInstanceSafe();

    static ResourceMgr& GetInstance();

    ~ResourceMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    //失败会抛出异常
    UnifiedRescUID RegisterResourceMgr(const ResourceDesc& desc);

    std::vector<UnifiedRescUID> GetAll() const;

    bool Exist(UnifiedRescUID uid) const;

    bool Exist(UnifiedRescUID uid, ResourceType type) const;

    template<typename T, ResourceType type>
    Ref<T> GetResourceMgrRef(UnifiedRescUID) = delete;

    template<>
    Ref<GPUMemMgr> GetResourceMgrRef<GPUMemMgr, Device>(UnifiedRescUID);

    template<>
    Ref<HostMemMgr> GetResourceMgrRef<HostMemMgr, Host>(UnifiedRescUID);

#define GetHostRef(UID) GetResourceMgrRef<HostMemMgr, ResourceMgr::Host>(UID)
#define GetGPURef(UID) GetResourceMgrRef<GPUMemMgr, ResourceMgr::Device>(UID)

private:
    ResourceMgr();

    std::unique_ptr<ResourceMgrPrivate> _;
};

VISER_END