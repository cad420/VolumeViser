#pragma once


#include <Core/GPUMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN


class ResourceMgrPrivate;
class ResourceMgr final{
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

    using UID = size_t;

    //失败会抛出异常
    UID RegisterResourceMgr(ResourceDesc desc);

    std::vector<UID> GetAll() const;

    bool Exist(UID uid) const;

    bool Exist(UID uid, ResourceType type) const;

    template<typename T, ResourceType type>
    Ref<T> GetResourceMgrRef(UID) = delete;

    template<>
    Ref<GPUMemMgr> GetResourceMgrRef<GPUMemMgr, Device>(UID);

    template<>
    Ref<HostMemMgr> GetResourceMgrRef<HostMemMgr, Host>(UID);

#define GetHostRef(UID) GetResourceMgrRef<HostMemMgr, ResourceMgr::Host>(UID)
#define GetGPURef(UID) GetResourceMgrRef<GPUMemMgr, ResourceMgr::Device>(UID)


    static ResourceMgr& GetInstance();

    ~ResourceMgr();

private:
    ResourceMgr();

    std::unique_ptr<ResourceMgrPrivate> _;
};


VISER_END
