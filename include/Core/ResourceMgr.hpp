#pragma once


#include <Core/GPUMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN

template<typename T>
class Ref{
public:

};

class ResourceMgr final{
public:
    enum ResourceType{
        Host,
        GPU
    };
    struct ResourceDesc{
        ResourceType type;
        int GPUIndex;
        size_t MaxMemBytes = 0;

    };

    using UID = size_t;

    UID RegisterResourceMgr(ResourceDesc desc);

    std::vector<std::pair<UID, ResourceDesc>> GetAll() const;

    ResourceDesc Query(UID) const;

    template<typename T, ResourceType type>
    Ref<T> GetResourceMgrRef(UID) = delete;

    template<>
    Ref<GPUMemMgr> GetResourceMgrRef<GPUMemMgr, GPU>(UID);

    template<>
    Ref<HostMemMgr> GetResourceMgrRef<HostMemMgr, Host>(UID);

    static ResourceMgr& GetInstance();

    ~ResourceMgr();

private:
    ResourceMgr() = default;
};


VISER_END
