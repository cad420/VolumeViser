#include <Core/ResourceMgr.hpp>

VISER_BEGIN

class ResourceMgrPrivate{
public:
    std::unordered_map<UnifiedRescUID, std::unique_ptr<GPUMemMgr>> gpu_mgrs;
    std::unordered_map<UnifiedRescUID, std::unique_ptr<HostMemMgr>> host_mgrs;

    std::mutex mtx;
};

ResourceMgr& ResourceMgr::GetInstance() {
    static ResourceMgr ins;
    return ins;
}

static auto init = [](){
    LOG_DEBUG("ResourceMgr init successfully...");
    return ResourceMgr::GetInstance().GetUID();
};

Ref<ResourceMgr> ResourceMgr::GetInstanceSafe() {
    return Ref<ResourceMgr>(&GetInstance());
}

ResourceMgr::ResourceMgr() {
    _ = std::make_unique<ResourceMgrPrivate>();

}

ResourceMgr::~ResourceMgr() {

}

UnifiedRescUID ResourceMgr::GetUID() const {
    return GenUnifiedRescUID(1, UnifiedRescType::RescMgr);
}

void ResourceMgr::Lock() {
    _->mtx.lock();
}

void ResourceMgr::UnLock() {
    _->mtx.unlock();
}

UnifiedRescUID ResourceMgr::RegisterResourceMgr(const ResourceDesc& desc) {
    try{
        if(desc.type == Host){
            auto resc = std::make_unique<HostMemMgr>(HostMemMgr::HostMemMgrCreateInfo{.MaxCPUMemBytes = desc.MaxMemBytes});
            auto uid = resc->GetUID();
            assert(_->host_mgrs.count(uid) == 0);
            _->host_mgrs[uid] = std::move(resc);
            return uid;
        }
        else if(desc.type == Device){
            auto resc = std::make_unique<GPUMemMgr>(GPUMemMgr::GPUMemMgrCreateInfo{.GPUIndex = desc.DeviceIndex, .MaxGPUMemBytes = desc.MaxMemBytes});
            auto uid = resc->GetUID();
            assert(_->gpu_mgrs.count(uid) == 0);
            _->gpu_mgrs[uid] = std::move(resc);
            return uid;
        }
        else
            assert(false);
    }
    catch (const std::exception& e) {
        // print desc info
        LOG_ERROR("RegisterResourceMgr with invalid params: ({0}, {1}, {2})",
                  desc.type == Host ? "Host" : (desc.type == Device ? "Device" : "Unknown"),
                  desc.MaxMemBytes, desc.DeviceIndex);
        // throw
        throw ViserResourceCreateError(std::string("RegisterResourceMgr exception : ") + e.what());
    }
}

bool ResourceMgr::Exist(UnifiedRescUID uid) const {
    if(_->host_mgrs.count(uid)) return true;
    if(_->gpu_mgrs.count(uid)) return true;
    return false;
}

bool ResourceMgr::Exist(UnifiedRescUID uid, ResourceType type) const {
    if(type == Host)
        return _->host_mgrs.count(uid);
    if(type == Device)
        return _->gpu_mgrs.count(uid);
    return false;
}

template<>
Ref<GPUMemMgr> ResourceMgr::GetResourceMgrRef<GPUMemMgr, ResourceMgr::Device>(UnifiedRescUID uid) {
    assert(Exist(uid, Device));
    return {_->gpu_mgrs.at(uid).get()};
}

template<>
Ref<HostMemMgr> ResourceMgr::GetResourceMgrRef<HostMemMgr, ResourceMgr::Host>(UnifiedRescUID uid) {
    assert(Exist(uid, Host));
    return {_->host_mgrs.at(uid).get()};
}

std::vector<UnifiedRescUID> ResourceMgr::GetAll() const {
    std::vector<UnifiedRescUID> all;
    for(auto& [uid, _] : _->host_mgrs){
        all.push_back(uid);
    }
    for(auto &[uid, _] : _->gpu_mgrs){
        all.push_back(uid);
    }
    return all;
}

VISER_END