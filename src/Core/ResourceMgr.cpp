#include <Core/ResourceMgr.hpp>

VISER_BEGIN

    class ResourceMgrPrivate{
    public:
        using RescUID = ResourceMgr::UID;

        std::unordered_map<RescUID, std::unique_ptr<GPUMemMgr>> gpu_mgrs;
        std::unordered_map<RescUID, std::unique_ptr<HostMemMgr>> host_mgrs;

        //生成唯一的全局RescUID
        RescUID GenRescUID(){
            static std::atomic<RescUID> g_uid = 0;
            auto uid = g_uid.fetch_add(1);
            return uid;
        }
    };

    ResourceMgr &ResourceMgr::GetInstance() {
        static ResourceMgr ins;

        return ins;
    }

    ResourceMgr::ResourceMgr() {
        _ = std::make_unique<ResourceMgrPrivate>();

    }

    ResourceMgr::~ResourceMgr() {

    }

    ResourceMgr::UID ResourceMgr::RegisterResourceMgr(ResourceMgr::ResourceDesc desc) {
        // check desc

        //...
        try{
            if(desc.type == Host){
                auto resc = std::make_unique<HostMemMgr>(HostMemMgr::HostMemMgrCreateInfo{.MaxCPUMemBytes = desc.MaxMemBytes});
                auto uid = _->GenRescUID();
                assert(_->host_mgrs.count(uid) == 0);
                _->host_mgrs[uid] = std::move(resc);
                return uid;
            }
            else if(desc.type == Device){
                auto resc = std::make_unique<GPUMemMgr>(GPUMemMgr::GPUMemMgrCreateInfo{.GPUIndex = desc.DeviceIndex, .MaxGPUMemBytes = desc.MaxMemBytes});
                auto uid = _->GenRescUID();
                assert(_->gpu_mgrs.count(uid) == 0);
                _->gpu_mgrs[uid] = std::move(resc);
                return uid;
            }
            else
                assert(false);
        }
        catch (const std::exception& e) {
            // print desc info

            // throw
            throw ViserResourceCreateError(std::string("RegisterResourceMgr exception : ") + e.what());
        }
    }


    bool ResourceMgr::Exist(ResourceMgr::UID uid) const {
        if(_->host_mgrs.count(uid)) return true;
        if(_->gpu_mgrs.count(uid)) return true;
        return false;
    }

    bool ResourceMgr::Exist(UID uid, ResourceType type) const {
        if(type == Host)
            return _->host_mgrs.count(uid);
        if(type == Device)
            return _->gpu_mgrs.count(uid);
        return false;
    }

    template<>
    Ref<GPUMemMgr> ResourceMgr::GetResourceMgrRef<GPUMemMgr, ResourceMgr::Device>(UID uid) {
        assert(Exist(uid, Device));
        return Ref<GPUMemMgr>(_->gpu_mgrs.at(uid).get());
    }

    template<>
    Ref<HostMemMgr> ResourceMgr::GetResourceMgrRef<HostMemMgr, ResourceMgr::Host>(UID uid) {
        assert(Exist(uid, Host));
        return Ref<HostMemMgr>(_->host_mgrs.at(uid).get());
    }

    std::vector<ResourceMgr::UID> ResourceMgr::GetAll() const {
        std::vector<UID> all;
        for(auto& [uid, _] : _->host_mgrs){
            all.push_back(uid);
        }
        for(auto &[uid, _] : _->gpu_mgrs){
            all.push_back(uid);
        }
        return all;
    }


VISER_END


