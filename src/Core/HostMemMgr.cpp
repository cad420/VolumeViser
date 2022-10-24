#include <Core/HostMemMgr.hpp>

VISER_BEGIN

    class HostMemMgrPrivate{
    public:


        std::mutex g_mtx;

        UnifiedRescUID uid;
        static UnifiedRescUID GenRescUID(){
            static std::atomic<size_t> g_uid = 0;
            auto uid = g_uid.fetch_add(1);
            return GenUnifiedRescUID(uid, UnifiedRescType::HostMemMgr);
        }
    };

    HostMemMgr::HostMemMgr(const HostMemMgr::HostMemMgrCreateInfo &info) {
        _ = std::make_unique<HostMemMgrPrivate>();



        _->uid = _->GenRescUID();
    }

    HostMemMgr::~HostMemMgr() {

    }

    void HostMemMgr::Lock() {
        _->g_mtx.lock();
    }

    void HostMemMgr::UnLock() {
        _->g_mtx.unlock();
    }

    UnifiedRescUID HostMemMgr::GetUID() const {
        return _->uid;
    }

    UnifiedRescUID HostMemMgr::RegisterFixedHostMemMgr(const HostMemMgr::FixedHostMemMgrCreateInfo &info) {
        return 0;
    }

    Ref<FixedHostMemMgr> HostMemMgr::GetFixedHostMemMgrRef(UnifiedRescUID uid) {
        return Ref<FixedHostMemMgr>();
    }

    UnifiedRescUID HostMemMgr::RegisterGridVolume(const HostMemMgr::GridVolumeCreateInfo &info) {
        return 0;
    }

    Ref<GridVolume> HostMemMgr::GetGridVolumeRef(UnifiedRescUID uid) {
        return Ref<GridVolume>();
    }

    template<>
    Handle<CUDAHostBuffer> HostMemMgr::AllocHostMem<CUDAHostBuffer, HostMemMgr::Pinned>(RescAccess access, size_t bytes){

    }



VISER_END


