#include <Core/HostMemMgr.hpp>

VISER_BEGIN

    class HostMemMgrPrivate{

    };

    HostMemMgr::HostMemMgr(const HostMemMgr::HostMemMgrCreateInfo &info) {
        _ = std::make_unique<HostMemMgrPrivate>();

    }

    HostMemMgr::~HostMemMgr() {

    }

    void HostMemMgr::Lock() {

    }

    void HostMemMgr::UnLock() {

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


VISER_END


