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




VISER_END


