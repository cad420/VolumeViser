#include <Core/FixedHostMemMgr.hpp>

VISER_BEGIN

class FixedHostMemMgrPrivate{
public:

};

FixedHostMemMgr::FixedHostMemMgr(const FixedHostMemMgrCreateInfo &info) {

}

FixedHostMemMgr::~FixedHostMemMgr() {

}

void FixedHostMemMgr::Lock() {

}

void FixedHostMemMgr::UnLock() {

}

Handle<CUDAHostBuffer> FixedHostMemMgr::GetBlock(UnifiedRescUID uid, bool emplace) {
    return Handle<CUDAHostBuffer>();
}

VISER_END


