#include <Core/FixedHostMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN

class FixedHostMemMgrPrivate{
public:
    using LRUCache = vutil::lru_t<UnifiedRescUID, Handle<CUDAHostBuffer>>;

    std::queue<Handle<CUDAHostBuffer>> freed;

    std::unique_ptr<LRUCache> lru;

    std::mutex g_mtx;

    UnifiedRescUID uid;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::FixedHostMemMgr);
    }
};

FixedHostMemMgr::FixedHostMemMgr(const FixedHostMemMgrCreateInfo &info) {
    assert(info.fixed_block_num && info.fixed_block_size);

    _ = std::make_unique<FixedHostMemMgrPrivate>();

    _->lru =  std::make_unique<FixedHostMemMgrPrivate::LRUCache>(info.fixed_block_num);

    for(int i = 0; i < info.fixed_block_num; i++){
        auto handle = info.host_mem_mgr->AllocPinnedHostMem(ResourceType::Buffer, info.fixed_block_size, true);
        assert(CheckUnifiedRescUID(handle.GetUID()));
        _->freed.push(std::move(handle));
    }

    _->uid = _->GenRescUID();
}

FixedHostMemMgr::~FixedHostMemMgr() {

}

void FixedHostMemMgr::Lock() {
    _->g_mtx.lock();
}

void FixedHostMemMgr::UnLock() {
    _->g_mtx.unlock();
}

UnifiedRescUID FixedHostMemMgr::GetUID() const {
    return _->uid;
}

Handle<CUDAHostBuffer> FixedHostMemMgr::GetBlock(UnifiedRescUID uid) {
    try {
        auto value = _->lru->get_value(uid);
        LOG_DEBUG("000 {}", uid);
        if (value.has_value()) {
            LOG_DEBUG("111 {}", uid);
            return value.value().AddReadLock();
        } else {
            if (_->freed.empty()) {
                // wait for unlock
                auto& t = _->lru->back();
                while(t.second.IsLocked());
                t.first = uid;
                _->lru->get_value(uid);
                LOG_DEBUG("222 {}", uid);
            } else {
                _->lru->emplace_back(uid, std::move(_->freed.front()));
                _->freed.pop();
                LOG_DEBUG("333 {}", uid);
            }
            LOG_DEBUG("444 {}", uid);
            return _->lru->get_value(uid).value().AddWriteLock();
        }
    } catch (const std::exception& err) {
        auto id = GridVolume::BlockUID(uid);
        LOG_ERROR("GetBlock error : {}, block uid: {}, {} {} {} {}", err.what(),
                  uid, id.x, id.y, id.z, id.GetLOD());
    }
}

VISER_END