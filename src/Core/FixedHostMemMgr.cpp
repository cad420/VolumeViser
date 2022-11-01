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
        auto handle = info.host_mem_mgr->AllocPinnedHostMem(RescAccess::Shared, info.fixed_block_size, true);
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
    auto value = _->lru->get_value(uid);
    if(value.has_value()){
        return value.value();
    }
    else{
        if(_->freed.empty()){
            _->lru->back().first = uid;
        }
        else{
            _->lru->emplace_back(uid, std::move(_->freed.front()));
            _->freed.pop();
        }
        return _->lru->get_value(uid).value();
    }
}

VISER_END