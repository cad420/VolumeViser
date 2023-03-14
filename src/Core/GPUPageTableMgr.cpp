#include <Core/GPUPageTableMgr.hpp>
#include <Core/HashPageTable.hpp>
VISER_BEGIN

class GPUPageTableMgrPrivate{
public:
    struct TexCoordIndex{
        uint32_t tid;
        UInt3 coord;
    };
    using BlockUID = GridVolume::BlockUID;
    using ReadWriteLock = vutil::rw_spinlock_t;

    struct BlockInfo{
        ReadWriteLock rw_lk;
        BlockUID block_uid;
    };



    std::unordered_map<uint32_t, std::unordered_map<UInt3, BlockInfo>> tex_table;

    std::queue<TexCoordIndex> freed;

    std::unique_ptr<vutil::lru_t<BlockUID, TexCoordIndex>> lru;

    //被cache机制淘汰的才会被erase
    //应该是记录加了读写锁的项目
    //这里被记录的都是实际存在显存当中的
    std::unordered_map<BlockUID, TexCoordIndex> record_block_mp;

    std::unique_ptr<HashPageTable> hpt;

    size_t total_items = 0;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUPageTableMgr);
    }

    UnifiedRescUID uid;
    std::mutex g_mtx;

    struct{
        int vtex_count;
        Int3 vtex_block_dim;
    };
};

GPUPageTableMgr::GPUPageTableMgr(const GPUPageTableMgrCreateInfo& info) {
    _ = std::make_unique<GPUPageTableMgrPrivate>();

    _->hpt = std::make_unique<HashPageTable>(Ref<GPUMemMgr>(info.gpu_mem_mgr._get_ptr(), false),
                                             Ref<HostMemMgr>(info.host_mem_mgr._get_ptr(), false));

    _->total_items = info.vtex_count * info.vtex_block_dim.x * info.vtex_block_dim.y * info.vtex_block_dim.z;

    _->lru = std::make_unique<vutil::lru_t<GridVolume::BlockUID, GPUPageTableMgrPrivate::TexCoordIndex>>(_->total_items);

    for(uint32_t tid = 0; tid < info.vtex_count; ++tid){
        for(uint32_t ix = 0; ix < info.vtex_block_dim.x; ++ix){
            for(uint32_t iy = 0; iy < info.vtex_block_dim.y; ++iy){
                for(uint32_t iz = 0; iz < info.vtex_block_dim.z; ++iz){
                    _->freed.emplace(GPUPageTableMgrPrivate::TexCoordIndex{tid,{ix, iy, iz}});
                }
            }
        }
    }
    _->vtex_count = info.vtex_count;
    _->vtex_block_dim = info.vtex_block_dim;

    _->uid = _->GenRescUID();
}

GPUPageTableMgr::~GPUPageTableMgr() {

}
void GPUPageTableMgr::Lock() {
    _->g_mtx.lock();
}
void GPUPageTableMgr::UnLock() {
    _->g_mtx.unlock();
}

UnifiedRescUID GPUPageTableMgr::GetUID() const {
    return _->uid;
}


void GPUPageTableMgr::GetAndLock(const std::vector<Key> &keys, std::vector<PageTableItem> &items) {
    if(keys.size() > _->total_items){
        throw std::runtime_error("Too many keys for GPUPageTableMgr to GetAndLock");
    }
    try {
        for (auto key: keys) {
            if (auto value = _->lru->get_value(key); value.has_value()) {
                auto [tid, coord] = value.value();
                auto& lk = _->tex_table[tid][coord].rw_lk;
                uint32_t flags = 0;
                if(lk.is_write_locked()){
                    //如果已经被加了写锁 那么不用等待 直接返回
                }
                else{
                    lk.lock_read();
                    flags |= TexCoordFlag_IsValid;
                }

                //可能被Clear过
                _->record_block_mp[key] = {tid, coord};

                items.push_back({key, {.sx = coord.x,
                        .sy = coord.y,
                        .sz = coord.z,
                        .tid = static_cast<uint16_t>(tid),
                        .flag = static_cast<uint16_t>(flags)}});
            } else {
                if (_->freed.empty()) {
                    auto [block_uid, tex_coord] = _->lru->back();
                    auto [tid, coord] = tex_coord;
                    _->tex_table[tid][coord].rw_lk.lock_write();
                    _->record_block_mp.erase(block_uid);
                    _->record_block_mp[key] = tex_coord;
                    _->lru->emplace_back(key, {tid, coord});

                    LOG_DEBUG("using lru for freed empty");

                    items.push_back({key, {.sx = coord.x,
                            .sy = coord.y,
                            .sz = coord.z,
                            .tid = static_cast<uint16_t>(tid),
                            .flag = static_cast<uint16_t>(0u)}});
                } else {
                    auto [tid, coord] = _->freed.front();
                    _->freed.pop();
                    _->tex_table[tid][coord].rw_lk.lock_write();
                    _->record_block_mp[key] = {tid, coord};
                    _->lru->emplace_back(key, {tid, coord});
                    items.push_back({key, {.sx = coord.x,
                            .sy = coord.y,
                            .sz = coord.z,
                            .tid = static_cast<uint16_t>(tid),
                            .flag = static_cast<uint16_t>(0u)}});
                };
            }
        }
    }catch(const std::exception& err){
        LOG_ERROR("GetAndLock error : {}", err.what());
        exit(0);
    }
}

void GPUPageTableMgr::Release(const std::vector<Key>& keys, bool readonly) {
    for(auto& key : keys){
        auto& [tid, coord] = _->record_block_mp.at(key);
        auto& lk = _->tex_table[tid][coord].rw_lk;
        if(lk.is_read_locked())
            lk.unlock_read();
        else if(!readonly && lk.is_write_locked())
            lk.unlock_write();
    }
}

HashPageTable& GPUPageTableMgr::GetPageTable(bool update) {
    //todo 消除不必要的项 否则哈希表会爆满
    if(update){
        _->hpt->Clear();
        for(auto& [block_uid, tex_coord] : _->record_block_mp){
            auto& [tid, coord] = tex_coord;
            if(_->tex_table[tid][coord].rw_lk.is_write_locked()){
                _->hpt->Append({block_uid, {coord.x, coord.y, coord.z, (uint16_t)tid, 0}});
            }
            else{
                uint16_t flag = TexCoordFlag_IsValid;
                if(block_uid.IsBlack())  flag |= TexCoordFlag_IsBlack;
                if(block_uid.IsSparse()) flag |= TexCoordFlag_IsSparse;
                if(block_uid.IsSWC())    flag |= TexCoordFlag_IsSWC;
                _->hpt->Append({block_uid, {coord.x, coord.y, coord.z, (uint16_t)tid, flag}});
            }
        }
    }
    return *_->hpt;
}

void GPUPageTableMgr::Promote(const Key &key) {
    auto& [tid, coord] = _->record_block_mp.at(key);
    LOG_DEBUG("promote start");
    auto& lk = _->tex_table[tid][coord].rw_lk;
    if(lk.is_write_locked())
        lk.converse_write_to_read();
    else if(lk.is_read_locked()){
        LOG_ERROR("promote lk is read locked");
//        throw std::runtime_error("promote lk is read locked");
    }
    else{
        LOG_ERROR("promote lk is not locked");
//        throw std::runtime_error("promote lk is not locked");
    }

    //暂时不更新缓存优先级
    LOG_ERROR("promote ok");

}
bool GPUPageTableMgr::Check(const GPUPageTableMgr::Key &key, const GPUPageTableMgr::Value &value)
{
    if(_->record_block_mp.count(key) == 0) return false;
    auto& record_val = _->record_block_mp.at(key);
    if(record_val.tid != value.tid || record_val.coord != UInt3{value.sx, value.sy, value.sz}) return false;
    return true;
}
void GPUPageTableMgr::Reset()
{
    _->record_block_mp.clear();
    _->lru->clear();

    while(!_->freed.empty()) _->freed.pop();

    for(uint32_t tid = 0; tid < _->vtex_count; ++tid){
        for(uint32_t ix = 0; ix < _->vtex_block_dim.x; ++ix){
            for(uint32_t iy = 0; iy < _->vtex_block_dim.y; ++iy){
                for(uint32_t iz = 0; iz < _->vtex_block_dim.z; ++iz){
                    _->freed.emplace(GPUPageTableMgrPrivate::TexCoordIndex{tid,{ix, iy, iz}});
                }
            }
        }
    }

    _->hpt->Clear();
}
void GPUPageTableMgr::ClearWithOption(std::function<bool(const Key& key)> ok)
{
    for(auto it = _->record_block_mp.begin(); it != _->record_block_mp.end();){
        if(ok(it->first)){
            _->record_block_mp.erase(it++);
        }
        else
            it++;
    }
}

VISER_END