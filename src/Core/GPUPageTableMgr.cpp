#include <Core/GPUPageTableMgr.hpp>

VISER_BEGIN

class GPUPageTableMgrPrivate{
public:
    struct TexCoordIndex{
        uint32_t tid;
        UInt3 coord;
    };
    using BlockUID = GridVolume::BlockUID;
    struct BlockInfo{
        vutil::rw_spinlock_t rw_lk;
    };

//    std::unordered_map<uint32_t, vutil::read_indepwrite_locker> tex_rw;

    std::unordered_map<uint32_t, std::unordered_map<UInt3, BlockInfo>> tex_table;

    std::queue<TexCoordIndex> freed;

    std::unique_ptr<vutil::lru_t<BlockUID, TexCoordIndex>> lru;

    //被cache机制淘汰的才会被erase
    std::unordered_map<BlockUID, TexCoordIndex> record_block_mp;

    size_t total_items = 0;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 0;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUPageTableMgr);
    }

    UnifiedRescUID uid;
    std::mutex g_mtx;
};

GPUPageTableMgr::GPUPageTableMgr(const GPUPageTableMgrCreateInfo& info) {
    _ = std::make_unique<GPUPageTableMgrPrivate>();

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

void GPUPageTableMgr::GetAndLock(const std::vector<Key> &keys, std::vector<PageTableItem> &items) {
    if(keys.size() > _->total_items){
        throw std::runtime_error("Too many keys for GPUPageTableMgr to GetAndLock");
    }

    items.clear();

    for(auto key : keys){
        auto value = _->lru->get_value(key);
        if(value.has_value()){
            auto [tid, coord] = value.value();
            _->tex_table[tid][coord].rw_lk.lock_read();

            items.push_back({key, {.sx = coord.x,
                                   .sy = coord.y,
                                   .sz = coord.z,
                                   .tid = static_cast<uint16_t>(tid),
                                   .flag = static_cast<uint16_t>(0u | TexCoordFlag_IsValid)}});
        }
        else{
            if(_->freed.empty()){
                auto [block_uid, tex_coord] = _->lru->back();
                auto [tid, coord] = tex_coord;
                _->tex_table[tid][coord].rw_lk.lock_write();
                _->record_block_mp.erase(block_uid);
                _->record_block_mp[key] = tex_coord;
                _->lru->emplace_back(key, {tid, coord});

                items.push_back({key, {.sx = coord.x,
                                       .sy = coord.y,
                                       .sz = coord.z,
                                       .tid = static_cast<uint16_t>(tid),
                                       .flag = static_cast<uint16_t>(0u)}});
            }
            else {
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


}

void GPUPageTableMgr::Release(const std::vector<Key>& keys) {
    for(auto& key : keys){
        auto& [tid, coord] = _->record_block_mp.at(key);
        _->tex_table[tid][coord].rw_lk.unlock_read();
    }
}

void GPUPageTableMgr::Promote(const Key &key) {
    auto& [tid, coord] = _->record_block_mp.at(key);
    _->tex_table[tid][coord].rw_lk.converse_write_to_read();
    //暂时不更新缓存优先级

}


VISER_END