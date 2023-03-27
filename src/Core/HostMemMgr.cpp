#include <Core/HostMemMgr.hpp>

VISER_BEGIN

class HostMemMgrPrivate{
public:
    std::atomic<size_t> used_mem_bytes = 0;
    size_t max_mem_bytes = 0;

    //只用于分配全局的pinned host memory
    CUDAContext ctx;

    std::unordered_map<UnifiedRescUID, std::unique_ptr<FixedHostMemMgr>> fixed_host_mgr_mp;

    std::mutex g_mtx;

    UnifiedRescUID uid;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::HostMemMgr);
    }
};

HostMemMgr::HostMemMgr(const HostMemMgr::HostMemMgrCreateInfo &info) {
    auto free = vutil::get_free_memory_bytes();
    if(free < info.MaxCPUMemBytes){
        throw ViserResourceCreateError("Create CPUMemMgr exception : free memory is not enough for require size " + std::to_string(info.MaxCPUMemBytes));
    }

    _ = std::make_unique<HostMemMgrPrivate>();

    _->max_mem_bytes = info.MaxCPUMemBytes;

    auto devs = cub::cu_physical_device::get_all_device();
    if(devs.empty()){
        throw ViserResourceCreateError("Create CPUMemMgr exception : no cuda device");
    }
    auto& dev = devs.front();
    _->ctx = dev.create_context(0);

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

template<>
Handle<CUDAHostBuffer> HostMemMgr::AllocHostMem<CUDAHostBuffer, HostMemMgr::Pinned>(ResourceType type, size_t bytes, bool required){
    if(!required){
        auto used = _->used_mem_bytes += bytes;
        if (used > _->max_mem_bytes) {
            _->used_mem_bytes.fetch_sub(bytes);
            throw ViserResourceCreateError(
                    "No enough free memory for HostMemMgr to alloc pinned buffer with size: " + std::to_string(bytes));
        }
    }
    return NewHandle<CUDAHostBuffer>(type, bytes, cub::cu_memory_type::e_cu_host, _->ctx);
}

template<>
Handle<HostBuffer> HostMemMgr::AllocHostMem<HostBuffer, HostMemMgr::Paged>(ResourceType type, size_t bytes, bool required){
    if(!required){
        auto used = _->used_mem_bytes += bytes;
        if (used > _->max_mem_bytes) {
            _->used_mem_bytes.fetch_sub(bytes);
            throw ViserResourceCreateError(
                    "No enough free memory for HostMemMgr to alloc paged buffer with size: " + std::to_string(bytes));
        }
    }
    return NewHandle<HostBuffer>(type, bytes);
}

UnifiedRescUID HostMemMgr::RegisterFixedHostMemMgr(const FixedHostMemMgrCreateInfo &info) {
    try{
        size_t alloc_size = info.fixed_block_size * info.fixed_block_num;
        auto used = _->used_mem_bytes += alloc_size;
        if(used > _->max_mem_bytes){
            _->used_mem_bytes.fetch_sub(alloc_size);
            throw std::runtime_error("No free GPU memory to register FixedHostMemMgr");
        }

        LOG_DEBUG("Register FixedHostMemMgr cost free memory: {}, remain free: {}",
                  alloc_size, _->max_mem_bytes - used);

        info.host_mem_mgr = Ref(this, false);

        auto resc = std::make_unique<FixedHostMemMgr>(info);
        auto uid = resc->GetUID();
        _->fixed_host_mgr_mp[uid] = std::move(resc);
        return uid;
    }
    catch (const std::exception& e) {
        LOG_ERROR("RegisterFixedHostMemMgr failed with create info : (fixed_block_size {}, fixed_block_num {})",
                  info.fixed_block_size, info.fixed_block_num);
        throw ViserResourceCreateError(std::string("Register FixedHostMemMgr exception : ") + e.what());
    }
}

Ref<FixedHostMemMgr> HostMemMgr::GetFixedHostMemMgrRef(UnifiedRescUID uid) {
    return Ref<FixedHostMemMgr>(_->fixed_host_mgr_mp.at(uid).get());
}

VISER_END