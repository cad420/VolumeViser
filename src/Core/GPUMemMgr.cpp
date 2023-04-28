#include <Core/GPUMemMgr.hpp>
#include <unordered_set>

VISER_BEGIN

class GPUMemMgrPrivate{
public:
    std::atomic<size_t> used_mem_bytes = 0;
    size_t max_mem_bytes = 0;

    int gpu_index = -1;
    CUDAContext ctx;

    std::unordered_map<UnifiedRescUID, std::unique_ptr<GPUVTexMgr>> vtex_mgr_mp;

    std::mutex g_mtx;

    UnifiedRescUID uid;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUMemMgr);
    }
};

GPUMemMgr::GPUMemMgr(const GPUMemMgrCreateInfo& info) {
    _ = std::make_unique<GPUMemMgrPrivate>();

    auto devs = cub::cu_physical_device::get_all_device();
    bool find = false;
    auto init = [&](const cub::cu_physical_device& dev){
        _->ctx = dev.create_context(0);
        _->gpu_index = info.GPUIndex;
        _->max_mem_bytes = info.MaxGPUMemBytes;
    };
    for(auto& dev : devs){
        if(dev.get_device_id() == info.GPUIndex){
            find = true;
            init(dev);
            break;
        }
    }
    if(!find){
        throw ViserResourceCreateError("Create GPUMemMgr with invalid GPUIndex : " + std::to_string(info.GPUIndex));
    }

    _->uid = _->GenRescUID();
}

GPUMemMgr::~GPUMemMgr() {

}

void GPUMemMgr::Lock() {
    _->g_mtx.lock();
}

void GPUMemMgr::UnLock() {
    _->g_mtx.unlock();
}

UnifiedRescUID GPUMemMgr::GetUID() const {
    return _->uid;
}

Handle<CUDABuffer> GPUMemMgr::AllocBuffer(ResourceType type, size_t bytes) {
    auto used = _->used_mem_bytes += bytes;
    if(used > _->max_mem_bytes){
        _->used_mem_bytes.fetch_sub(bytes);
        throw ViserResourceCreateError("No enough free memory for GPUMemMgr to alloc buffer with size: " + std::to_string(bytes));
    }
    return NewHandle<CUDABuffer>(type, bytes, cub::cu_memory_type::e_cu_device, _->ctx).AddCallback([this, bytes = bytes]{
        if(this && _) _->used_mem_bytes -= bytes;
    });
}

Handle<CUDAPitchedBuffer> GPUMemMgr::AllocPitchedBuffer(ResourceType type, size_t width, size_t height, size_t ele_size) {
    auto bytes = width * height * ele_size;
    auto used = _->used_mem_bytes += bytes;
    if(used > _->max_mem_bytes){
        _->used_mem_bytes.fetch_sub(bytes);
        throw ViserResourceCreateError("No enough free memory for GPUMemMgr to alloc pitched buffer with size: " + std::to_string(bytes));
    }
    return NewHandle<CUDAPitchedBuffer>(type, width, height, ele_size, _->ctx).AddCallback([this, bytes = bytes]{
        if(this && _) _->used_mem_bytes -= bytes;
    });
}

Handle<CUDATexture> GPUMemMgr::AllocTexture(ResourceType type, const TextureCreateInfo& info) {
    auto bytes = info.resc_info.alloc_bytes();
    auto used = _->used_mem_bytes += bytes;
    if(used > _->max_mem_bytes){
        _->used_mem_bytes.fetch_sub(bytes);
        throw ViserResourceCreateError("No enough free memory for GPUMemMgr to alloc texture with size: " + std::to_string(bytes));
    }
    return NewHandle<CUDATexture>(type, info.resc_info, info.view_info, _->ctx).AddCallback([this, bytes = bytes]{
        if(this && _) _->used_mem_bytes -= bytes;
    });
}

Handle<CUDATexture> GPUMemMgr::_AllocTexture(ResourceType type, const TextureCreateInfo& info) {
    return NewHandle<CUDATexture>(type, info.resc_info, info.view_info, _->ctx);
}

UnifiedRescUID GPUMemMgr::RegisterGPUVTexMgr(const GPUVTexMgrCreateInfo &info) {
    try{
#ifndef USE_LINEAR_BUFFER_FOR_TEXTURE
        size_t alloc_size = (size_t)info.vtex_count * info.vtex_shape.x * info.vtex_shape.y * info.vtex_shape.z
                * info.bits_per_sample * info.samples_per_channel / 8;
        auto used = _->used_mem_bytes += alloc_size;
        if(used > _->max_mem_bytes){
            _->used_mem_bytes.fetch_sub(alloc_size);
            throw std::runtime_error("No free GPU memory to register GPUVTexMgr");
        }
#endif


        info.gpu_mem_mgr = Ref(this, false);
        auto resc = std::make_unique<GPUVTexMgr>(info);
        auto uid = resc->GetUID();
        assert(_->vtex_mgr_mp.count(uid) == 0);
        _->vtex_mgr_mp[uid] = std::move(resc);
#ifndef USE_LINEAR_BUFFER_FOR_TEXTURE
        LOG_DEBUG("Register GPUVTexMgr cost free memory: {}, remain free: {}",
                  alloc_size, _->max_mem_bytes - used);
#endif
        return uid;
    }
    catch (const std::exception& e) {
        LOG_ERROR("RegisterGPUVTexMgr failed with create info: "
                  "\n\t(vtex_count {}, vtex_shape {} {} {},"
                  "\n\tbits_per_sample {}, samplers_per_channle {},"
                  "\n\tvtex_block_length {}, is_float {}, exclusive {})",
                  info.vtex_count, info.vtex_shape.x, info.vtex_shape.y, info.vtex_shape.z,
                  info.bits_per_sample, info.samples_per_channel, info.vtex_block_length, info.is_float, info.exclusive);

        throw ViserResourceCreateError(std::string("RegisterGPUVTexMgr exception : ") + e.what());
    }
}

Ref<GPUVTexMgr> GPUMemMgr::GetGPUVTexMgrRef(UnifiedRescUID uid) {
    assert(CheckUnifiedRescUID(uid));
    return {_->vtex_mgr_mp.at(uid).get()};
}

CUDAContext GPUMemMgr::_get_cuda_context() const {
    return _->ctx;
}
void GPUMemMgr::ReleaseGPUVTexMgr(UnifiedRescUID uid)
{
    //todo
}

VISER_END