#include <Core/GPUVTexMgr.hpp>
#include <Core/GPUPageTableMgr.hpp>
#include <Core/GPUMemMgr.hpp>
VISER_BEGIN

class GPUVTexMgrPrivate{
public:
    using GPUTexUnit = GPUVTexMgr::GPUTexUnit;
    std::unordered_map<GPUTexUnit, Handle<CUDATexture>> tex_mp;

    struct{
        int vtex_count;
        Int3 vtex_shape;
        int vtex_ele_size;// bits_per_sample * sample_per_channel / 8
        int vtex_block_size;
        Int3 vtex_block_dim;
        bool exclusive;
    };

    std::unique_ptr<GPUPageTableMgr> pt_mgr;

    std::mutex g_mtx;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUVTexMgr);
    }
};

inline auto GetFormat(int bits_per_sampler, bool is_float){
    if(bits_per_sampler == 8)
        return CUDATexture::e_uint8;
    else if(bits_per_sampler == 16)
        return CUDATexture::e_uint16;
    else if(bits_per_sampler == 32){
        if(is_float) return CUDATexture::e_float;
        else assert(false);
    }
}

GPUVTexMgr::GPUVTexMgr(const GPUVTexMgrCreateInfo &info) {
    assert(info.vtex_count > 0 && info.vtex_block_size > 0
    && info.bits_per_sample > 0 && info.samples_per_channel > 0
    && info.vtex_shape.x % info.vtex_block_size == 0
    && info.vtex_shape.y % info.vtex_block_size == 0
    && info.vtex_shape.z % info.vtex_block_size == 0);

    _ = std::make_unique<GPUVTexMgrPrivate>();


    GPUMemMgr::TextureCreateInfo tex_info{
        .format = GetFormat(info.bits_per_sample, info.is_float),
        .channels = (uint32_t)info.samples_per_channel,
        .address = CUDATexture::e_clamp,
        .filter = CUDATexture::e_linear,
        .read = CUDATexture::e_normalized_float,
        .extent = {(uint32_t)info.vtex_shape.x, (uint32_t)info.vtex_shape.y, (uint32_t)info.vtex_shape.z},
        .type = cub::cu_array3d_type::ARRAY3D,
        .normalized_coords = true
    };
    for(int i = 0; i < info.vtex_count; i++){
        auto tex_handle = info.gpu_mem_mgr->_AllocTexture(RescAccess::Shared, tex_info);
        auto tex_uid = tex_handle.GetUID();
        assert(_->tex_mp.count(tex_uid) == 0);
        _->tex_mp[tex_uid] = std::move(tex_handle);
    }

    GPUPageTableMgr::GPUPageTableMgrCreateInfo pt_info{
        .vtex_count = info.vtex_count,
        .vtex_block_dim = info.vtex_shape / info.vtex_block_size
    };
    _->pt_mgr = std::make_unique<GPUPageTableMgr>(pt_info);

    _->vtex_count = info.vtex_count;
    _->vtex_shape = info.vtex_shape;
    _->vtex_ele_size = info.samples_per_channel * info.bits_per_sample / 8;
    _->vtex_block_size = _->vtex_block_size;
    _->vtex_block_dim = info.vtex_shape / info.vtex_block_size;

    _->GenRescUID();
}

GPUVTexMgr::~GPUVTexMgr() {

}

void GPUVTexMgr::Lock() {
    _->g_mtx.lock();
}

void GPUVTexMgr::UnLock() {
    _->g_mtx.unlock();
}

UnifiedRescUID GPUVTexMgr::GetUID() const {
    return _->uid;
}

Ref<GPUPageTableMgr> GPUVTexMgr::GetGPUPageTableMgrRef() {
    return Ref<GPUPageTableMgr>(_->pt_mgr.get());
}

void GPUVTexMgr::UploadBlockToGPUTex(Handle<CUDAHostBuffer> src, GPUVTexMgr::TexCoord dst) {

}

std::vector<GPUVTexMgr::GPUVTex> GPUVTexMgr::GetAllTextures() {
    std::vector<GPUVTexMgr::GPUVTex> res;
    for(auto& item : _->tex_mp){
        res.emplace_back(item);
    }
    return res;
}




VISER_END

