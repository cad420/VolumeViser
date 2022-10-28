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
        int vtex_block_length; // block的长度
        size_t vtex_block_size_bytes; //一个block占据的字节大小
        Int3 vtex_block_dim;
        bool exclusive;
    };

    std::unique_ptr<GPUPageTableMgr> pt_mgr;

    cub::cu_context ctx;

    cub::cu_stream transfer_stream;

    cub::cu_submitted_tasks submitted_tasks;

    std::mutex g_mtx;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUVTexMgr);
    }

    auto GetTransferInfo(const GPUVTexMgr::TexCoord& tex_coord){
        cub::memory_transfer_info info;
        uint32_t tid = tex_coord.tid;
        info.src_x_bytes = 0;
        info.src_y = 0;
        info.src_z = 0;
        info.width_bytes = vtex_block_length * vtex_ele_size;
        info.height = vtex_block_length;
        info.depth = vtex_block_length;
        return std::make_tuple(tid, info);
    };
};

inline auto GetFormat(int bits_per_sampler, bool is_float){
    if(bits_per_sampler == 8)
        return cub::e_uint8;
    else if(bits_per_sampler == 16)
        return cub::e_uint16;
    else if(bits_per_sampler == 32){
        if(is_float) return cub::e_float;
        else assert(false);
    }
}

GPUVTexMgr::GPUVTexMgr(const GPUVTexMgrCreateInfo &info) {
    assert(info.vtex_count > 0 && info.vtex_block_length > 0
    && info.bits_per_sample > 0 && info.samples_per_channel > 0
    && info.vtex_shape.x % info.vtex_block_length == 0
    && info.vtex_shape.y % info.vtex_block_length == 0
    && info.vtex_shape.z % info.vtex_block_length == 0);

    _ = std::make_unique<GPUVTexMgrPrivate>();


    GPUMemMgr::TextureCreateInfo tex_info{
        .resc_info ={
                .format = GetFormat(info.bits_per_sample, info.is_float),
                .channels = (uint32_t)info.samples_per_channel,
                .extent = {(uint32_t)info.vtex_shape.x,
                           (uint32_t)info.vtex_shape.y,
                           (uint32_t)info.vtex_shape.z},
         },
         .view_info = {
                .address = cub::e_clamp,
                .filter = cub::e_linear,
                .read = cub::e_normalized_float,
                .normalized_coords = true
        }
    };
    for(int i = 0; i < info.vtex_count; i++){
        auto tex_handle = info.gpu_mem_mgr->_AllocTexture(RescAccess::Shared, tex_info);
        auto tex_uid = tex_handle.GetUID();
        assert(_->tex_mp.count(tex_uid) == 0);
        _->tex_mp[tex_uid] = std::move(tex_handle);
    }

    GPUPageTableMgr::GPUPageTableMgrCreateInfo pt_info{
        .vtex_count = info.vtex_count,
        .vtex_block_dim = info.vtex_shape / info.vtex_block_length
    };
    _->pt_mgr = std::make_unique<GPUPageTableMgr>(pt_info);

    _->vtex_count = info.vtex_count;
    _->vtex_shape = info.vtex_shape;
    _->vtex_ele_size = info.samples_per_channel * info.bits_per_sample / 8;
    _->vtex_block_length = info.vtex_block_length;
    _->vtex_block_dim = info.vtex_shape / info.vtex_block_length;
    _->vtex_block_size_bytes = (size_t)_->vtex_ele_size * _->vtex_block_length * _->vtex_block_length * _->vtex_block_length;

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
    //使用异步的memcpy，要不要同步等待其全部完成，可以之后测试
    //因为可能渲染启动的时候，拷贝没有完成，内部也会做一些同步，比如读之前必须等待写完成
    auto [tid, transfer_info] = _->GetTransferInfo(dst);
    cub::cu_memory_transfer(*src, *_->tex_mp.at(tid), transfer_info).launch(_->transfer_stream);

}

void GPUVTexMgr::UploadBlockToGPUTexAsync(Handle<CUDAHostBuffer> src, GPUVTexMgr::TexCoord dst) {
    auto [tid, transfer_info] = _->GetTransferInfo(dst);
    _->submitted_tasks.add(
            cub::cu_memory_transfer(*src, *_->tex_mp.at(tid), transfer_info)
            .launch_async(_->transfer_stream));

}

void GPUVTexMgr::Flush() {
    auto ret = _->submitted_tasks.wait();
    for(const auto& r : ret){
        if(r.error()){
            LOG_ERROR("{}, {}", r.name(), r.msg());
        }
    }
    LOG_DEBUG("GPUVTexMgr Flush...");
}


std::vector<GPUVTexMgr::GPUVTex> GPUVTexMgr::GetAllTextures() {
    std::vector<GPUVTexMgr::GPUVTex> res;
    for(auto& item : _->tex_mp){
        res.emplace_back(item);
    }
    return res;
}




VISER_END

