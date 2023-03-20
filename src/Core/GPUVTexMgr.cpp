#include <Core/GPUVTexMgr.hpp>
#include <Core/GPUPageTableMgr.hpp>
#include <Core/GPUMemMgr.hpp>
#include <Core/HashPageTable.hpp>
VISER_BEGIN

#ifdef VISER_RENDER_ONLY
#define VTEX_READ_WRITE 0
#else
#define VTEX_READ_WRITE 1
#endif

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

    CUDAContext ctx;

    CUDAStream transfer_stream;

    cub::cu_submitted_tasks submitted_tasks;
    std::unordered_map<size_t, GPUPageTableMgr::TexCoord> submitted_items;

    Handle<CUDABuffer> cu_black_buffer;

    std::mutex g_mtx;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUVTexMgr);
    }

    auto GetTransferInfo(const GPUVTexMgr::TexCoord& tex_coord){
        cub::memory_transfer_info info;
        uint32_t tid = tex_coord.tid;
        info.src_x_bytes = 0;
        info.src_y = 0;
        info.src_z = 0;
        info.dst_x_bytes = tex_coord.sx * vtex_block_length * vtex_ele_size;
        info.dst_y = tex_coord.sy * vtex_block_length;
        info.dst_z = tex_coord.sz * vtex_block_length;
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

    _->ctx = info.gpu_mem_mgr._get_ptr()->_get_cuda_context();

    _->transfer_stream = cub::cu_stream(_->ctx);

    GPUMemMgr::TextureCreateInfo tex_info{
        .resc_info ={
                .format = GetFormat(info.bits_per_sample, info.is_float),
                .channels = (uint32_t)info.samples_per_channel,
                .extent = {(uint32_t)info.vtex_shape.x,
                           (uint32_t)info.vtex_shape.y,
                           (uint32_t)info.vtex_shape.z},
                           .read_write = VTEX_READ_WRITE
         },
         .view_info = {
                .address = cub::e_border,
                .filter = cub::e_linear,
                .read = cub::e_normalized_float,
                .normalized_coords = true
        }
    };
    for(int i = 0; i < info.vtex_count; i++){
        auto tex_handle = info.gpu_mem_mgr.Invoke(&GPUMemMgr::_AllocTexture, ResourceType::Buffer, tex_info);
        auto tex_uid = i;
        assert(_->tex_mp.count(tex_uid) == 0);
        _->tex_mp[tex_uid] = std::move(tex_handle);
    }

    GPUPageTableMgr::GPUPageTableMgrCreateInfo pt_info{
        .gpu_mem_mgr = Ref<GPUMemMgr>(info.gpu_mem_mgr._get_ptr(), false),
        .host_mem_mgr = Ref<HostMemMgr>(info.host_mem_mgr._get_ptr(), false),
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

    _->cu_black_buffer = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, _->vtex_block_size_bytes);
    CUB_CHECK(cudaMemset(_->cu_black_buffer->get_data(), 0, _->vtex_block_size_bytes));

    _->uid = _->GenRescUID();
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

bool GPUVTexMgr::UploadBlockToGPUTex(Handle<CUDAHostBuffer> src, GPUVTexMgr::TexCoord dst) {
    //使用异步的memcpy，要不要同步等待其全部完成，可以之后测试
    //因为可能渲染启动的时候，拷贝没有完成，内部也会做一些同步，比如读之前必须等待写完成
    auto [tid, transfer_info] = _->GetTransferInfo(dst);

    //还是要检查上传的数据是否仍需要 是否有效
    auto block_uid = GridVolume::BlockUID(src.GetUID());
    auto __ = GetGPUPageTableMgrRef().LockRef().AutoLock();
    if(_->pt_mgr->Check(block_uid, dst)){
        cub::cu_memory_transfer(*src, *_->tex_mp.at(tid), transfer_info).launch(_->transfer_stream);
        _->pt_mgr->Promote(block_uid);
        return true;
    }
    else{
        LOG_ERROR("block upload to vtex but not needed any more");
        return false;
    }
}

void GPUVTexMgr::UploadBlockToGPUTexAsync(Handle<CUDAHostBuffer> src, GPUVTexMgr::TexCoord dst) {
    auto [tid, transfer_info] = _->GetTransferInfo(dst);
    _->submitted_tasks.add(
            cub::cu_memory_transfer(*src, *_->tex_mp.at(tid), transfer_info)
            .launch_async(_->transfer_stream), src.GetUID());
    _->submitted_items[src.GetUID()] = dst;
}

void GPUVTexMgr::Flush() {
    //这里调用GPUPageTableMgr的Promote 将写锁改为读锁
    auto ret = _->submitted_tasks.wait();
    for(const auto& [uid, r] : ret){
        if(r.error()){
            LOG_ERROR("{}, {}", r.name(), r.msg());
        }
        else{
            auto tex_coord = _->submitted_items[uid];
            tex_coord.flag = TexCoordFlag_IsValid;
            _->submitted_items.erase(uid);
//            _->pt_mgr->GetPageTable().Update({GridVolume::BlockUID(uid), tex_coord});
            _->pt_mgr->Promote(GridVolume::BlockUID(uid));
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

void GPUVTexMgr::Clear(UnifiedRescUID uid, TexCoord dst) {
    auto [tid, transfer_info] = _->GetTransferInfo(dst);
    cub::cu_memory_transfer(*_->cu_black_buffer, *_->tex_mp.at(tid), transfer_info)
    .launch(_->transfer_stream).check_error_on_throw();
}


VISER_END