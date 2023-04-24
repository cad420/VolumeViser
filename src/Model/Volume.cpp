#include <Model/Volume.hpp>
#include <Core/GPUMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN

class GridVolumePrivate{
public:
    UnifiedRescUID host_mem_mgr_uid;
    UnifiedRescUID gpu_mem_mgr_uid;

    Ref<GPUMemMgr> gpu_mem_mgr;
    Ref<HostMemMgr> host_mem_mgr;

    std::unordered_map<uint32_t, Handle<VolumeIOInterface>> lod_vol_file_io;

    using GridVolumeDesc = GridVolume::GridVolumeDesc;
    GridVolumeDesc desc;

    size_t block_bytes;

    UnifiedRescUID uid;

    static UnifiedRescUID GenRescUID() {
        static std::atomic<size_t> g_uid = 1;
        auto _ = g_uid.fetch_add(1);
        return GenUnifiedRescUID(_, UnifiedRescType::GridVolume);
    }

    bool CheckValidation(const GridVolume::BlockUID& block_uid) const{
        return block_uid.IsValid()
        && block_uid.x < desc.blocked_dim.x
        && block_uid.y < desc.blocked_dim.y
        && block_uid.z < desc.blocked_dim.z
        && lod_vol_file_io.count(block_uid.w) != 0;
    }

    auto GetReadRegion(const GridVolume::BlockUID& block_uid) const{
        auto idx = Int3(block_uid.x, block_uid.y, block_uid.z);
        auto beg_pos = idx * (int)desc.block_length - (int)desc.padding;
        auto end_pos = (idx + 1) * (int)desc.block_length + (int)desc.padding;
        return std::make_pair(beg_pos, end_pos);
    }

    size_t CalcRegionSizeBytes(Int3 beg_pos, Int3 end_pos){
        auto region = end_pos - beg_pos;
        return (size_t)region.x * region.y * region.z * desc.bits_per_sample * desc.samples_per_voxel / 8;
    }

    std::mutex g_mtx;
};


GridVolume::GridVolume(const GridVolume::GridVolumeCreateInfo &info) {
    _ = std::make_unique<GridVolumePrivate>();

    _->lod_vol_file_io = info.lod_vol_file_io;
    _->host_mem_mgr_uid = info.host_mem_mgr_uid;
    _->gpu_mem_mgr_uid = info.gpu_mem_mgr_uid;
    _->gpu_mem_mgr = std::move(info.gpu_mem_mgr);
    _->host_mem_mgr = std::move(info.host_mem_mgr);

    _->uid = _->GenRescUID();

    for(uint32_t lod = 0; lod < info.levels; lod++){
        assert(_->lod_vol_file_io.count(lod) != 0);
    }

    _->desc = _->lod_vol_file_io.at(0u)->GetVolumeDesc();

    size_t block_size = _->desc.block_length + _->desc.padding * 2;
    _->block_bytes = block_size * block_size * block_size * _->desc.bits_per_sample * _->desc.samples_per_voxel / 8;


}

GridVolume::~GridVolume() {

}

void GridVolume::Lock() {
    _->g_mtx.lock();
}

void GridVolume::UnLock() {
    _->g_mtx.unlock();
}

GridVolume::GridVolumeDesc GridVolume::GetDesc() const {
    return _->desc;
}

UnifiedRescUID GridVolume::GetUID() const {
    return _->uid;
}

//CUDAHostBuffer GridVolume::ReadBlock(const GridVolume::BlockUID &uid) {
//    NOT_IMPL
//    return viser::CUDAHostBuffer(0, cub::e_cu_device, cub::cu_context());
//}


//CUDAHostBuffer GridVolume::ReadRegion(Int3 beg, Int3 end, uint32_t lod) {
//    NOT_IMPL
//    return viser::CUDAHostBuffer(0, cub::e_cu_device, cub::cu_context());
//}



void GridVolume::ReadBlock(const GridVolume::BlockUID &uid, CUDAHostBuffer &buffer) {
    assert(_->CheckValidation(uid));
    assert(buffer.get_size() == _->block_bytes);

    auto lod = uid.GetLOD();
    auto& file = _->lod_vol_file_io.at(lod);

    //因为文件读取是串行的，因此需要再加个锁，即多个一起读也是不可以的
    file->Lock();

    auto [beg_pos, end_pos] = _->GetReadRegion(uid);

    _->lod_vol_file_io.at(lod)->ReadVolumeRegion(beg_pos, end_pos, buffer.get_data());

    file->UnLock();
}

void GridVolume::ReadRegion(Int3 beg, Int3 end, uint32_t lod, CUDAHostBuffer &buffer) {
    assert(beg.x < end.x && beg.y < end.y && beg.z < end.z);
    assert(buffer.get_size() == _->CalcRegionSizeBytes(beg, end));
    assert(_->lod_vol_file_io.count(lod));

    auto& file = _->lod_vol_file_io.at(lod);

    //因为文件读取是串行的，因此需要再加个锁，即多个一起读也是不可以的
    file->Lock();

    _->lod_vol_file_io.at(lod)->ReadVolumeRegion(beg, end, buffer.get_data());

    file->UnLock();
}

//CUDAPitchedBuffer GridVolume::ReadBlockGPU(const GridVolume::BlockUID &uid) {
//    NOT_IMPL
//    return viser::CUDAPitchedBuffer(0, 0, 0, cub::cu_context());
//}

//CUDAPitchedBuffer GridVolume::ReadRegionGPU(Int3 beg, Int3 end, uint32_t lod) {
//    NOT_IMPL
//    return viser::CUDAPitchedBuffer(0, 0, 0, cub::cu_context());
//}

void GridVolume::ReadBlockGPU(const GridVolume::BlockUID &uid, CUDAPitchedBuffer &buffer) {
    NOT_IMPL
}

void GridVolume::ReadRegion(Int3 beg, Int3 end, uint32_t lod, CUDAPitchedBuffer &buffer) {
    NOT_IMPL
}
int GridVolume::GetMaxLOD() const noexcept
{
    return _->lod_vol_file_io.size() - 1;
}

bool GridVolume::BlockUID::IsValid() const {
    return *this != GridVolume::INVALID_BLOCK_UID;
}


VISER_END