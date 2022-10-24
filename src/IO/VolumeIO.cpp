#include <IO/VolumeIO.hpp>

#include <VolumeUtils/Volume.hpp>

VISER_BEGIN

class EBVolumeFilePrivate{
public:
    using VolumeDesc = EBVolumeFile::VolumeDesc;

    VolumeDesc desc;
    std::unique_ptr<vol::EncodedBlockedGridVolumeReader> vol_reader;

    UnifiedRescUID uid;

    std::mutex g_mtx;
};

EBVolumeFile::EBVolumeFile(std::string_view filename) {
    _ = std::make_unique<EBVolumeFilePrivate>();

    _->vol_reader = std::make_unique<vol::EncodedBlockedGridVolumeReader>(filename.data());

    auto eb_desc = _->vol_reader->GetVolumeDesc();

    _->desc.volume_name = eb_desc.volume_name;
    _->desc.bits_per_sample = vol::GetVoxelBits(eb_desc.voxel_info.type);
    _->desc.samples_per_voxel = vol::GetVoxelSampleCount(eb_desc.voxel_info.format);
    _->desc.is_float = eb_desc.voxel_info.type == vol::VoxelType::float32;
    _->desc.shape = {eb_desc.extend.width, eb_desc.extend.height, eb_desc.extend.depth};
    _->desc.block_length = eb_desc.block_length;
    _->desc.padding = eb_desc.padding;
    _->desc.blocked_dim = (_->desc.shape + _->desc.block_length - UInt3(1)) / _->desc.block_length;
    _->desc.decoding_cpu_only = true;

    _->uid = VolumeIOInterface::GenUnifiedRescUID();
}

EBVolumeFile::~EBVolumeFile() {

}

void EBVolumeFile::Lock() {
    _->g_mtx.lock();
}

void EBVolumeFile::UnLock() {
    _->g_mtx.unlock();
}

UnifiedRescUID EBVolumeFile::GetUID() {
    return _->uid;
}

EBVolumeFile::VolumeDesc EBVolumeFile::GetVolumeDesc() {
    return _->desc;
}

void EBVolumeFile::ReadVolumeRegion(const Int3 &beg_pos, const Int3 &end_pos, void *ptr) {
    Int3 block_beg_pos = beg_pos + Int3(_->desc.padding);
    Int3 block_idx = block_beg_pos / (int)_->desc.block_length;
    auto region = end_pos - beg_pos;
    size_t bytes = (size_t)_->desc.bits_per_sample * _->desc.samples_per_voxel * region.x * region.y * region.z;
    if(block_idx * (int)_->desc.block_length == block_beg_pos
    && (end_pos - beg_pos - Int3(_->desc.padding)) == Int3(_->desc.block_length)){
        _->vol_reader->ReadBlockData({block_idx.x, block_idx.y, block_idx.y}, ptr, bytes);
    }
    else{
        LOG_DEBUG("NOTE: ReadVolumeRegion is not block!!!");
        _->vol_reader->ReadVolumeData(beg_pos.x, beg_pos.y, beg_pos.z, end_pos.x, end_pos.y, end_pos.z,
                                      ptr, bytes);
    }
}

void EBVolumeFile::WriteVolumeRegion(const Int3 &beg_pos, const Int3 &end_pos, const void *ptr) {
    NOT_IMPL
}



VISER_END

