#include <Model/Volume.hpp>

VISER_BEGIN

class GridVolumePrivate{
public:

};


    GridVolume::GridVolume(const GridVolumeDesc& desc){

    }

    GridVolume::~GridVolume() {

    }

    GridVolume::GridVolumeDesc GridVolume::GetDesc() const {
        return viser::GridVolume::GridVolumeDesc();
    }

    GridVolume::UID GridVolume::GetUID() const {
        return 0;
    }

    CUDAHostBuffer GridVolume::ReadBlock(const GridVolume::BlockUID &uid) {
        return viser::CUDAHostBuffer(0, cub::e_cu_device, cub::cu_context());
    }

    CUDAHostBuffer GridVolume::ReadEncodedBlock(const GridVolume::BlockUID &uid) {
        return viser::CUDAHostBuffer(0, cub::e_cu_device, cub::cu_context());
    }

    CUDAHostBuffer GridVolume::ReadRegion(Int3 beg, Int3 end) {
        return viser::CUDAHostBuffer(0, cub::e_cu_device, cub::cu_context());
    }

    void GridVolume::Lock() {

    }

    void GridVolume::UnLock() {

    }

    void GridVolume::ReadBlock(const GridVolume::BlockUID &uid, CUDAHostBuffer &buffer) {

    }


VISER_END