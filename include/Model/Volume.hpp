#pragma once
#include <memory>
#include <IO/VolumeIO.hpp>

// 体数据使用一个统一的对象模型表示
// 一个grid volume可以看作是多个block组成，当然可以只由一个block组成
// 有两种类型创建的GridVolume，一种是直接在程序中创建，不与具体的文件关联，可以用于线转体
// 另一种是由IO接口创建，与具体的文件关联

// 因为解压操作可能位于CPU或者GPU，因此GridVolume对于数据块的缓存是可选的
// 即为减少磁盘IO的耗时而建立的cache

VISER_BEGIN

class GridVolumePrivate;
class GridVolume{
public:
    struct BlockUID{
        uint32_t x, y, z, w;
    };

    using GridVolumeDesc = VolumeFile::VolumeDesc;


    explicit GridVolume(const GridVolumeDesc& desc);

    ~GridVolume();

    GridVolumeDesc GetDesc() const;

    using UID = uint16_t;

    UID GetUID() const;

    CUDAHostBuffer ReadBlock(BlockUID);

    CUDAHostBuffer ReadEncodedBlock(BlockUID);

    CUDAHostBuffer ReadRegion();

    //与文件关联的会返回nullptr
    std::unique_ptr<GridVolume> GetSubGridVolume();

protected:
    std::unique_ptr<GridVolumePrivate> _;
};



VISER_END