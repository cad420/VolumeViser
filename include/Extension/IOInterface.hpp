#pragma once

#include <Common/Common.hpp>

VISER_BEGIN

class VolumeIOInterface : public UnifiedRescBase{
public:
    virtual ~VolumeIOInterface() = default;

    struct VolumeDesc{
        std::string volume_name;
        int bits_per_sample = 8;
        int samples_per_voxel = 1;
        bool is_float = false;
        UInt3 shape;
        uint32_t block_length = 0;
        uint32_t padding = 0;
        UInt3 blocked_dim;
        Float3 voxel_space;
        //preserved and not used
        bool decoding_cpu_only = false;
    };

    virtual VolumeDesc GetVolumeDesc() = 0;

    /**
     * @param beg_pos 读取的起始位置，可以在体数据本身范围之外
     * @param end_pos 读取的结束位置，必须要大于起始位置，也可以在体数据本身范围之外
     * @param ptr 可以是host ptr也可以是device ptr，取决于具体实现类，不能是nullptr
     */
    virtual void ReadVolumeRegion(const Int3& beg_pos, const Int3& end_pos, void* ptr) = 0;

    virtual void WriteVolumeRegion(const Int3& beg_pos, const Int3& end_pos, const void* ptr) = 0;

    static UnifiedRescUID GenUnifiedRescUID(){
        std::atomic<size_t> g_uid = 0;
        auto uid = g_uid.fetch_add(1);
        return ::viser::GenUnifiedRescUID(uid, UnifiedRescType::VolumeIO);
    }
};

class SWCIOInterface : public UnifiedRescBase{
public:

};

class MeshIOInterface : public UnifiedRescBase{
public:

};


VISER_END

