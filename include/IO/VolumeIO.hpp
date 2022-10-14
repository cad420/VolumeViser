#pragma once
#include <Extension/IOInterface.hpp>

VISER_BEGIN

//不考虑写更新或生成
class VolumeFilePrivate;
class VolumeFile : public VolumeIOInterface{
public:
    VolumeFile(std::string_view filename);

    ~VolumeFile();

    struct VolumeDesc{
        std::string volume_name;
        std::string data_path;
        int bits_per_sample = 8;
        int samples_per_voxel = 1;
        bool is_float = false;
        UInt3 shape;
        uint32_t block_length;
        UInt3 blocked_dim;
        uint32_t padding = 1;
        bool decoding_cpu_only = false;
    };

    void ReadVolumeRegion(Int3 beg_pos, Int3 end_pos, void* ptr, size_t size) ;

    void WriteVolumeRegion() ;

    VolumeDesc GetVolumeDesc();



protected:
    std::unique_ptr<VolumeFilePrivate> _;
};

VISER_END
