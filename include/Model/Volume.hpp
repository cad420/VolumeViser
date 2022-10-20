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

//为了充分使用CPU资源，因此考虑数据块的解压在CPU进行，而且CPU支持更广泛的数据格式
//目前GPU的解压只支持10bit的yuv，对于16位的数据来说，可能略有不足，而对于CPU则有12bit的支持，甚至14bit
//使用CUDA host内存，可以利用DMA优势，加快数据从CPU到GPU的传输速率
class GridVolumePrivate;
class GridVolume{
public:
    struct BlockUID{
        uint32_t x, y, z, w;

        BlockUID() = default;

        BlockUID(UnifiedRescUID uid){

        }

        BlockUID(uint32_t x, uint32_t y, uint32_t z, uint32_t w){

        }

        bool operator==(const BlockUID& uid) const{
            return x == uid.x && y == uid.y && z == uid.z && w == uid.w;
        }

        //生成唯一的RescUID
        UnifiedRescUID ToUnifiedRescUID() const{
            return 0;
        }

        //类型不同，但是代表的block是一样的
        bool IsSame(UnifiedRescUID uid) const{

            return true;
        }

        bool IsValid() const{

        }

        int GetLOD() const{

        }
    };


    using GridVolumeDesc = VolumeFile::VolumeDesc;

    // 创建的时候应该要指定所属于的资源
    struct GridVolumeCreateInfo{
        Ref<GPUMemMgr> gpu_mem_mgr;
        Ref<HostMemMgr> host_mem_mgr;
        // other params...
        GridVolumeDesc volume_desc;
    };

    explicit GridVolume(const GridVolumeCreateInfo& info);
    explicit GridVolume(const GridVolumeDesc& desc);

    ~GridVolume();

    // 整体的加锁
    void Lock();

    void UnLock();

    GridVolumeDesc GetDesc() const;

    using UID = uint16_t;

    UID GetUID() const;

    //读取数据块，内部完成解压
    CUDAHostBuffer ReadBlock(const BlockUID& uid);

    CUDAHostBuffer ReadEncodedBlock(const BlockUID& uid);

    CUDAHostBuffer ReadRegion(Int3 beg, Int3 end);

    void ReadBlock(const BlockUID& uid, CUDAHostBuffer& buffer);

    //与文件关联的会返回nullptr
    std::unique_ptr<GridVolume> GetSubGridVolume();

protected:
    std::unique_ptr<GridVolumePrivate> _;
};



VISER_END

namespace std{

    template<>
    struct hash<viser::GridVolume::BlockUID>{
      size_t operator()(const viser::GridVolume::BlockUID& uid) const {
          return vutil::hash(uid.x, uid.y, uid.z, uid.w);
      }
    };

}