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
class GridVolume : public UnifiedRescBase{
public:
    struct BlockUID{
        //不超过65535
        //w‘ low 16bits 代表lod, high 16bits 代表flag
        uint32_t x = 0xffff, y = 0xffff, z = 0xffff, w = 0xff;

        CUB_CPU_GPU BlockUID() = default;

        BlockUID(UnifiedRescUID uid){
            x = uid & 0xffff;
            y = (uid >> 16) & 0xffff;
            z = (uid >> 32) & 0xffff;
            w = (uid >> 48) & 0xff;
        }

        BlockUID(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
        :x(x), y(y), z(z), w(w)
        {
            assert(x < (1u << 16) && y < (1u << 16) && z < (1u << 16) && w < (1u << 8));
        }

        bool operator==(const BlockUID& uid) const{
            return x == uid.x && y == uid.y && z == uid.z && w == uid.w;
        }

        UnifiedRescUID ToUnifiedRescUID() const{
            return ((size_t)x) | (((size_t)y) << 16) | (((size_t)z) << 32) |
                    (((size_t)(w & 0xff)) << 48) | (((size_t)UnifiedRescType::GridVolumeBlock) << 56);
        }

        //转换成xyzw后判断是否相等
        bool IsSame(UnifiedRescUID uid) const{
            BlockUID other(uid);
            return *this == other;
        }

        bool IsValid() const;

        bool IsBlack() const{
            return (w >> 8) & 2;
        }

        bool IsSparse() const{
            return (w >> 8) & 4;
        }
        bool IsSWC() const{
            return (w >> 8) & 8;
        }

        CUB_CPU_GPU int GetLOD() const{
            return static_cast<int>(w & 0xff);
        }
    };
    inline static BlockUID INVALID_BLOCK_UID = { 0xffffu,  0xffffu,  0xfffu, 0xffu};

    using GridVolumeDesc = VolumeIOInterface::VolumeDesc;

    struct GridVolumeCreateInfo{
        UnifiedRescUID host_mem_mgr_uid;//用于动态申请内存资源
        UnifiedRescUID gpu_mem_mgr_uid;
        uint32_t levels = 0;
        std::unordered_map<uint32_t, Handle<VolumeIOInterface>> lod_vol_file_io;
    };

    explicit GridVolume(const GridVolumeCreateInfo& info);

    ~GridVolume();

    // 整体的加锁，可选的
    void Lock() override;

    void UnLock() override;

    GridVolumeDesc GetDesc() const;

    UnifiedRescUID GetUID() const override;

    // 读取数据块，内部完成解压
    CUDAHostBuffer ReadBlock(const BlockUID& uid);

    CUDAHostBuffer ReadRegion(Int3 beg, Int3 end, uint32_t lod);

    // buffer的大小应该是正确的
    void ReadBlock(const BlockUID& uid, CUDAHostBuffer& buffer);

    void ReadRegion(Int3 beg, Int3 end, uint32_t lod, CUDAHostBuffer& buffer);

    // 隐藏cpu到gpu的传输过程，可能是内部解压后直接在gpu的，也可能是解压完传输到gpu的
    CUDAPitchedBuffer ReadBlockGPU(const BlockUID& uid);

    CUDAPitchedBuffer ReadRegionGPU(Int3 beg, Int3 end, uint32_t lod);

    void ReadBlockGPU(const BlockUID& uid, CUDAPitchedBuffer& buffer);

    void ReadRegion(Int3 beg, Int3 end, uint32_t lod, CUDAPitchedBuffer& buffer);

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