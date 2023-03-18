#pragma once

#include <Common/Common.hpp>
#include <Model/Volume.hpp>
#include "HostMemMgr.hpp"
VISER_BEGIN

//只负责管理一个gpu上对应vtexture的使用情况
class GPUPageTableMgrPrivate;
class GPUPageTableMgr : public UnifiedRescBase{
public:
    struct GPUPageTableMgrCreateInfo{
        mutable Ref<GPUMemMgr> gpu_mem_mgr;
        mutable Ref<HostMemMgr> host_mem_mgr;
        int vtex_count;
        Int3 vtex_block_dim;
    };

    GPUPageTableMgr(const GPUPageTableMgrCreateInfo& info);

    ~GPUPageTableMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    using Key = GridVolume::BlockUID;
    inline static Key INVALID_KEY = GridVolume::INVALID_BLOCK_UID;

// also used in gpu kernel
#define TexCoordFlag_IsValid  0x1u //代表数据是否存在显存
#define TexCoordFlag_IsBlack  VolumeBlock_IsBlack //代表数据块本身是否全为0
#define TexCoordFlag_IsSparse VolumeBlock_IsSparse //代表数据块本身是否为稀疏的
#define TexCoordFlag_IsSWC    VolumeBlock_IsSWC //代表数据块本身是否为SWC体素化使用
#define TexCoordFlag_IsSWCV   16u // 代表数据块在体素化算法过程中被更新了

    struct TexCoord{
        uint32_t sx, sy, sz;//一块纹理内部的偏移坐标
        uint16_t tid;//纹理索引
        uint16_t flag;//对于对应的key来说，是否是有效的，即是否原来存在于页表中

        bool Missed() const {
            return (flag & TexCoordFlag_IsValid) == 0;
        }
    };

    using Value = TexCoord;
    inline static Value INVALID_VALUE = {0, 0, 0, 0, 0};

    using PageTableItem = std::pair<Key, Value>;

    /**
     * @note 如果有些key对应的页表项被占用了，会等待其释放，因为被占用的绘制耗时相对于从磁盘重新导入加载的耗时是更小的
     * @param keys 所有要用到的索引
     * @param items 所有索引对应的查询结果，包括原来就在的页表项，以及原本不在但是通过调度得到的结果
     */
    void GetAndLock(const std::vector<Key>& keys, std::vector<PageTableItem>& items);

    //释放最近一次GetAndLock传入的keys

    CUB_CPU_GPU void Release(const std::vector<Key>& keys, bool readonly  = true);

    void Discard(const std::vector<Key>& keys);

    //todo 清除所有页表项，否则哈希页表会爆满
    void Reset();

    void ClearWithOption(std::function<bool(const Key& key)>);

    HashPageTable& GetPageTable(bool update = true);

    void Promote(const Key& key);

    //是否是一对有效的键值
    bool Check(const Key& key, const Value& value);

protected:
    friend class GPUVTexMgr;

    std::unique_ptr<GPUPageTableMgrPrivate> _;
};

VISER_END