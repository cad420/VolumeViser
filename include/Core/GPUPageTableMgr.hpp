#pragma once

#include <Common/Common.hpp>
#include <Model/Volume.hpp>

VISER_BEGIN

//只负责管理一个gpu上对应vtexture的使用情况
class GPUPageTableMgrPrivate;
class GPUPageTableMgr{
public:
    friend class GPUVTexMgr;

    ~GPUPageTableMgr();

    void Lock();

    void UnLock();

    using Key = GridVolume::BlockUID;

    struct TexCoord{
        uint32_t tid;
        uint32_t sx, sy, sz;
    };

    using Value = TexCoord;

    using PageTableItem = std::pair<Key, Value>;

    //只返回不在PageTable中的Key对应的Item
    void GetAndLock(const std::vector<Key>& keys, std::vector<PageTableItem>& items);

    //释放最近一次GetAndLock传入的keys
    void Release();

protected:
    GPUPageTableMgr();

    std::unique_ptr<GPUPageTableMgrPrivate> _;
};

VISER_END