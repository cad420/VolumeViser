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
//        uint32_t tid;
        uint32_t sx, sy, sz;
        uint16_t tid;
        uint16_t flag;

        bool Missed() const{

            return true;
        }
    };

    using Value = TexCoord;

    using PageTableItem = std::pair<Key, Value>;

    void GetAndLock(const std::vector<Key>& keys, std::vector<PageTableItem>& items);

    //释放最近一次GetAndLock传入的keys
    void Release();

    Ref<HashPageTable> GetPageTable();

protected:
    GPUPageTableMgr();

    std::unique_ptr<GPUPageTableMgrPrivate> _;
};

VISER_END