#pragma once

#include <IO/SWCIO.hpp>

VISER_BEGIN

//每个节点都有唯一的编号，为了支持随机的删除和插入，
//考虑使用list+map来实现
class SWCPrivate;
class SWC{
public:

    using SWCPoint = SWCFile::SWCPoint;
    using SWCPointKey = int;

    // 减少点的密度，针对算法自动生成的点

    SWC();

    ~SWC();

    void InsertNodeLeaf(const SWCPoint& point) noexcept;

    void InsertNodeInternal(const SWCPoint& point, SWCPointKey kid) noexcept;

    bool QueryNode(SWCPointKey id) noexcept;

    SWCPoint& GetNode(SWCPointKey id) noexcept;

    inline bool IsRoot(const SWCPoint& point) noexcept{
        return point.pid == -1;
    }

    bool IsRoot(SWCPointKey id) noexcept;

    std::vector<SWCPointKey> GetNodeKids(SWCPointKey id) noexcept;

    void DeleteNode(SWCPointKey id, bool connect = false) noexcept;

    // id0 <-- id1
    void ConnectNode(SWCPointKey id0, SWCPointKey id1) noexcept;

    // 是否联通，即同一个祖先
    bool CheckConnection(SWCPointKey id0, SWCPointKey id1) noexcept;

    // 每次根据几何区域进行查询和删除前才更新重建BVH
    // 设置一个标志记录SWC是否被更改
    std::vector<SWCPoint> QueryNode(const BoundingBox3D& box) noexcept;

    void DeleteNode(const BoundingBox3D& box) noexcept;

    struct Iterator{
        using iter_t = std::unordered_map<SWCPointKey, SWCPoint>::iterator;
        iter_t it;
        Iterator(iter_t it):it(it){}
        SWCPoint& operator*(){
            return it->second;
        }
        //++it
        Iterator& operator++(){
            ++it;
            return *this;
        }
        //it++
        Iterator operator++(int){
            Iterator res(it);
            ++it;
            return res;
        }
    };

    Iterator begin();

    Iterator end();

    std::vector<SWCPoint> PackAll() noexcept;

protected:
    std::unique_ptr<SWCPrivate> _;
};

VISER_END

