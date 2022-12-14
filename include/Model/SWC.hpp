#pragma once

#include <IO/SWCIO.hpp>

VISER_BEGIN

//每个节点都有唯一的编号，为了支持随机的删除和插入，
//考虑使用unordered_map来实现

// 对一个点的更改操作只能是以下三种
// 1. 添加一个新点 2. 删除一个旧店 3.改变一个现有点的半径
// 注意 无法更改一个现有点的位置和连接关系 如果要处理这种情况 那么直接删除再添加

// SWC局部更新
// 每次对SWC进行更改操作后 内部都会记录那些改动过的点 一直到显式调用commit
// commit会清除所有当前记录的改动过的点


// 全局SWC体素化
// 根据所有的SWC点建立一个BVH树，每个BVH Node进行体素化后再进行marching cube算法，适用于无错误的整个swc文件生成mesh。
// 这个算法不支持局部SWC更新，因此更新后的BVH树会发生变化，Node发生变化的话无法与之前生成的mesh对应

// 支持局部快速更新的SWC体素化
// 目标 每次只重新生成更新部分的mesh 没有变化部分的mesh不会被影响
// 也就是说 每个mesh需要有固定的编号对应 而且是与SWC本身数据无关的
// SWC数据影响的是体素化过程会写入到哪些block
// 使用BlockUID，包括了xyz索引和lod，可以转换到UnifiedRescUID(size_t)
// 那么需要确定每个swc point与block uid的对应关系
// 一个swc point影响到多个block
// 一个block对应唯一生成的mesh
// 每次commit前，获取所有直接改动以及被间接影响的点，也就是所有体素化会有所改变的线段segment
// 根据所有改变线段的AABB包围盒得到所有可能受影响的block，并进行标记
// 对所有改变线段依次进行体素化，这样子不行，比如一个点的半径改变导致一个block受影响，
// 这个block的其它线段也应该重新进行体素化
// 因此，在标记那些受影响的block之后，再遍历所有的线段，判断每一条线段影响的blocks是否被标记，如果是那么重新体素化生成mesh
// 这一过程类似于对虚拟纹理进行写入操作，然后标记真正被改动过的block
// 对改动过的block依次进行marching cube算法，得到block所对应的mesh，并按照key-value进行存储
// 因此相对于的SWC渲染器，每个patch对应的是一个block所属的mesh
// 也就是说，一个swc对应生成的mesh以block单位分别存储，如果要进行光滑操作的话，可以先全局合并重复顶点，生成索引
// 因为光滑操作可以最后统一进行，因此选择使用CPU进行比较方便写，不需要实时光滑，因为光滑需要生成索引，因此和mesh化简应该是属于最后导出前的转换任务

// 虚拟纹理的写入 在创建cuda array的时候加入读写的flag 然后生成cuda surface对象 使用surf3Dread surf3Dwrite

// SWC线渲染
// 一个SWC对应多条神经元，每条神经元只有一个根节点，其是树型的，因此可以用一个root id代表一条神经元
// 一条神经元与一个patch id唯一对应
// 如果在中间插入或者任意的删除，都需要重新将整个数据上传到渲染器
// 如果是添加新的点，那么不需要重新上传

// SWC生成的Mesh
// 一个SWC最后会生成一个网格文件，保存到文件中的网格是经过合并、光滑、简化的
// 如果再次从文件中加载回Mesh，这时候的Mesh因为Merged而不是Blocked，是无法局部更新的
// 因此如果要编辑的话，会首先生成全部swc涉及到的Blocked Mesh然后再进行更改

class SWCPrivate;
class SWC : public UnifiedRescBase{
public:

    using SWCPoint = SWCFile::SWCPoint;
    using SWCPointKey = int;

    static constexpr SWCPointKey INVALID_SWC_KEY = 0;

    // 减少点的密度，针对算法自动生成的点

    SWC();

    ~SWC();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    enum Ops : int{
        New_Add = 1,
        Old_Del = 2,
        Old_UpdateR = 3,
        Old_ConnectSeg = 4
    };


    //获取所有直接被改动的点集合
    std::vector<std::pair<SWCPoint, Ops>> GetAllModifiedSWCPts() noexcept;

    //获取所有直接改动加间接影响的点集合
    std::vector<SWCPoint> GetAllModifiedAndInfluencedPts() noexcept;

    /**
     * @brief 将所有从上一次Commit之后记录下来的被更改的点清除
     * @note 被更改只包括三种操作 1.添加新的点 2.删除一个点 3.更改一个点的半径
     */
    void Commit() noexcept;

    void PrintInfo() noexcept;

    void InsertNodeLeaf(const SWCPoint& point) noexcept;

    void InsertNodeInternal(const SWCPoint& point, SWCPointKey kid) noexcept;

    bool QueryNode(SWCPointKey id) noexcept;

    SWCPoint& GetNode(SWCPointKey id) noexcept;

    inline bool IsRoot(const SWCPoint& point) noexcept{
        return point.pid == -1;
    }

    SWCPointKey GetNodeRoot(SWCPointKey id) noexcept;

    std::vector<SWCPointKey> GetAllRootIDs() noexcept;

    bool IsRoot(SWCPointKey id) noexcept;

    // if a is root of b
    bool IsRoot(SWCPointKey a, SWCPointKey b) noexcept;

    SWCPointKey GetFirstCommonRoot(SWCPointKey a, SWCPointKey b) noexcept;

    int GetNodeToRootLength(SWCPointKey a) noexcept;

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
        bool operator!=(const Iterator& other) const {
            return it != other.it;
        }
        iter_t& operator->(){
            return it;
        }
        const iter_t& operator->() const {
            return it;
        }
    };

    Iterator begin();

    Iterator end();

    //打包所有点
    std::vector<SWCPoint> PackAll() noexcept;

    //打包所有不同的连通集
    std::vector<std::vector<SWCPoint>> PackLines() noexcept;

protected:
    std::unique_ptr<SWCPrivate> _;
};

VISER_END