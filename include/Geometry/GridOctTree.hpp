#pragma once
#include <Common/Common.hpp>


VISER_BEGIN

// 1.与SWC进行相交，快速判断显存是否足够这一条SWC体素化，粗略估计，可以取保守的上界
// 2.与视锥体进行相交
// 3. swc segment与aabb包围盒求交，把前者的空间范围视为两个球+圆台的外切四棱锥

// special grid octree, no primitives or just is the node's aabb, build from bottom to up
class GridOctTreePrivate;
class GridOctTree : public UnifiedRescBase{
public:
    struct OctTreeCreateInfo{
        Ref<HostMemMgr> host_mem_mgr_ref;

        Float3 leaf_node_shape;
        Float3 world_origin;
        Float3 world_range;
        Float3 expand_boundary;
        bool leaf_is_valid = true;
    };
    struct NodeIndex{
        int x;
        int y;
        int z;
        int level;


        bool operator==(const NodeIndex& other) const {
            return x == other.x && y == other.y && z == other.z && level == other.level;
        }
    };

    explicit GridOctTree(const OctTreeCreateInfo& info);

    ~GridOctTree();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    // clear all leaf
    void Clear(bool empty);

    // set one leaf node empty/non-empty
    void Set(const NodeIndex&, bool empty);

    void Set(const std::vector<NodeIndex>&, bool empty);

    bool TestIntersect(const BoundingBox3D& aabb,
                       std::function<bool(const BoundingBox3D&)> accurateIntersectTest) const;


    bool TestIntersect(const BoundingBox3D& aabb) const;

    std::vector<NodeIndex> GetIntersectNodes(const BoundingBox3D& aabb) const;

    int GetLevels() const;

    bool IsValidNode(const NodeIndex&) const;

protected:
    std::unique_ptr<GridOctTreePrivate> _;
};


VISER_END