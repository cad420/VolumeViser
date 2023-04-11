#include <Geometry/GridOctTree.hpp>

#include <unordered_set>

namespace std{
template<>
struct hash<viser::GridOctTree::NodeIndex>{
    size_t operator()(const viser::GridOctTree::NodeIndex& index) const{
        return vutil::hash(index.x, index.y, index.z, index.level);
    }
};
}

VISER_BEGIN

//根据视锥体的包围盒快速筛选可能相交的数据块lod0节点，然后可以并行求出哪些是相交的

namespace{
    using NodeIndex = GridOctTree::NodeIndex;
    struct OctPrimitive{
        NodeIndex index;
        int is_valid;
    };

    struct OctBuildNode{
        size_t first_prim_offset;
        int is_leaf;
        OctBuildNode* kids[8] = {nullptr};
        BoundingBox3D aabb;
#ifdef VISER_DEBUG
        NodeIndex index;
#endif
    };

    struct alignas(64) LinearOctNode{
        BoundingBox3D aabb;
        int kids_offset[8] = {-1};
        int is_leaf;
        int prim_offset;
    };



}

class GridOctTreePrivate{
  public:
    LinearOctNode* linear_nodes = nullptr;
    std::vector<OctPrimitive> primitives;
    int levels = 0;

    std::mutex mtx;

    UnifiedRescUID uid;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::OctTree);
    }
};

GridOctTree::GridOctTree(const OctTreeCreateInfo &info)
{
    _ = std::make_unique<GridOctTreePrivate>();

    _->uid = _->GenRescUID();

    // build
    Int3 leaf_nodes_dim(
        std::ceil(info.world_range.x / info.leaf_node_shape.x),
        std::ceil(info.world_range.y / info.leaf_node_shape.y),
        std::ceil(info.world_range.z / info.leaf_node_shape.z)
    );

    std::vector<OctPrimitive> primitives;
    std::vector<OctBuildNode> leaves;

    auto gen_aabb = [&](float x, float y, float z, const Float3& shape, const Float3 boundary = Float3(0)){
        return BoundingBox3D(
            Float3(x, y, z) * shape - boundary,
            Float3(x + 1.f, y + 1.f, z + 1.f) * shape + boundary
        ) + info.world_origin;
    };

    Float3 node_shape = info.leaf_node_shape;

    size_t total_nodes = 0;
    bool leaf_is_valid = info.leaf_is_valid;
    for(int z = 0; z < leaf_nodes_dim.z; ++z){
        for(int y = 0; y < leaf_nodes_dim.y; ++y){
            for(int x = 0; x < leaf_nodes_dim.x; ++x){
                //todo: use space curve
                primitives.push_back(OctPrimitive{.index = {x, y, z, 0}, .is_valid = leaf_is_valid});
                leaves.push_back(OctBuildNode{.first_prim_offset = total_nodes, .is_leaf = true,
                                              .aabb = gen_aabb(x, y, z, node_shape, info.expand_boundary)});
                total_nodes++;
            }
        }
    }
    std::map<int, std::vector<OctBuildNode>> level_nodes;
    level_nodes[0] = std::move(leaves);
    Int3 last_level_nodes_dim = leaf_nodes_dim;
    int last_level = 0;
    Float3 cur_node_shape = node_shape;
    Float3 cur_ext_bound = info.expand_boundary;
    LOG_INFO("level: {}, dim: {} {} {}", 0, leaf_nodes_dim.x, leaf_nodes_dim.y, leaf_nodes_dim.z);
    while(true){
        auto current_level_nodes_dim = (last_level_nodes_dim + 1) / 2;
        auto current_level = last_level + 1;
        LOG_INFO("level: {}, dim: {} {} {}", current_level, current_level_nodes_dim.x, current_level_nodes_dim.y, current_level_nodes_dim.z);
        bool build_finished = false;
        if(current_level_nodes_dim == Int3(1))
            build_finished = true;

        auto get_kid = [&](int x, int y, int z)->OctBuildNode*{
            if(x >= last_level_nodes_dim.x || y >= last_level_nodes_dim.y || z >= last_level_nodes_dim.z) return nullptr;
            size_t index = x + y * last_level_nodes_dim.x + z * last_level_nodes_dim.x * last_level_nodes_dim.y;
            return level_nodes.at(last_level).data() + index;// ok
        };
        cur_node_shape *= 2.f;
        cur_ext_bound *= 2.f;

        std::vector<OctBuildNode> current_level_nodes;
        for(int z = 0; z < current_level_nodes_dim.z; ++z){
            for(int y = 0; y < current_level_nodes_dim.y; ++y){
                for(int x = 0; x < current_level_nodes_dim.x; ++x){
                    auto& node = current_level_nodes.emplace_back();
                    total_nodes++;
                    node.aabb = gen_aabb(x, y, z, cur_node_shape, cur_ext_bound);
                    node.is_leaf = false;
#ifdef VISER_DEBUG
                    node.index = {x, y, z, current_level};
#endif
                    node.kids[0] = get_kid(x * 2, y * 2, z * 2);
                    node.kids[1] = get_kid(x * 2 + 1, y * 2, z * 2);
                    node.kids[2] = get_kid(x * 2, y * 2 + 1, z * 2);
                    node.kids[3] = get_kid(x * 2 + 1, y * 2 + 1, z * 2);
                    node.kids[4] = get_kid(x * 2, y * 2, z * 2 + 1);
                    node.kids[5] = get_kid(x * 2 + 1, y * 2, z * 2 + 1);
                    node.kids[6] = get_kid(x * 2, y * 2 + 1, z * 2 + 1);
                    node.kids[7] = get_kid(x * 2 + 1, y * 2 + 1, z * 2 + 1);
                }
            }
        }
        level_nodes[current_level] = std::move(current_level_nodes);

        last_level_nodes_dim = current_level_nodes_dim;
        last_level = current_level;

        if(build_finished) break;


    }
    _->levels = last_level;
    assert(level_nodes.at(last_level).size() == 1);
    auto root = level_nodes.at(last_level).data();

    int count = 0;
    std::function<size_t(OctBuildNode* node, size_t&)> dfs = [&](OctBuildNode* node, size_t& offset){
//        LOG_INFO("count: {}, node: {} {} {} {}", count++, node->index.x, node->index.y, node->index.z, node->index.level);
        LinearOctNode* linear_node = _->linear_nodes + offset;
        size_t node_offset = offset++;
        linear_node->aabb = node->aabb;
        if(node->is_leaf){
            linear_node->prim_offset = node->first_prim_offset;
            linear_node->is_leaf = true;
        }
        else{
            linear_node->is_leaf = false;
            for(int i = 0; i < 8; i++)
                if(node->kids[i]) linear_node->kids_offset[i] = dfs(node->kids[i], offset);
                else linear_node->kids_offset[i] = -1;
        }
        return node_offset;
    };

    _->linear_nodes = vutil::aligned_alloc<LinearOctNode>(total_nodes);
    _->primitives = std::move(primitives);
    size_t offset = 0;
    dfs(root, offset);
    assert(offset == total_nodes);
}

GridOctTree::~GridOctTree()
{

}

void GridOctTree::Lock()
{
    _->mtx.lock();
}

void GridOctTree::UnLock()
{
    _->mtx.unlock();
}

UnifiedRescUID GridOctTree::GetUID() const
{
    return _->uid;
}

void GridOctTree::Clear(bool empty)
{
    for(auto& prim : _->primitives)
        prim.is_valid = false;
}

void GridOctTree::Set(const NodeIndex & node, bool empty)
{
    Set({node}, empty);
}

void GridOctTree::Set(const std::vector<NodeIndex> & nodes, bool empty)
{
    std::unordered_set<NodeIndex> indices;
    for(auto& node : nodes) indices.insert(node);
    for(auto& prim : _->primitives){
        if(indices.count(prim.index) != 0){
            prim.is_valid = !empty;
        }
        else{
            prim.is_valid = empty;
        }
    }
}


bool GridOctTree::TestIntersect(const BoundingBox3D &aabb) const
{
    if(!_->linear_nodes) return false;

    std::vector<size_t> st; st.reserve(64);
    st.push_back(0);
    while(!st.empty()){
        auto node_index = st.back();
        st.pop_back();
        const auto node = _->linear_nodes + node_index;
        if(!node->aabb.intersect(aabb))
            continue;

        if(node->is_leaf){
            auto ret = _->primitives.at(node->prim_offset).is_valid;
            if(ret) return true;
        }
        else{
            for(int i = 0; i < 8; i++){
                auto offset = node->kids_offset[i];
                if(offset >= 0)
                    st.push_back(node->kids_offset[i]);
            }
        }
    }

    return false;
}

std::vector<GridOctTree::NodeIndex> GridOctTree::GetIntersectNodes(const BoundingBox3D& aabb) const
{
    std::vector<NodeIndex> ret;
    if(!_->linear_nodes) return ret;

    std::vector<size_t> st;
    st.push_back(0);
    while(!st.empty()){
        auto node_index = st.back();
        st.pop_back();
        const auto node = _->linear_nodes + node_index;
        if(!node->aabb.intersect(aabb))
            continue;

        if(node->is_leaf){
            auto& prim = _->primitives.at(node->prim_offset);
            if(prim.is_valid) ret.push_back(prim.index);
        }
        else{
            for(int i = 0; i < 8; i++)
                st.push_back(node->kids_offset[i]);
        }
    }
    return ret;
}

bool GridOctTree::TestIntersect(const BoundingBox3D &aabb,
                                std::function<bool(const BoundingBox3D &)> accurateIntersectTest) const
{

    return false;
}

int GridOctTree::GetLevels() const
{
    return _->levels;
}

bool GridOctTree::IsValidNode(const NodeIndex & index) const
{
    for(auto& prim : _->primitives){
        if(prim.index == index) return prim.is_valid;
    }
}


VISER_END


