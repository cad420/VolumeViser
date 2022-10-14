#include <Model/SWC.hpp>
#include <unordered_set>
VISER_BEGIN

class SWCPrivate{
public:
    using SWCPoint = SWC::SWCPoint;
    using SWCPointKey = SWC::SWCPointKey;
    std::unordered_map<SWCPointKey , SWCPoint> swc_point_mp;
    struct Node{
        SWCPointKey id;
        SWCPointKey parent = -1;
        std::unordered_set<SWCPointKey> children;
    };
    std::unordered_map<SWCPointKey, Node> node_mp;
    bool updated = false;

    SWCPointKey GetRoot(SWCPointKey id) noexcept{
        assert(id > 0);
        while(true){
            auto& node = node_mp[id];
            if(node.parent == -1)
                return node.id;
            id = node.parent;
        }
        return 0;
    }
};

    SWC::SWC() {

    }

    SWC::~SWC() {

    }

    bool SWC::QueryNode(SWC::SWCPointKey id) noexcept {
        return _->swc_point_mp.count(id);
    }

    SWC::SWCPoint& SWC::GetNode(SWCPointKey id) noexcept {
        static SWCPoint INVALID{0};

        if(_->swc_point_mp.count(id) != 0){
            return _->swc_point_mp[id];
        }

        return INVALID;
    }

    std::vector<SWC::SWCPointKey> SWC::GetNodeKids(SWC::SWCPointKey id) noexcept {
        std::vector<SWC::SWCPointKey> res;
        if(_->node_mp.count(id)){
            for(auto& c : _->node_mp[id].children)
                res.push_back(c);
        }
        return res;
    }

    void SWC::InsertNodeLeaf(const SWC::SWCPoint& point) noexcept {
        assert(point.id > 0);
        assert(!QueryNode(point.id));
        _->updated = true;
        _->swc_point_mp[point.id] = point;
        auto& p_node = _->node_mp[point.pid];
        p_node.children.insert(point.id);
        _->node_mp[point.id] = {point.id, point.pid};
    }

    void SWC::InsertNodeInternal(const SWCPoint& point, SWCPointKey kid) noexcept {
        auto id = point.id, pid = point.pid;
        assert(id > 0);
        assert(!QueryNode(id));
        _->updated = true;
        _->swc_point_mp[id] = point;
        assert(_->node_mp[pid].children.count(kid));
        _->node_mp[pid].children.erase(kid);
        _->node_mp[kid].parent = id;
        _->swc_point_mp[kid].pid = id;
        _->node_mp[id] = {id, pid, {kid}};
    }

    bool SWC::CheckConnection(SWC::SWCPointKey id0, SWC::SWCPointKey id1) noexcept {
        assert(id0 > 0 && id1 > 0);
        assert(QueryNode(id0) && QueryNode(id1));
        return _->GetRoot(id0) == _->GetRoot(id1);
    }

    void SWC::ConnectNode(SWC::SWCPointKey id0, SWC::SWCPointKey id1) noexcept {
        assert(id0 > 0 && id1 > 0);
        assert(QueryNode(id0) && QueryNode(id1));
        assert(_->node_mp[id1].parent == -1);
        // debug
        assert(!CheckConnection(id0, id1));
        auto& node0 = _->node_mp[id0];
        if(node0.parent == id1 || node0.children.count(id1) == 1){
            std::cerr << "Connect two node that already connected" << std::endl;
            return;
        }
        _->updated = true;
        _->swc_point_mp[id1].pid = id0;
        // id1 should be root
        _->node_mp[id1].parent = id0;
        _->node_mp[id0].children.insert(id1);
    }

    void SWC::DeleteNode(SWCPointKey id, bool connect) noexcept {
        if(!QueryNode(id)) return;
        _->updated = true;
        _->swc_point_mp.erase(id);
        auto d_node = _->node_mp[id];
        _->node_mp.erase(id);
        auto pid = d_node.parent;
        if(!connect){
            _->node_mp[pid].children.erase(id);
            for(auto c : d_node.children){
                _->swc_point_mp[c].pid = -1;
                _->node_mp[c].parent = -1;
            }
        }
        else{
            _->node_mp[pid].children.erase(id);
            for(auto c : d_node.children){
                _->swc_point_mp[c].pid = pid;
                _->node_mp[c].parent = pid;
                if(pid != -1){
                    _->node_mp[pid].children.insert(c);
                }
            }
        }
    }

    std::vector<SWC::SWCPoint> SWC::QueryNode(const BoundingBox3D &box) noexcept {
        if(_->updated){

        }

        return {};
    }

    void SWC::DeleteNode(const BoundingBox3D &box) noexcept {
        if(_->updated){

        }


    }

    bool SWC::IsRoot(SWC::SWCPointKey id) noexcept {
        return _->node_mp[id].parent == -1;
    }

    SWC::Iterator SWC::begin() {
        return SWC::Iterator(_->swc_point_mp.begin());
    }

    SWC::Iterator SWC::end() {
        return SWC::Iterator(_->swc_point_mp.end());
    }


VISER_END

