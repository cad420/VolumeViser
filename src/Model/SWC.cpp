#include <Model/SWC.hpp>
#include <unordered_set>
#include <stack>
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

    std::unordered_map<SWC::Ops, std::vector<SWCPoint>> direct_influenced_pts;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::SWC);
    }
    std::mutex mtx;
};

    SWC::SWC() {
        _ = std::make_unique<SWCPrivate>();

        _->uid = _->GenRescUID();
    }

    SWC::~SWC() {

    }

    void SWC::Lock() {
        _->mtx.lock();
    }

    void SWC::UnLock() {
        _->mtx.unlock();
    }

    UnifiedRescUID SWC::GetUID() const {
        return _->uid;
    }

    void SWC::PrintInfo() noexcept{
        int line_count = 0;
        for(auto& [_, node] : _->node_mp){
            if(node.parent > 0){
                line_count += 1;
            }
        }
        std::cout << "[SWC Infos] : "
                  << "\n\tTotal SWC Point Count : " << _->swc_point_mp.size()
                  << "\n\tTotal SWC Line Count : " << line_count
                  << std::endl;
    }

    bool SWC::QueryNode(SWC::SWCPointKey id) noexcept {
        if(id == -1) return false;
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
        _->direct_influenced_pts[New_Add].emplace_back(point);
        _->updated = true;
        _->swc_point_mp[point.id] = point;
        if(point.pid <= 0) return;
        auto& p_node = _->node_mp[point.pid];
        p_node.children.insert(point.id);
        _->node_mp[point.id] = {point.id, point.pid};


    }

    void SWC::InsertNodeInternal(const SWCPoint& point, SWCPointKey kid) noexcept {
        auto id = point.id, pid = point.pid;
        assert(id > 0);
        assert(!QueryNode(id));
        _->updated = true;
        _->direct_influenced_pts[New_Add].emplace_back(point);
        _->swc_point_mp[id] = point;
        assert(_->node_mp[pid].children.count(kid));
        _->node_mp[pid].children.erase(kid);
        _->node_mp[pid].children.insert(id);
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

        _->direct_influenced_pts[Old_ConnectSeg].emplace_back(_->swc_point_mp.at(id1));
    }

    void SWC::DeleteNode(SWCPointKey id, bool connect) noexcept {
        if(!QueryNode(id)) return;
        _->direct_influenced_pts[Old_Del].emplace_back(_->swc_point_mp.at(id));
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

    std::vector<SWC::SWCPoint> SWC::PackAll() noexcept{
        std::vector<SWC::SWCPoint> ret;
        ret.reserve(_->swc_point_mp.size());
        for(auto& [_, pt] : _->swc_point_mp){
            ret.emplace_back(pt);
        }
        return ret;
    }

    std::vector<std::vector<SWC::SWCPoint>> SWC::PackLines() noexcept{
        auto roots = GetAllRootIDs();
        LOG_DEBUG("SWC PackLines which has root count : {}", roots.size());
        std::vector<std::vector<SWC::SWCPoint>> ret;
        for(auto root : roots){
            std::stack<SWCPointKey> st;
            st.push(root);
            auto& line = ret.emplace_back();
            while(!st.empty()){
                auto pt_key = st.top();
                st.pop();
                auto node = GetNode(pt_key);
                if(node.pid != -1){
                    line.push_back(GetNode(node.pid));
                    line.push_back(node);
                }
                auto kids = GetNodeKids(pt_key);
                for(auto it = kids.cbegin(); it != kids.cend(); it++){
                    st.push(*it);
                }
            }
            LOG_DEBUG("SWC line has segment count : {}", line.size() / 2);
        }
        return ret;
    }

    std::vector<SWC::SWCPointKey> SWC::GetAllRootIDs() noexcept {
        std::vector<SWCPointKey> roots;

        for(auto& [_, pt] : _->swc_point_mp){
            if(pt.pid == -1){
                roots.emplace_back(_);
            }
        }

        return roots;
    }

    SWC::SWCPointKey SWC::GetNodeRoot(SWC::SWCPointKey id) noexcept {
        auto pid = _->swc_point_mp.at(id).pid;
        if(pid == -1)
            return id;
        return GetNodeRoot(pid);
    }

    std::vector<std::pair<SWC::SWCPoint, SWC::Ops>> SWC::GetAllModifiedSWCPts() noexcept {
        std::vector<std::pair<SWCPoint,Ops>> ret;
        for(auto& [op, pts] : _->direct_influenced_pts){
            for(auto& pt : pts){
                ret.emplace_back(pt, op);
            }
        }
        return ret;
    }

    std::vector<SWC::SWCPoint> SWC::GetAllModifiedAndInfluencedPts() noexcept {
        std::unordered_set<SWCPoint> st;
        for(auto& [op, pts] : _->direct_influenced_pts){
            if(op == Old_ConnectSeg){
                //两个点相连必定有一个是根节点，所以记录的必须是根节点
                for(auto& pt : pts){
                    assert(pt.pid == -1);
                    st.insert(pt);
                    if(QueryNode(pt.id) && QueryNode(pt.pid)){
                        st.insert(GetNode(pt.pid));
                    }
                }
            }
            else{
//                if(op == Old_UpdateR){
//                    for(auto& pt : pts){
//                        //更新半径前后的点都会影响最后的范围
//                        st.insert(pt);
//                        if(QueryNode(pt.id)){
//                            st.insert(GetNode(pt.id));
//                        }
//                    }
//                    //半径还会影响相邻的点
//                }
                for(auto& pt : pts){
                    st.insert(pt);
                    if(QueryNode(pt.id)){
                        if(QueryNode(pt.pid))
                            st.insert(GetNode(pt.pid));
                        for(auto kid : GetNodeKids(pt.id)){
                            st.insert(GetNode(kid));
                        }
                    }
                }
            }
        }
        std::vector<SWCPoint> ret;
        ret.reserve(st.size());
        for(auto& pt : st){
            ret.emplace_back(pt);
        }
        return ret;
    }

    void SWC::Commit() noexcept {
        _->direct_influenced_pts.clear();
    }
    bool SWC::IsRoot(SWC::SWCPointKey a, SWC::SWCPointKey b) noexcept{
        if(b == -1) return false;
        if(a == b) return true;
        return IsRoot(a, _->swc_point_mp.at(b).pid);
    }

    SWC::SWCPointKey SWC::GetFirstCommonRoot(SWCPointKey a, SWCPointKey b) noexcept{
        assert(CheckConnection(a, b));
        int len_a = GetNodeToRootLength(a);
        int len_b = GetNodeToRootLength(b);
        if(len_a < len_b){
            std::swap(len_a, len_b);
            std::swap(a, b);
        }
        int d = len_a - len_b;
        while(d-- > 0) a = _->swc_point_mp.at(a).pid;
        for(int i = 0; i < len_b; i++){
            if(a == b) return a;
            a = _->swc_point_mp.at(a).pid;
            b = _->swc_point_mp.at(b).pid;
        }
        LOG_ERROR("GetFirstCommonRoot : a and b is not connected");
        return -1;
    }
    int SWC::GetNodeToRootLength(SWC::SWCPointKey a) noexcept {
        if(a == -1) return 0;
        return GetNodeToRootLength(_->swc_point_mp.at(a).pid) + 1;
    }
    void SWC::UpdateRadius(SWCPointKey id, float r) noexcept
    {
        if(!QueryNode(id)) return;
        _->swc_point_mp.at(id).radius = r;
        _->direct_influenced_pts[Old_UpdateR].push_back(_->swc_point_mp.at(id));
    }
    bool SWC::CheckUniquePath(SWC::SWCPointKey id0, SWC::SWCPointKey id1) noexcept
    {
        if(!QueryNode(id0) || !QueryNode(id1)) return false;
        if(!IsRoot(id0, id1) && !IsRoot(id1, id0)) return false;
        if(IsRoot(id1, id0)) std::swap(id0, id1);

        auto id = _->swc_point_mp.at(id1).pid;
        while(id != id0){
            if(_->node_mp.at(id).children.size() != 1) return false;
            id = _->swc_point_mp.at(id).pid;
        }

        return true;
    }
    void SWC::DeleteUniquePath(SWC::SWCPointKey id0, SWC::SWCPointKey id1) noexcept
    {
        if(!CheckUniquePath(id0, id1) && !CheckUniquePath(id1, id0)) return;
        if(IsRoot(id1, id0)) std::swap(id0, id1);
        auto id = _->swc_point_mp.at(id1).pid;
        std::vector<SWCPointKey> ids;
        SWCPointKey lst_key = -1;
        _->updated = true;
        while(id != id0){
            ids.emplace_back(id);
            auto pid = _->swc_point_mp.at(id).pid;
            _->swc_point_mp.erase(id);
            _->node_mp.erase(id);
            lst_key = id;
            _->direct_influenced_pts[Old_Del].emplace_back(id);
            id = pid;
        }
        _->swc_point_mp.at(id1).pid = id0;
        _->node_mp.at(id1).parent = id0;
        _->node_mp.at(id0).children.erase(lst_key);
        _->node_mp.at(id0).children.insert(id1);
    }

    VISER_END