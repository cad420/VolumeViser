//
// Created by wyz on 2022/11/11.
//
#include <Model/Mesh.hpp>
#include <Algorithm/MeshSmooth.hpp>

VISER_BEGIN

class MeshPrivate{
public:
    std::unordered_map<int, Mesh::MeshData> shapes;

    std::mutex mtx;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::Mesh);
    }
};

Mesh::Mesh() {
    _ = std::make_unique<MeshPrivate>();

    _->uid = _->GenRescUID();

}

Mesh::~Mesh() {

}

void Mesh::Smooth(float lambda, float mu, int iterations) {
    if(GetMeshShapeCount() > 1){
        MergeShape();
    }

    MeshSmoothing(_->shapes.at(0), lambda, mu, iterations, 0);
}

void Mesh::Simplify() {

}

void Mesh::Lock() {
    _->mtx.lock();
}

void Mesh::UnLock() {
    _->mtx.unlock();
}

UnifiedRescUID Mesh::GetUID() const {
    return _->uid;
}

Mesh::MeshData Mesh::GetPackedMeshData() const {
    std::vector<MeshData0> ret;
    for(auto& [id, mesh_data] : _->shapes){
        ret.emplace_back(mesh_data);
    }
    return ::viser::Merge(ret);
}


int Mesh::GetMeshShapeCount() const {
    return _->shapes.size();
}

Mesh::MeshData Mesh::GetShapeData(int index) const {
    if(_->shapes.count(index == 0)){
        LOG_ERROR("GetShapeData with invalid index");
        return {};
    }
    return _->shapes.at(index);
}


void Mesh::Insert(MeshData0 data, int shape_index) {
    if(_->shapes.count(shape_index) != 0){
        LOG_WARN("Mesh Insert with existed index");
    }
    _->shapes[shape_index] = std::move(data);
}


void Mesh::MergeShape() {
    std::vector<MeshData0> res;
    for(auto& [id, mesh_data] : _->shapes){
        res.emplace_back(std::move(mesh_data));
    }
    auto ret = ::viser::Merge(res);
    _->shapes.clear();
    _->shapes[0] = std::move(ret);
}

void Mesh::Merge(const Mesh &mesh) {
    MergeShape();
    auto ret = ::viser::Merge(_->shapes.at(0), mesh.GetPackedMeshData());
    _->shapes.at(0) = std::move(ret);
}

void Mesh::Clear() {
    _->shapes.clear();
}

Handle<Mesh> Mesh::Merge(const std::vector<Handle<Mesh>> &meshes) {
    auto ret = NewHandle<Mesh>(ResourceType::Object);
    std::vector<MeshData0> datas;
    for(auto mesh : meshes){
        datas.emplace_back(mesh->GetPackedMeshData());
    }
    ret->Insert(::viser::Merge(datas), 0);

    return ret;
}

bool Mesh::Empty() const {
    for(auto& [id, mesh_data] : _->shapes){
        if(mesh_data.indices.size() > 0)
            return false;
    }
    return true;
}

VISER_END


