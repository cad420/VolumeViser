#include <IO/MeshIO.hpp>
#include <fstream>
#include <misc.hpp>
VISER_BEGIN

class MeshFilePrivate{
public:
    std::vector<MeshData0> data;

    void ReadMeshFromObjFile(const std::string& filename){
        auto meshes = vutil::load_mesh_from_obj_file(filename);

        for(auto& mesh : meshes){
            auto& mesh_data = data.emplace_back();
            mesh_data.indices = mesh.indices;
            mesh_data.vertices.reserve(mesh.vertices.size());
            for(auto& vert : mesh.vertices){
                mesh_data.vertices.emplace_back(vert.pos, vert.normal);
            }
        }

    }

    std::fstream fs;

    UnifiedRescUID uid;

    std::mutex mtx;
};

MeshFile::MeshFile() {
    _ = std::make_unique<MeshFilePrivate>();

    _->uid = MeshIOInterface::GenUnifiedRescUID();

}

MeshFile::~MeshFile() {
    Close();
}

void MeshFile::Lock() {
    _->mtx.lock();
}

void MeshFile::UnLock() {
    _->mtx.unlock();
}

UnifiedRescUID MeshFile::GetUID() const {
    return _->uid;
}

void MeshFile::Open(std::string_view filename, MeshFile::Mode mode) {
    if(mode == Read){
        // read while open
        if(vutil::ends_with(filename, MESH_FILENAME_EXT_OBJ)){
            _->ReadMeshFromObjFile(filename.data());
        }
        else{
            throw ViserFileOpenError("Invalid ext for open Mesh file to read");
        }
    }
    else if(mode == Write){
        if(vutil::ends_with(filename, MESH_FILENAME_EXT_OBJ)){
            _->fs.open(filename.data(), std::ios::out);
        }
        else{
            throw ViserFileOpenError("Invalid ext for open Mesh file to write");
        }
    }
    else{
        assert(false);
    }
}

std::vector<MeshData0> MeshFile::GetMesh() {
    return std::move(_->data);
}

void MeshFile::WriteMeshData(const MeshData0& data) {
    assert(data.indices.size() % 3 == 0);

    int tri_count = data.indices.size() / 3;
    uint32_t norm_idx = 0;
    std::unordered_map<Float3, uint32_t> normals;
    struct Tri{
        uint32_t pos[3];
        uint32_t norm[3];
    };
    std::vector<Tri> tris;
    tris.reserve(tri_count);
    for(int i = 0; i < tri_count; i++){
        auto& tri = tris.emplace_back();
        for(int j = 0; j < 3; j++){
            auto k = i * 3 + j;
            auto idx = data.indices[k];
            auto& pos = data.vertices[idx].pos;
            auto& norm = data.vertices[idx].normal;
            tri.pos[j] = idx + 1;
            if(normals.count(norm) == 0){
                normals[norm] = norm_idx++;
            }
            tri.norm[j] = normals[norm] + 1;
        }
    }
    auto Float3_to_str = [](std::string prefix, const Float3& f){
        std::string ret = prefix + std::to_string(f.x) + " "
                + std::to_string(f.y) + " " + std::to_string(f.z) + "\n";
        return ret;
    };
    for(auto& vert : data.vertices){
        auto str = Float3_to_str("v ", vert.pos);
        _->fs.write(str.data(), str.length());
    }

    std::vector<std::pair<Float3, uint32_t>> normal_array;
    normal_array.reserve(normals.size());
    for(auto& norm : normals) normal_array.emplace_back(norm);
    std::sort(normal_array.begin(), normal_array.end(), [](const auto& a, const auto& b){
       return a.second < b.second;
    });
    for(auto& [norm, idx] : normal_array){
        auto str = Float3_to_str("vn ", norm);
        _->fs.write(str.data(), str.length());
    }
    for(auto& tri : tris){
        std::string str = "f";
        for(int i = 0; i < 3; i++){
            str += " " + std::to_string(tri.pos[i]) + "/" + std::to_string(tri.norm[i]);
        }
        str += "\n";
        _->fs.write(str.data(), str.length());
    }
    _->fs.flush();
}

void MeshFile::Close() {
    _->data.clear();
    if(_->fs.is_open())
        _->fs.close();
}

MeshData1 ConvertTo_C(const MeshData0& data) noexcept {
    MeshData1 ret;
    ret.indices = data.indices;
    int vert_count = data.vertices.size();
    ret.pos.resize(vert_count);
    ret.normal.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        ret.pos[i] = data.vertices[i].pos;
        ret.normal[i] = data.vertices[i].normal;
    }
    return ret;
}
MeshData1 ConvertTo(MeshData0 data) noexcept{
    MeshData1 ret;
    ret.indices = std::move(data.indices);
    int vert_count = data.vertices.size();
    ret.pos.resize(vert_count);
    ret.normal.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        ret.pos[i] = data.vertices[i].pos;
        ret.normal[i] = data.vertices[i].normal;
    }
    return ret;
}
MeshData0 ConvertFrom_C(const MeshData1& data) noexcept {
    MeshData0 ret;
    ret.indices = data.indices;
    assert(data.pos.size() == data.normal.size());
    int vert_count = data.pos.size();
    ret.vertices.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        ret.vertices[i].pos = data.pos[i];
        ret.vertices[i].normal = data.normal[i];
    }
    return ret;
}
MeshData0 ConvertFrom(MeshData1 data) noexcept{
    MeshData0 ret;
    ret.indices = std::move(data.indices);
    assert(data.pos.size() == data.normal.size());
    int vert_count = data.pos.size();
    ret.vertices.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        ret.vertices[i].pos = data.pos[i];
        ret.vertices[i].normal = data.normal[i];
    }
    return ret;
}

MeshData0 Merge(const MeshData0& data0, const MeshData0& data1){
    return Merge({data0, data1});
}

MeshData0 Merge(const std::vector<MeshData0>& meshes){
    int n = meshes.size();
    uint32_t idx = 0;
    std::unordered_map<Vertex, uint32_t> vert_mp;
    MeshData0 ret;
    for(auto& mesh : meshes){
        int tri_count = mesh.indices.size() / 3;
        for(int i = 0; i < tri_count; i++){
            for(int j = 0; j < 3; j++){
                uint32_t vert_idx = mesh.indices[i * 3 + j];
                auto& vert = mesh.vertices[vert_idx];
                if(vert_mp.count(vert) == 0){
                    vert_mp[vert] = idx++;
                    ret.vertices.push_back(vert);
                }
                ret.indices.push_back(vert_mp.at(vert));
            }
        }
    }
    return ret;
}

VISER_END