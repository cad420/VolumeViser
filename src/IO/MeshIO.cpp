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
            str += " " + std::to_string(tri.pos[i]) + "//" + std::to_string(tri.norm[i]);
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

MeshData0::MeshData0(int tri_count, std::function<const PosType &(int)> get_vert) {
    struct TriFace{
        uint32_t idx[3];
    };
    struct V2F{
        std::vector<uint32_t> faces;
    };
    uint32_t index = 0;
    std::unordered_map<PosType, uint32_t> tri_pos_idx_mp;
    std::vector<PosType> unique_pos;
    std::vector<NormalType> unique_norm;
    std::vector<TriFace> tris;
    std::unordered_map<uint32_t, V2F> vert_face_mp;
    tris.reserve(tri_count);

    this->indices.resize(tri_count * 3);

    for(int i = 0; i < tri_count; i++){
        auto& tri_face = tris.emplace_back();
        for(int j = 0; j < 3; j++){
            auto& vert_pos = get_vert(i * 3 + j);
//            if(vert_pos.x < 3.f || vert_pos.y < 5.f || vert_pos.z < 5.f){
//                LOG_ERROR("vert pos error: i {}, j {}, {} {} {}", i, j, vert_pos.x, vert_pos.y, vert_pos.z);
//            }
            if(tri_pos_idx_mp.count(vert_pos) == 0){
                unique_pos.emplace_back(vert_pos);
                tri_pos_idx_mp[vert_pos] = index++;
            }
//            try{
                auto idx = tri_pos_idx_mp.at(vert_pos);
                tri_face.idx[j] = idx;
                vert_face_mp[idx].faces.emplace_back(i);
                this->indices[i * 3 + j] = idx;
//            }
//            catch(const std::exception& err){
//                LOG_ERROR("vert pos error: i {}, j {}, {} {} {}, has: {}", i, j, vert_pos.x, vert_pos.y, vert_pos.z,
//                          tri_pos_idx_mp.count(vert_pos));
//            }
        }
    }



    auto get_normal = [&](uint32_t face_idx){
        auto vert_a = unique_pos[tris[face_idx].idx[0]];
        auto vert_b = unique_pos[tris[face_idx].idx[1]];
        auto vert_c = unique_pos[tris[face_idx].idx[2]];
        auto ab = vert_b - vert_a;
        auto ac = vert_c - vert_a;
        auto norm = vutil::cross(ab, ac);
        return norm.normalized();
    };
    int vert_count = unique_pos.size();
    unique_norm.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        auto& vert = unique_pos[i];
        auto vert_idx = tri_pos_idx_mp.at(vert);
        NormalType norm;
        for(auto f : vert_face_mp.at(vert_idx).faces){
            norm += get_normal(f);
        }
        unique_norm[i] = norm.normalized();
    }

    this->vertices.resize(vert_count);
    for(int i = 0; i < vert_count; i++){
        this->vertices[i] = {unique_pos[i], unique_norm[i]};
    }
    LOG_TRACE("input tri num : {}, final gen vert num : {}", tri_count, vert_count);
}

VISER_END


