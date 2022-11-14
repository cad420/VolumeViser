//
// Created by wyz on 2022/11/11.
//
#include <Model/Mesh.hpp>

VISER_BEGIN

    class MeshPrivate{
    public:


        std::mutex mtx;

        UnifiedRescUID uid;
        static UnifiedRescUID GenRescUID(){
            std::atomic<size_t> g_uid = 1;
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

    Mesh Mesh::Merge(const std::vector<Mesh> &meshes, bool remove_dup, bool gen_indices) {
        return Mesh();
    }

    void Mesh::Smooth() {

    }

    void Mesh::Simplify() {

    }

    void Mesh::Lock() {

    }

    void Mesh::UnLock() {

    }

    UnifiedRescUID Mesh::GetUID() const {
        return 0;
    }

    MeshData0 Mesh::GetPackedMeshData0() const {
        return MeshData0();
    }

    MeshData1 Mesh::GetPackedMeshData1() const {
        return MeshData1();
    }

    int Mesh::GetMeshShapeCount() const {
        return 0;
    }

    MeshData0 Mesh::GetShapeData0(int index) const {
        return MeshData0();
    }

    MeshData1 Mesh::GetShapeData1(int index) const {
        return MeshData1();
    }


VISER_END


