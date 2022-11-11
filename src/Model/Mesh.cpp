//
// Created by wyz on 2022/11/11.
//
#include <Model/Mesh.hpp>

VISER_BEGIN

    class MeshPrivate{

    };

    Mesh::Mesh() {

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


VISER_END


