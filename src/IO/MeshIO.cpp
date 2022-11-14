#include <IO/MeshIO.hpp>

VISER_BEGIN

class MeshFilePrivate{
public:


};

MeshFile::MeshFile(std::string_view filename) {

}

MeshFile::~MeshFile() {

}

void MeshFile::Lock() {

}

void MeshFile::UnLock() {

}

UnifiedRescUID MeshFile::GetUID() const {
    return 0;
}

void MeshFile::Open(std::string_view filename, MeshFile::Mode mode) {

}

std::vector<MeshData0> MeshFile::GetMesh() {
    return std::vector<MeshData0>();
}

void MeshFile::WriteMeshData(const MeshData0 &) {

}

void MeshFile::Close() {

}


VISER_END