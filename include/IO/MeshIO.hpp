#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN

using PosType = Float3;
using NormalType = Float3;
using IndexType = uint32_t;
struct Vertex{
    PosType pos;
    NormalType normal;
};
struct MeshData0{
    std::vector<Vertex> vertices;
    std::vector<IndexType> indices;
};
struct MeshData1{
    std::vector<PosType> pos;
    std::vector<NormalType> normal;
    std::vector<IndexType> indices;
};

MeshData1 ConvertTo(const MeshData0&) noexcept;
MeshData0 ConvertFrom(const MeshData1&) noexcept;

class MeshFilePrivate;
class MeshFile : public MeshIOInterface{
public:
    explicit MeshFile();

    ~MeshFile();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    enum Mode{
        Read,
        Write
    };

    void Open(std::string_view filename, Mode mode);

    std::vector<MeshData0> GetMesh();

    void WriteMeshData(const MeshData0&);

    void Close();

private:
    std::unique_ptr<MeshFilePrivate> _;
};


VISER_END
