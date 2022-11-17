#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN

using PosType = Float3;
using NormalType = Float3;
using IndexType = uint32_t;
struct Vertex{
    Vertex() = default;
    Vertex(const PosType& pos, const NormalType& normal)
    :pos(pos), normal(normal)
    {}

    PosType pos;
    NormalType normal;

    bool operator==(const Vertex& vert) const{
        return pos == vert.pos && normal == vert.normal;
    }
};
struct MeshData0{
    MeshData0() = default;
    MeshData0(const MeshData0&) = default;
    MeshData0& operator=(const MeshData0&) = default;
    MeshData0(MeshData0&& other) noexcept
    :vertices(std::move(other.vertices)), indices(std::move(other.indices))
    {}
    MeshData0& operator=(MeshData0&& other) noexcept{
        vertices = std::move(other.vertices);
        indices = std::move(other.indices);
        return *this;
    }
    std::vector<Vertex> vertices;
    std::vector<IndexType> indices;
};
struct MeshData1{
    MeshData1() = default;
    MeshData1(const MeshData1&) = default;
    MeshData1& operator=(const MeshData1&) = default;
    MeshData1(MeshData1&& other) noexcept
    :pos(std::move(other.pos)), normal(std::move(other.normal)), indices(std::move(other.indices))
    {}
    MeshData1& operator=(MeshData1&& other) noexcept{
        pos = std::move(other.pos);
        normal = std::move(other.normal);
        indices = std::move(other.indices);
        return *this;
    }
    std::vector<PosType> pos;
    std::vector<NormalType> normal;
    std::vector<IndexType> indices;
};

MeshData1 ConvertTo_C(const MeshData0&) noexcept;
MeshData1 ConvertTo(MeshData0) noexcept;
MeshData0 ConvertFrom_C(const MeshData1&) noexcept;
MeshData0 ConvertFrom(MeshData1) noexcept;

MeshData0 Merge(const MeshData0& data0, const MeshData0& data1);

MeshData0 Merge(const std::vector<MeshData0>& meshes);

class MeshFilePrivate;
class MeshFile : public MeshIOInterface{
public:
    static constexpr const char* MESH_FILENAME_EXT_OBJ = ".obj";

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

namespace std{
    template<>
    struct hash<viser::Vertex>{
        size_t operator()(const viser::Vertex& vert) const{
            return vutil::hash(vert.pos, vert.normal);
        }
    };
}