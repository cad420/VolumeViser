#pragma once

#include <IO/MeshIO.hpp>

VISER_BEGIN

// Mesh相关的算法暂时使用CPU进行处理，有多线程的加速

class MeshPrivate;
class Mesh : public UnifiedRescBase{
public:
    Mesh();

    ~Mesh();

    static Mesh Merge(const std::vector<Mesh>& meshes, bool remove_dup, bool gen_indices);

    void Smooth();

    void Simplify();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    int GetMeshShapeCount() const;

    /**
     * @brief 获取单个shape的网格数据
     */
    MeshData0 GetShapeData0(int index) const;

    MeshData1 GetShapeData1(int index) const;

    /**
     * @brief 将所有的shape进行合并后返回
     */
    MeshData0 GetPackedMeshData0() const;

    MeshData1 GetPackedMeshData1() const;

protected:
    std::unique_ptr<MeshPrivate> _;
};


VISER_END
