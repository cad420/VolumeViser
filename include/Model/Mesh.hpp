#pragma once

#include <IO/MeshIO.hpp>

VISER_BEGIN

// Mesh相关的算法暂时使用CPU进行处理，有多线程的加速



class MeshPrivate;
class Mesh : public UnifiedRescBase{
public:
    using MeshData = MeshData0;

    Mesh();

    ~Mesh();

    static Handle<Mesh> Merge(const std::vector<Handle<Mesh>>& meshes);


    void Merge(const Mesh& mesh);

    void Clear();

    void Insert(MeshData data, int shape_index);

    void Smooth();

    void Simplify();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    int GetMeshShapeCount() const;

    /**
     * @brief 获取单个shape的网格数据
     */
    MeshData GetShapeData(int index) const;


    //将一个mesh里的所有shape合并为一个
    void MergeShape();

    /**
     * @brief 将所有的shape进行合并后返回
     */
    MeshData GetPackedMeshData() const;


protected:
    std::unique_ptr<MeshPrivate> _;
};


VISER_END
