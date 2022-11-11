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

protected:
    std::unique_ptr<MeshPrivate> _;
};


VISER_END
