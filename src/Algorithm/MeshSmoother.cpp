#undef UTIL_ENABLE_OPENGL

#include <Algorithm/MeshSmooth.hpp>


VISER_BEGIN

namespace {

    struct MeshSmoothingParams{

    };


    CUB_KERNEL void MeshSmoothingKernel(MeshSmoothingParams params){

    }

}

class MeshSmootherPrivate{
  public:

};

MeshSmoother::MeshSmoother(const MeshSmootherCreateInfo & info)
{

}

MeshSmoother::~MeshSmoother()
{

}

void MeshSmoother::Lock()
{

}

void MeshSmoother::UnLock()
{

}

UnifiedRescUID MeshSmoother::GetUID() const
{
    return 0;
}

void MeshSmoother::Smoothing(MeshData0 &mesh, float lambda, float mu, int iterations)
{

}

VISER_END