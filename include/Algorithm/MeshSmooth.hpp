#pragma once

#include <Common/Common.hpp>
#include <IO/MeshIO.hpp>

VISER_BEGIN

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, int worker_count = 0);

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, vutil::thread_group_t& threads);

class MeshSmootherPrivate;
class MeshSmoother : public UnifiedRescBase{
  public:
    struct MeshSmootherCreateInfo{
        Ref<GPUMemMgr> gpu_mem_mgr;
        Ref<HostMemMgr> host_mem_mgr;
        size_t reserved_pinned_host_mem_bytes = 1ull << 30;
    };

    explicit MeshSmoother(const MeshSmootherCreateInfo& );

    ~MeshSmoother();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void Smoothing(MeshData0& mesh, float lambda, float mu, int iterations);

  private:
    std::unique_ptr<MeshSmootherPrivate> _;
};

VISER_END