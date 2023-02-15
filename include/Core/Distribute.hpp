#pragma once

#include <Common/Common.hpp>

//分布式资源、任务管理，除了实时大屏渲染可以使用多节点并行渲染外，
//离线渲染可以使用多个节点的资源，比如每个节点分配不同的时间序列
//


VISER_BEGIN

class DistributeMgrPrivate;
class DistributeMgr : public UnifiedRescBase{
  public:
    struct DistributeMgrCreateInfo{
        int root_rank = 0;

    };

    explicit DistributeMgr(const DistributeMgrCreateInfo&);

    ~DistributeMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void RunTask(std::function<void()> task);

    template<typename T>
    void Bcast(std::function<void(T*, int, int)>);

    void WaitForSync();

    int GetWorldRank() const;

    int GetWorldSize() const;

    int GetRootRank() const;

    bool IsRoot() const;

  private:
    std::unique_ptr<DistributeMgrPrivate> _;
};



VISER_END

