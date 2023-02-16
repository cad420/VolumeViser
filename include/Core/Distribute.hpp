#pragma once

#include <Common/Common.hpp>

//分布式资源、任务管理，除了实时大屏渲染可以使用多节点并行渲染外，
//离线渲染可以使用多个节点的资源，比如每个节点分配不同的时间序列
//


VISER_BEGIN

class DistributeMgr : public UnifiedRescBase{

    DistributeMgr();

  public:
    static DistributeMgr& GetInstance();

    ~DistributeMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;


    template<typename T>
    void Bcast(std::function<void(T*, int, int)>);

    void WaitForSync();

    void SetWorldRank(int rank);

    int GetWorldRank();

    int GetWorldSize();

    int GetRootRank();

    bool IsRoot();

};


VISER_END

