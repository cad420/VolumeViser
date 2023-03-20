#pragma once

#include <Common/Common.hpp>

//分布式资源、任务管理，除了实时大屏渲染可以使用多节点并行渲染外，
//离线渲染可以使用多个节点的资源，比如每个节点分配不同的时间序列
//


VISER_BEGIN

enum class DistrType: int {
    UINT8 = 0,
    INT8,
    INT32,
    UINT32,
    FLOAT32,
    INT64,
    UINT64,
    FLOAT64
};

class DistributeMgr : public UnifiedRescBase{

    DistributeMgr();

  public:
    static DistributeMgr& GetInstance();

    ~DistributeMgr();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    template<typename T>
    void Bcast(T* data, int count);

    void WaitForSync();

    void SetRootRank(int rank);

    int GetWorldRank();

    int GetWorldSize();

    int GetRootRank();

    bool IsRoot();

  private:
    void Bcast(void* data, int count, int type);

};

template <typename T>
void DistributeMgr::Bcast(T *data, int count)
{
    if constexpr(std::is_same_v<T, int>){
        Bcast(data, count, static_cast<int>(DistrType::INT32));
    }
    else if constexpr(std::is_same_v<T, uint64_t>){
        Bcast(data, count, static_cast<int>(DistrType::UINT64));
    }
    else if constexpr(std::is_same_v<T, int64_t>){
        Bcast(data, count, static_cast<int>(DistrType::INT64));
    }
    else if constexpr(std::is_same_v<T, uint32_t>){
        Bcast(data, count, static_cast<int>(DistrType::UINT32));
    }
    else if constexpr(std::is_same_v<T, float>){
        Bcast(data, count, static_cast<int>(DistrType::FLOAT32));
    }
    else if constexpr(std::is_same_v<T, double>){
        Bcast(data, count, static_cast<int>(DistrType::FLOAT64));
    }
    else if constexpr(std::is_same_v<T, uint8_t>){
        Bcast(data, count, static_cast<int>(DistrType::UINT8));
    }
    else if constexpr(std::is_same_v<T, char>){
        Bcast(data, count, static_cast<int>(DistrType::INT8));
    }
    else{
        LOG_ERROR("type not implied for Bcast");
    }
}

VISER_END