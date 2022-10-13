#pragma once

#include "common.hpp"
#include <mutex>
#include <future>
CUB_BEGIN

    namespace detail{
        template<typename T, int N>
        struct cu_buffer_transfer;
        template<typename T, int N>
        struct cu_array_transfer;
    }

class cu_stream{
public:
    friend class cu_event;
    friend class cu_kernel;

    template<typename T, int N>
    friend struct detail::cu_buffer_transfer;
    template<typename T, int N>
    friend struct detail::cu_array_transfer;
    struct lock_t{
        lock_t(std::mutex& mtx):lk(mtx){}
        std::unique_lock<std::mutex> lk;
    };
    lock_t lock(){
        return lock_t(mtx);
    }
    cu_result wait(){
        return cuStreamSynchronize(stream);
    }
private:
    std::mutex mtx;
protected:
    CUstream stream;
};


class cu_event{
public:
    cu_event(bool enable_blocking = true, bool enable_timing = true){
        uint32_t flags = 0;
        if(enable_blocking)
            flags |= CU_EVENT_BLOCKING_SYNC;
        if(!enable_timing)
            flags |= CU_EVENT_DISABLE_TIMING;
        CUB_CHECK(cuEventCreate(&event, flags));
    }

    void record(cu_stream& stream) const{
        CUB_CHECK(cuEventRecord(event, stream.stream));
    }

    cu_result wait() const{
        return cuEventSynchronize(event);
    }


    static float elapsed(const cu_event& a, const cu_event& b){
        float t;
        CUB_CHECK(cuEventElapsedTime(&t, a.event, b.event));
        return t;
    }

private:
    CUevent event;
};

class cu_task{
public:
    cu_task(std::function<void(cu_stream&)> func)
    :task(std::move(func))
    {

    }
    cu_task(cu_task&& other) noexcept
    :task(std::move(other.task))
    {}

    void launch(cu_stream stream){

    }

    void launch_async(cu_stream stream){

    }
    friend class cu_task_group;
protected:
    std::function<void(cu_stream&)> task;
};

class cu_task_group{
public:
    cu_task_group() = default;

    cu_result launch(cu_stream& stream){
        auto lk = stream.lock();
        for(auto& task : tasks){
            task.task(stream);
        }
        return stream.wait();
    }

    std::future<cu_result> launch_async(cu_stream& stream){
        cu_event start, stop;
        {
            auto lk = stream.lock();
            start.record(stream);
            for(auto& task : tasks)
                task.task(stream);
            stop.record(stream);
        }
        return std::async(std::launch::deferred, [=]{
            return stop.wait();
        });
    }

    cu_task_group& enqueue_task(cu_task&& task){
        tasks.emplace_back(std::move(task));
        return *this;
    }

private:
    std::vector<cu_task> tasks;
};

struct cu_submitted_tasks{

};

CUB_END
