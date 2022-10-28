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
    struct Inner{
        std::mutex mtx;
        cu_context ctx;
        CUstream stream = nullptr;
    };
public:
    friend class cu_event;
    friend class cu_kernel;

    template<typename T, int N>
    friend struct detail::cu_buffer_transfer;
    template<typename T, int N>
    friend struct detail::cu_array_transfer;

    struct lock_t{
        lock_t(Inner& inner):
        lk(inner.mtx),
        stream(inner.stream)
        {}

        auto get() {
            return stream;
        }

        CUstream stream;
        std::unique_lock<std::mutex> lk;
    };

    lock_t lock(){
        return lock_t(*_);
    }

    cu_result wait(){
        _->ctx.set_ctx();
        return cuStreamSynchronize(_->stream);
    }

    cu_stream(cu_context ctx)
    {
        assert(ctx.is_valid());
        ctx.set_ctx();
        // todo CU_STREAM_NON_BLOCKING
        CUB_CHECK(cuStreamCreate(&_->stream, CU_STREAM_DEFAULT));
        _->ctx = ctx;
    }

    static cu_stream null(cu_context ctx){
        return cu_stream(nullptr, ctx);
    }

    cu_stream() = default;

    bool is_valid() {
        return _->ctx.is_valid();
    }

private:
    cu_stream(std::nullptr_t, cu_context ctx)
    {
        assert(ctx.is_valid());
        _->ctx = ctx;
    }


protected:
    std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

class cu_event{
public:
    cu_event(bool enable_blocking = true, bool enable_timing = false){
        uint32_t flags = 0;
        if(enable_blocking)
            flags |= CU_EVENT_BLOCKING_SYNC;
        if(!enable_timing)
            flags |= CU_EVENT_DISABLE_TIMING;
        CUB_CHECK(cuEventCreate(&event, flags));
    }

    void record(cu_stream& stream) const{
        CUB_CHECK(cuEventRecord(event, stream.lock().get()));
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

    cu_result launch(cu_stream stream){
        task(stream);
        return stream.wait();
    }

    std::future<cu_result> launch_async(cu_stream stream){
        cu_event start, stop;
        start.record(stream);
        task(stream);
        stop.record(stream);
        return std::async(std::launch::deferred, [=]{
            auto ret = stop.wait();
            CUB_WHEN_DEBUG(std::cout << "async task cost time : " << cu_event::elapsed(start, stop)
                                     << "ms" << std::endl)
            return ret;
        });
    }
    friend class cu_task_group;
protected:
    std::function<void(cu_stream&)> task;
};

class cu_task_group{
public:
    cu_task_group() = default;

    cu_result launch(cu_stream& stream){
        for(auto& task : tasks){
            task.task(stream);
        }
        return stream.wait();
    }

    std::future<cu_result> launch_async(cu_stream& stream){
        cu_event start, stop;

        start.record(stream);
        for(auto& task : tasks)
            task.task(stream);
        stop.record(stream);

        return std::async(std::launch::deferred, [=]{
            auto ret = stop.wait();
            CUB_WHEN_DEBUG(std::cout << "async task cost time : " << cu_event::elapsed(start, stop)
            << "ms" << std::endl)
            return ret;
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
    cu_submitted_tasks& add(std::future<cu_result>&& res){
        tasks.emplace_back(std::move(res));
        return *this;
    }
    std::vector<cu_result> wait(){
        std::vector<cu_result> ret;
        for(auto& t : tasks){
            t.wait();
            ret.emplace_back(t.get());
        }
        tasks.clear();
        return ret;
    }
private:
    std::vector<std::future<cu_result>> tasks;
};

CUB_END