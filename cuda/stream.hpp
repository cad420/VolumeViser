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

    constexpr cu_stream() = default;

    cu_stream(cu_context_handle_t ctx)
    {
        assert(ctx->is_valid());
        this->ctx = ctx;
        auto _ = ctx->temp_ctx();
        // todo CU_STREAM_NON_BLOCKING
        CUB_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    }

    static cu_stream null(cu_context_handle_t ctx){
        return cu_stream(nullptr, ctx);
    }

    cu_result wait() const {
        auto _ = ctx->temp_ctx();
        return cuStreamSynchronize(stream);
    }


    bool is_valid() const {
        return stream;
    }

    auto get_context() const {
        return ctx;
    }

    auto get_handle() const {
        return stream;
    }

private:
    cu_stream(std::nullptr_t, cu_context_handle_t ctx)
    {
        assert(ctx->is_valid());
        this->ctx = ctx;
    }

    cu_context_handle_t ctx{nullptr};
    CUstream stream{nullptr};
};

class cu_event{
public:
    cu_event(cu_context_handle_t ctx, bool enable_blocking = true, bool enable_timing = true){
        assert(ctx->is_valid());
        this->ctx = ctx;
        auto _ = ctx->temp_ctx();

        uint32_t flags = 0;
        if(enable_blocking)
            flags |= CU_EVENT_BLOCKING_SYNC;
        if(!enable_timing)
            flags |= CU_EVENT_DISABLE_TIMING;
        CUB_CHECK(cuEventCreate(&event, flags));
    }

    void record(const cu_stream& stream) const{
        auto _ = ctx->temp_ctx();
        CUB_CHECK(cuEventRecord(event, stream.get_handle()));
    }

    cu_result wait() const{
        auto _ = ctx->temp_ctx();
        return cuEventSynchronize(event);
    }

    static float elapsed(const cu_event& a, const cu_event& b){
        assert(a.ctx == b.ctx);
        auto _ = a.ctx->temp_ctx();
        float t;
        CUB_CHECK(cuEventElapsedTime(&t, a.event, b.event));
        return t;
    }

private:
    CUevent event;
    cu_context_handle_t ctx;
};

class cu_task{
public:
    cu_task(std::function<void(const cu_stream&)> func)
    :task(std::move(func))
    {

    }

    cu_task(cu_task&& other) noexcept
    :task(std::move(other.task))
    {}

    cu_result launch(const cu_stream& stream){
        task(stream);
        return stream.wait();
    }

    std::future<cu_result> launch_async(const cu_stream& stream){
        cu_event start(stream.get_context()), stop(stream.get_context());
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
    std::function<void(const cu_stream&)> task;
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
        cu_event start(stream.get_context()), stop(stream.get_context());

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
    cu_submitted_tasks& add(std::future<cu_result>&& res, size_t uid = 0){
        tasks.emplace_back(uid, std::move(res));
        return *this;
    }
    std::vector<std::pair<size_t, cu_result>> wait(){
        std::vector<std::pair<size_t, cu_result>> ret;
        for(auto& [_, t] : tasks){
            t.wait();
            ret.emplace_back(_, t.get());
        }
        tasks.clear();
        return ret;
    }
private:
    std::vector<std::pair<size_t, std::future<cu_result>>> tasks;
};

CUB_END