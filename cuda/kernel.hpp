#pragma once
#include "context.hpp"
#include "stream.hpp"
CUB_BEGIN



namespace detail{
#ifdef __NVCC__
    template<typename F>
    __global__ void kernel_impl(F func){
        func(blockIdx, threadIdx);
    }
#else
        template<typename F>
        void kernel_impl(F func){

        }
#endif
}

struct cu_kernel_launch_info{
    dim3 grid_dim;
    dim3 block_dim;
    uint32_t shared_mem_bytes = 0;
};


class cu_kernel{
public:

    explicit cu_kernel(cu_context ctx)
    :ctx(ctx)
    {

    }

    cu_context get_context() const {
        return ctx;
    }

    template<typename F>
    cu_task pending(const cu_kernel_launch_info& info, F func){
        auto kernel = &detail::kernel_impl<F>;
        void* params[] = {&func};
        auto task = [=](cu_stream& stream) mutable {
            CUB_CHECK(cuLaunchKernel((CUfunction)kernel,
                           info.grid_dim.x,info.grid_dim.y,info.grid_dim.z,
                           info.block_dim.x,info.block_dim.y,info.block_dim.z,
                           info.shared_mem_bytes,stream.stream,
                           params, nullptr));
            std::cout << "cu_kernel launch task: " << info.grid_dim.x << " " << info.grid_dim.y << " " << info.grid_dim.z << std::endl;
        };
        return cu_task(task);
    }
    template<typename... Args>
    cu_task pending(const cu_kernel_launch_info& info, void(*kernel)(Args...), Args... args){
        void* params[sizeof...(args)] = {&args...};
        auto task = [=](cu_stream& stream) mutable {
            CUB_CHECK(cuLaunchKernel((CUfunction)kernel,
                                     info.grid_dim.x,info.grid_dim.y,info.grid_dim.z,
                                     info.block_dim.x,info.block_dim.y,info.block_dim.z,
                                     info.shared_mem_bytes,stream.stream,
                                     params, nullptr));
            std::cout << "cu_kernel launch task: " << info.grid_dim.x << " " << info.grid_dim.y << " " << info.grid_dim.z << std::endl;
        };
        return cu_task(task);
    }
private:
    cu_context ctx;
};


    inline cu_kernel cu_context::create_kernel() {
        return cu_kernel(*this);
    }



CUB_END