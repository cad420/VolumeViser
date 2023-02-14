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
            NOT_IMPL
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
    /**
     * @brief 以lambda函数形式启动kernel，函数参数类型必须是两个dim3，代表blockIdx和threadIdx
     */
    template<typename F>
    static cu_task pending(const cu_kernel_launch_info& info, F func){
        auto kernel = &detail::kernel_impl<F>;
        void* params[] = {&func};
        auto task = [=](const cu_stream& stream) mutable {
            CUB_CHECK(cuLaunchKernel((CUfunction)kernel,
                           info.grid_dim.x,info.grid_dim.y,info.grid_dim.z,
                           info.block_dim.x,info.block_dim.y,info.block_dim.z,
                           info.shared_mem_bytes,stream.get_handle(),
                           params, nullptr));
            CUB_WHEN_DEBUG(std::cout << "cu_kernel launch task: (" << info.grid_dim.x << " " << info.grid_dim.y << " " << info.grid_dim.z
                                     << "), (" << info.block_dim.x << " " << info.block_dim.y << " " << info.block_dim.z << ")" << std::endl)
        };
        return cu_task(task);
    }

    template<typename... Args>
    static cu_task pending(const cu_kernel_launch_info& info, void(*kernel)(Args...), void** params){
        auto task = [=](const cu_stream& stream) mutable {
            auto _ = stream.get_context()->temp_ctx();
            CUB_CHECK(cudaLaunchKernel((const void*)kernel,
                             info.grid_dim, info.block_dim,params,
                                     info.shared_mem_bytes, stream.get_handle()));
            CUB_WHEN_DEBUG(std::cout << "cuda kernel launch task: (" << info.grid_dim.x << " " << info.grid_dim.y << " " << info.grid_dim.z
                                     << "), (" << info.block_dim.x << " " << info.block_dim.y << " " << info.block_dim.z << ")" << std::endl)
        };
        return cu_task(task);
    }

};

CUB_END