#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>
#include <functional>
#include <iostream>
#include <cassert>
#ifndef __CUDACC__
//#error "Can not find CUDA"
#define CUB_KERNEL
#define CUB_CPU
#define CUB_GPU
#define CUB_CPU_GPU
#define CUB_CONST const
#define CUB_NOINLINE
#define CUB_CPU_GPU_LAMBDA_CLS(...) [=, *this](__VA_ARGS__) mutable
#define CUB_CPU_GPU_LAMBDA(...) [=](__VA_ARGS__) mutable
#define CUB_GPU_LAMBDA(...) [=](__VA_ARGS__) mutable
#define CUB_L1_CACHE_LINE_SIZE 64
#else
#define CUB_KERNEL __global__
#define CUB_CPU __host__
#define CUB_GPU __device__
#define CUB_CPU_GPU __host__ __device__
#define CUB_CONST __device__ const
#define CUB_NOINLINE __attribute__((noinline))

#define CUB_CPU_GPU_LAMBDA_CLS(...) [=, *this] CUB_CPU_GPU(__VA_ARGS__) mutable
#define CUB_CPU_GPU_LAMBDA(...) [=] CUB_CPU_GPU(__VA_ARGS__) mutable
#define CUB_GPU_LAMBDA(...) [=] CUB_GPU(__VA_ARGS__) mutable


#define CUB_L1_CACHE_LINE_SIZE 128
#endif

#ifndef NDEBUG
#define CUB_DEBUG
#define CUB_WHEN_DEBUG(op) do { op; } while(false);
#else
#define CUB_WHEN_DEBUG(op) do { } while(false);
#endif

#define CUB_CHECK(expr) check_cuda_call(expr)

#define CUB_BEGIN namespace cub{
#define CUB_END }


CUB_BEGIN

class cuda_error : public std::runtime_error{
public:
    using std::runtime_error::runtime_error;
};

inline void check_cuda_call(CUresult ret){
    if(ret != CUDA_SUCCESS){
        const char* err_str;
        cuGetErrorString(ret, &err_str);
        throw cuda_error(err_str);
    }
}

inline void check_cuda_call(cudaError err){
    if(err != cudaSuccess)
        throw cuda_error(cudaGetErrorString(err));
}

class cu_result{
public:
    cu_result(CUresult ret = CUDA_SUCCESS)
            :ret(ret)
    {}
    bool ok() const{
        return ret == CUDA_SUCCESS;
    }
    bool error() const{
        return ret != CUDA_SUCCESS;
    }

    const char* name() const{
        const char* s;
        cuGetErrorName(ret, &s);
        return s;
    }
    const char* msg() const{
        const char* s;
        cuGetErrorString(ret, &s);
        return s;
    }
    void check_error_on_throw(){
        check_cuda_call(ret);
    }

    bool is(CUresult other) const{
        return ret == other;
    }

    friend std::ostream& operator<<(std::ostream& os, const cu_result& r){
        if(r.ok())
            os << "CUDA result OK";
        else
            os << "CUDA result error(" << r.name()<<") : " << r.msg();
        return os;
    }
private:
    CUresult ret;
};




struct cu_extent{
    size_t width = 0;
    size_t height = 1;
    size_t depth = 1;
};

enum memory_type{
    e_cu_host,
    e_cu_device
};

class cu_context;
class cu_physical_device;
class cu_kernel;
template<bool>
class cu_buffer;
template<typename, int>
class cu_array;
class cu_texture;
struct texture_resc_info;
struct texture_view_info;
class cu_texture_wrap;
class cu_event;
class cu_stream;

CUB_END