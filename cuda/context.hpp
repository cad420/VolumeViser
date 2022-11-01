#pragma once

#include "common.hpp"

CUB_BEGIN

/**
 * @brief CUDA的上下文，分配各种CUDA相关的资源，不同上下文之间的资源是不可见的，
 * 本质上是一个句柄，只有8bytes大小
 */
class cu_context{
public:
    cu_context() = default;

    cu_context(cu_physical_device device, uint32_t flags);

    /**
     * @brief 分配host或者device buffer
     */
    cu_buffer<false> alloc_buffer(size_t size, memory_type type);

    /**
     * @brief 分配device上pitched buffer
     */
    cu_buffer<true> alloc_pitched_buffer(size_t width_bytes, size_t height, uint32_t ele_size);

    template<typename T, int N, typename... Args>
    cu_array<T, N> alloc_array(Args&&... args) const;

    /**
     * @brief 分配包括了array的texture资源
     */
    cu_texture_wrap alloc_texture(const texture_resc_info& resc_info, const texture_view_info& view_info);

    bool operator==(const cu_context& other) const{
        return _->ctx == other._->ctx;
    }

    #define CHECK_CTX_SAME(a, b) assert(a.get_context() == b.get_context());

    void push_ctx(cu_context other){
        CUB_CHECK(cuCtxPushCurrent(other._->ctx));
    }

    cu_context pop_ctx(){
        CUcontext t;
        CUB_CHECK(cuCtxPopCurrent(&t));
        return cu_context(t);
    }

    void set_ctx(){
        CUB_CHECK(cuCtxSetCurrent(_->ctx));
    }

    bool is_valid(){
        return _ && _->ctx;
    }

    size_t get_free_memory_bytes(){
        size_t free;
        CUB_CHECK(cuMemGetInfo(&free, nullptr));
        return free;
    }

    CUcontext _get_handle(){
        return _->ctx;
    }
private:
    cu_context(CUcontext cu_ctx){
        _->ctx = cu_ctx;
    }

    struct Inner{
//        ~Inner(){
//            if(ctx){
//                CUB_CHECK(cuCtxSetCurrent(ctx));
//                CUB_CHECK(cuCtxDestroy(ctx));
//            }
//        }
        CUcontext ctx = nullptr;
    };
    std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

CUB_END