#pragma once

#include "common.hpp"
#include <misc.hpp>

CUB_BEGIN

namespace detail
{
struct cu_init
{
    cu_init()
    {
        static auto _ = [] {
            try
            {
                CUB_CHECK(cuInit(0));
                CUB_WHEN_DEBUG(std::cout << "CUDA init successfully..." << std::endl)
            }
            catch (const std::exception &err)
            {
                std::cerr << err.what() << std::endl;
                exit(0);
            }
            return 0;
        }();
    }
};
inline cu_init _cu_init = cu_init();
}

struct cu_context_deleter{
    constexpr cu_context_deleter() noexcept = default;

    void operator()(cu_context* p) const noexcept;
};

using cu_context_handle_t = vutil::intrusive_ptr_t<cu_context>;

class cu_context
    : public vutil::intrusive_ptr_enabled_t<cu_context,
                                            cu_context_deleter,
                                            vutil::multi_thread_counter>
{
public:
    cu_context() = default;

    cu_context(cu_physical_device device, uint32_t flags);

    /**
     * @brief 分配host或者device buffer
     */
    cu_buffer<false> alloc_buffer(size_t size, cu_memory_type type);

    /**
     * @brief 分配device上pitched buffer
     */
    cu_buffer<true> alloc_pitched_buffer(size_t width_bytes, size_t height, uint32_t ele_size);

    template<typename T, int N, typename... Args>
    cu_array<T, N> alloc_array(Args&&... args);

    /**
     * @brief 分配包括了array的texture资源
     */
    cu_texture_wrap alloc_texture(const texture_resc_info& resc_info, const texture_view_info& view_info);

    bool operator==(const cu_context& other) const{
        return ctx == other.ctx;
    }

    #define CHECK_CTX_SAME(a, b) assert(a.get_context() == b.get_context());


    void push_ctx() const {
        CUB_CHECK(cuCtxPushCurrent(ctx));
    }

    void pop_ctx() const {
        CUB_CHECK(cuCtxPopCurrent(nullptr));
    }

    auto temp_ctx() const {
        push_ctx();
        return vutil::scope_bomb_t([this]{pop_ctx();});
    }

    void set_ctx() const {
        CUB_CHECK(cuCtxSetCurrent(ctx));
    }

    bool is_valid() const {
        return ctx;
    }

    size_t get_free_memory_bytes() const {
        auto _ = temp_ctx();
        size_t free;
        CUB_CHECK(cuMemGetInfo(&free, nullptr));
        return free;
    }

    auto _get_naive_ctx() const {
        return ctx;
    }

private:
    CUcontext ctx{nullptr};
    int device_id{-1};
};

inline void cu_context_deleter::operator()(cu_context *p) const noexcept
{
    if(auto ctx = p->_get_naive_ctx()){
        CUB_CHECK(cuCtxDestroy(ctx));
    }
}

CUB_END