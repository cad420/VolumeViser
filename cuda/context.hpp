#pragma once

#include "common.hpp"

CUB_BEGIN

    enum memory_type{
        e_cu_host,
        e_cu_device
    };

    class cu_context{

    public:
        cu_context() = default;

        cu_context(cu_physical_device device, uint32_t flags);

        cu_kernel create_kernel();


        // alloc cuda device memory or host(pinned) memory
        cu_buffer<false> alloc_buffer(size_t size, memory_type type);

        // alloc cuda device pitched memory
        cu_buffer<true> alloc_buffer_pitched(size_t width_bytes, size_t height, uint32_t ele_size);


        template<typename T, int N, typename... Args>
        cu_array<T, N> alloc_array(Args&&... args) const;

        bool operator==(const cu_context& other) const{
            return ctx == other.ctx;
        }

        #define CHECK_CTX_SAME(a, b) assert(a.get_context() == b.get_context());

        struct Binder{
            Binder(std::function<void()> f):f(std::move(f)){}
            ~Binder(){
                if(f)
                    f();
            }
        private:
            std::function<void()> f;
        };

        auto get(){
            push_ctx();
            return Binder([this](){
                this->pop_ctx();
            });
        }

        #define CTX_SCOPE_SET auto binder = ctx.get();

        #define GET_CTX_SCOPE_SET(obj) auto bindier = obj.get_context().get();

        void push_ctx(){

            CUB_CHECK(cuCtxPushCurrent(ctx));
        }


        void pop_ctx(){

            CUcontext t;
            CUB_CHECK(cuCtxPopCurrent(&t));
            if(t != ctx){
                throw std::logic_error("cu_context popped context not right");
            }
        }

        void set_ctx(){

            CUB_CHECK(cuCtxSetCurrent(ctx));
        }


    private:
        CUcontext ctx = nullptr;
    };







CUB_END