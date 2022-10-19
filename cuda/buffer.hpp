#pragma once

#include "context.hpp"

CUB_BEGIN

    template<typename T, int D>
    class buffer_view;

    struct pitched_buffer_info{
        size_t pitch = 0;
        size_t xsize = 0;
        size_t ysize = 0;
    };

    namespace detail{
        template<typename T, int N>
        struct buffer_view_base{
            CUB_CPU_GPU const T* data() const {
                return reinterpret_cast<const T*>(this->ptr);
            }
            CUB_CPU_GPU T* data() {
                return reinterpret_cast<T*>(this->ptr);
            }
            cu_context get_context() const{
                return ctx;
            }
            bool is_device() const{
                return type == e_cu_device;
            }
            size_t pitch() const{
                return pitched_info.pitch;
            }
        protected:
            pitched_buffer_info pitched_info;
            void* ptr = nullptr;
            memory_type type;
            cu_context ctx;
            template<bool>
            friend class ::cub::cu_buffer;
        };
    }

    template<typename T>
    class buffer_view<T, 1> : public detail::buffer_view_base<T, 1>{
    public:
        buffer_view(void* ptr, size_t len)
        {
            this->ptr = ptr;
            this->pitched_info = {0, len, 1};
        }

        CUB_CPU_GPU T& at(size_t x) {
            return reinterpret_cast<T*>(this->ptr)[x];
        }

        CUB_CPU_GPU size_t size() const {
            return this->pitched_info.xsize / sizeof(T);
        }

        CUB_CPU_GPU auto sub_view(size_t origin, size_t len){
            auto view = *this;
            view->ptr = reinterpret_cast<unsigned char*>(this->ptr) + origin;
            view.pitched_info.xsize = len;
            return view;
        }

    };

    template<typename T>
    class buffer_view<T, 2> : public detail::buffer_view_base<T, 2>{
    public:
        buffer_view(void* ptr, const pitched_buffer_info& info){
            this->ptr = ptr;
            this->pitched_info = info;
        }

        CUB_CPU_GPU T& at(size_t x, size_t y) {
            auto row = reinterpret_cast<unsigned char*>(this->ptr) + y * this->pitched_info.pitch;
            return reinterpret_cast<T*>(row)[x];
        }

        CUB_CPU_GPU size_t width() const{
            return this->pitched_info.xsize / sizeof(T);
        }

        CUB_CPU_GPU size_t height() const{
            return this->pitched_info.ysize;
        }

    };

    template<typename T>
    class buffer_view<T, 3> : public detail::buffer_view_base<T, 3>{
    public:
        buffer_view(void* ptr, const pitched_buffer_info& info, const cu_extent& extent)
        : extent(extent)
        {
            this->ptr = ptr;
            this->pitched_info = info;
        }

        CUB_CPU_GPU T& at(size_t x, size_t y, size_t z) {
            auto row = reinterpret_cast<unsigned char*>(this->ptr) + y * this->pitched_info.pitch
                    + z * this->pitched_info.pitch * extent.height;
            return reinterpret_cast<T*>(row)[x];
        }

        CUB_CPU_GPU size_t width() const{
            return extent.width;
        }
        CUB_CPU_GPU size_t height() const{
            return extent.height;
        }
        CUB_CPU_GPU size_t depth() const{
            return extent.depth;
        }
    private:
        cu_extent extent;
    };

    template<bool Pitched>
    class cu_buffer;

    //TODO move and copy construct
    template<>
    class cu_buffer<false>{
    public:
        cu_buffer(size_t size, memory_type type, cu_context ctx)
        :ctx(ctx), size(size), type(type)
        {
            ctx.set_ctx();
//            CUcontext t;
//            CUB_CHECK(cuCtxGetCurrent(&t));

            if(type == e_cu_device){
                CUB_CHECK(cuMemAlloc((CUdeviceptr*)(&ptr), size));
            }
            else if(type == e_cu_host){
                CUB_CHECK(cuMemAllocHost(&ptr, size));
                CUB_CHECK(cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE));
            }
        }

        ~cu_buffer(){
            CTX_SCOPE_SET
            if(type == e_cu_device){
                CUB_CHECK(cuMemFree((CUdeviceptr)ptr));
            }
            else if(type == e_cu_host){
                CUB_CHECK(cuMemHostUnregister(ptr));
                CUB_CHECK(cuMemFreeHost(ptr));
            }
        }

        template<typename T>
        buffer_view<T,1> view_1d(size_t len, size_t offset = 0) const {
            auto view = buffer_view<T,1>(reinterpret_cast<uint8_t*>(ptr) + offset, len);
            static_cast<detail::buffer_view_base<T, 1>&>(view).type = type;
            static_cast<detail::buffer_view_base<T, 1>&>(view).ctx = ctx;

            return view;
        }

        template<typename T>
        buffer_view<T,2> view_2d(const pitched_buffer_info& info, size_t offset = 0) const {
            assert(info.pitch >= info.xsize);
            auto view = buffer_view<T, 2>(reinterpret_cast<uint8_t*>(ptr) + offset,info);
            static_cast<detail::buffer_view_base<T, 2>&>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>&>(view).type = type;
            return view;
        }

        template<typename T>
        buffer_view<T,3> view_3d(const pitched_buffer_info& info, const cu_extent& extent, size_t offset = 0) const {
            assert(info.pitch >= info.xsize);
            auto view = buffer_view<T, 3>(reinterpret_cast<uint8_t*>(ptr) + offset,info, extent);
            static_cast<detail::buffer_view_base<T, 3>&>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 3>&>(view).type = type;
            return view;
        }

        cu_context get_context() const {
            return ctx;
        }

        memory_type get_type() const {
            return type;
        }

        size_t get_size() const {
            return size;
        }

        void* get_data() const {
            return ptr;
        }
    private:
        void* ptr;
        size_t size;
        cu_context ctx;
        memory_type type;
    };

    // must be device
    template<>
    class cu_buffer<true>{
    public:
        cu_buffer(size_t width_bytes, size_t height, uint32_t ele_size, cu_context ctx)
        :ele_size(ele_size), ctx(ctx)
        {
            ctx.set_ctx();
            CUB_CHECK(cuMemAllocPitch((CUdeviceptr*)(&ptr), &info.pitch, width_bytes, height, ele_size));
            info.xsize = width_bytes;
            info.ysize = height;
        }
        ~cu_buffer(){
            CTX_SCOPE_SET

            CUB_CHECK(cuMemFree((CUdeviceptr)ptr));

        }

        template<typename T>
        buffer_view<T,2> view_2d(size_t height, size_t offset = 0) const {
            assert(offset % info.pitch == 0);
            if(sizeof(T) > ele_size){
                throw std::logic_error("buffer view element size large than pitched memory alloc set");
            }
            auto view = buffer_view<T, 2>(reinterpret_cast<unsigned char*>(ptr) + offset,
                                          {info.pitch, info.xsize, height});
            static_cast<detail::buffer_view_base<T, 2>>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>>(view).type = e_cu_device;
            return view;
        }

        template<typename T>
        buffer_view<T,3> view_3d(size_t height, size_t depth, size_t offset = 0) const {
            assert(offset % info.pitch == 0);
            if(sizeof(T) > ele_size){
                throw std::logic_error("buffer view element size large than pitched memory alloc set");
            }
            auto view = buffer_view<T, 3>(reinterpret_cast<unsigned char*>(ptr) + offset,
                                         {info.pitch, info.xsize, height * depth},
                                         {info.xsize / sizeof(T), height, depth});
            static_cast<detail::buffer_view_base<T, 3>>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>>(view).type = e_cu_device;
            return view;
        }

        memory_type get_type() const{
          return e_cu_device;
        }

    private:
        pitched_buffer_info info;
        size_t ele_size;
        cu_context ctx;
        void* ptr;
    };

    inline cu_buffer<false> cu_context::alloc_buffer(size_t size, memory_type type) {
        return cu_buffer<false>(size, type, *this);
    }

    inline cu_buffer<true> cu_context::alloc_buffer_pitched(size_t width_bytes, size_t height, uint32_t ele_size) {
        return cu_buffer<true>(width_bytes, height, ele_size, *this);
    }



CUB_END