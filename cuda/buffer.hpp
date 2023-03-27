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
            auto get_context() const{
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
            cu_memory_type type;
            cu_context_handle_t ctx;
            template<bool>
            friend class ::cub::cu_buffer;
        };
    }

    template<typename T>
    class buffer_view<T, 1> : public detail::buffer_view_base<T, 1>{
    public:
        buffer_view() = default;

        buffer_view(void* ptr, size_t len)
        {
            this->ptr = ptr;
            this->pitched_info = {0, len, 1};
        }

        CUB_CPU_GPU T& at(size_t x) {
            return reinterpret_cast<T*>(this->ptr)[x];
        }
        // bytes
        CUB_CPU_GPU size_t size() const {
            return this->pitched_info.xsize;
        }

        CUB_CPU_GPU auto sub_view(size_t origin, size_t len) const {
            auto view = *this;
            view->ptr = reinterpret_cast<unsigned char*>(this->ptr) + origin;
            view.pitched_info.xsize = len;
            return view;
        }

    };

    template<typename T>
    class buffer_view<T, 2> : public detail::buffer_view_base<T, 2>{
    public:
        buffer_view() = default;

        buffer_view(void* ptr, const pitched_buffer_info& info){
            this->ptr = ptr;
            this->pitched_info = info;
        }

        CUB_CPU_GPU T& at(size_t x, size_t y) {
            auto row = reinterpret_cast<unsigned char*>(this->ptr) + y * this->pitched_info.pitch;
            return reinterpret_cast<T*>(row)[x];
        }

        CUB_CPU_GPU size_t width() const{
            return this->pitched_info.xsize;
        }

        CUB_CPU_GPU size_t height() const{
            return this->pitched_info.ysize;
        }

    };

    template<typename T>
    class buffer_view<T, 3> : public detail::buffer_view_base<T, 3>{
    public:
        buffer_view() = default;

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


    template<>
    class cu_buffer<false> : public vutil::no_copy_t, vutil::no_heap_t{
    public:
        cu_buffer(size_t size, cu_memory_type type, cu_context_handle_t ctx)
        :ctx(ctx), size(size), type(type)
        {
            assert(ctx->is_valid());
            auto _ = ctx->temp_ctx();

            if(type == e_cu_device){
                CUB_CHECK(cuMemAlloc((CUdeviceptr*)(&ptr), size));
            }
            else if(type == e_cu_host){
                CUB_CHECK(cuMemAllocHost(&ptr, size));
                std::memset(ptr, 0, size);
            }
        }

        ~cu_buffer(){
            if(type == e_cu_device){
                CUB_CHECK(cuMemFree((CUdeviceptr)ptr));
            }
            else if(type == e_cu_host){
                CUB_CHECK(cuMemFreeHost(ptr));
            }
        }
        // bytes
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

        auto get_context() const {
            return ctx;
        }

        cu_memory_type get_type() const {
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
        cu_context_handle_t ctx;
        cu_memory_type type;
    };

    using cu_mem = cu_buffer<false>;

    // must be device
    template<>
    class cu_buffer<true> : public vutil::no_copy_t, vutil::no_heap_t{
    public:
        cu_buffer(size_t width, size_t height, uint32_t ele_size, cu_context_handle_t ctx)
        :ele_size(ele_size), ctx(ctx)
        {
            assert(ctx->is_valid());
            auto _ = ctx->temp_ctx();
            CUB_CHECK(cuMemAllocPitch((CUdeviceptr*)(&ptr), &info.pitch, width * ele_size, height, ele_size));
            info.xsize = width;
            info.ysize = height;
        }
        ~cu_buffer(){
            assert(ctx->is_valid());
            ctx->temp_ctx();
            CUB_CHECK(cuMemFree((CUdeviceptr)ptr));
        }

        template<typename T>
        buffer_view<T,2> view_2d(size_t offset = 0) const {
            assert(offset % info.pitch == 0);
            if(sizeof(T) > ele_size){
                throw std::logic_error("buffer view element size large than pitched memory alloc set");
            }
            auto height = info.ysize;
            auto view = buffer_view<T, 2>(reinterpret_cast<unsigned char*>(ptr) + offset,
                                          {info.pitch, info.xsize, height});
            static_cast<detail::buffer_view_base<T, 2>&>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>&>(view).type = e_cu_device;
            return view;
        }

        template<typename T>
        buffer_view<T,2> view_2d(size_t height, size_t offset = 0) const {
            assert(offset % info.pitch == 0);
            if(sizeof(T) > ele_size){
                throw std::logic_error("buffer view element size large than pitched memory alloc set");
            }
            auto view = buffer_view<T, 2>(reinterpret_cast<unsigned char*>(ptr) + offset,
                                          {info.pitch, info.xsize, height});
            static_cast<detail::buffer_view_base<T, 2>&>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>&>(view).type = e_cu_device;
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
            static_cast<detail::buffer_view_base<T, 3>&>(view).ctx = ctx;
            static_cast<detail::buffer_view_base<T, 2>&>(view).type = e_cu_device;
            return view;
        }

        auto get_type() const{
          return e_cu_device;
        }

    private:
        pitched_buffer_info info;
        size_t ele_size;
        cu_context_handle_t ctx;
        void* ptr;
    };

    using cu_pitched_mem = cu_buffer<true>;

    inline cu_mem cu_context::alloc_buffer(size_t size, cu_memory_type type) {
        return {size, type, this->ref_from_this()};
    }

    inline cu_pitched_mem cu_context::alloc_pitched_buffer(size_t width_bytes, size_t height, uint32_t ele_size) {
        return {width_bytes, height, ele_size, this->ref_from_this()};
    }



CUB_END