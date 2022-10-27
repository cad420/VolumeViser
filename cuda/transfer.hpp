#pragma once

#include "buffer.hpp"
#include "array.hpp"
#include "stream.hpp"
CUB_BEGIN

// NOTE: memory copy must in same cuda context except copy from/to cpu memory

// 很具体繁琐的信息... 失败的设计
struct memory_transfer_info{
    uint32_t src_x_bytes = 0;
    uint32_t src_y = 0;
    uint32_t src_z = 0;
    uint32_t dst_x_bytes = 0;
    uint32_t dst_y = 0;
    uint32_t dst_z = 0;
    uint32_t width_bytes = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
};

namespace detail {
    template<typename T, int N>
    struct cu_buffer_transfer;

    template<typename T>
    struct cu_buffer_transfer<T, 1> {
        static cu_task transfer(const buffer_view<T, 1> &src, const buffer_view<T, 1> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                auto src_p = reinterpret_cast<unsigned char *>(src.data()) + info.src_x_bytes;
                auto dst_p = reinterpret_cast<unsigned char *>(dst.data()) + info.dst_x_bytes;
                CUB_CHECK(cuMemcpyAsync((CUdeviceptr) dst_p, (CUdeviceptr) src_p, info.width_bytes, stream._->stream));
            }};
        }

        static cu_task transfer(const buffer_view<T, 1> &src, const cu_array<T, 1> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                auto src_p = reinterpret_cast<unsigned char *>(src.data()) + info.src_x_bytes;
                if (src.is_device()) {
                    CUB_CHECK(cuMemcpyDtoA(dst.get_handle(), info.dst_x_bytes, (CUdeviceptr) src_p, info.width_bytes));
                } else {
                    CUB_CHECK(cuMemcpyHtoAAsync(dst.get_handle(), info.dst_x_bytes, src_p, info.width_bytes,
                                                stream._->stream));
                }
            }};
        }
    };

    template<typename T>
    struct cu_buffer_transfer<T, 2> {
        static cu_task transfer(const buffer_view<T, 2> &src, const buffer_view<T, 2> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY2D m;
                std::memset(&m, 0, sizeof(m));
                if (src.is_device()) {
                    m.srcDevice = (CUdeviceptr) src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.srcPitch = src.pitch();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;

                if (dst.is_device()) {
                    m.srcDevice = (CUdeviceptr) dst.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = dst.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.dstPitch = dst.pitch();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;

                CUB_CHECK(cuMemcpy2DAsync(&m, stream._->stream));
            }};
        }

        static cu_task transfer(const buffer_view<T, 2> &src, const cu_array<T, 2> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY2D m;
                std::memset(&m, 0, sizeof(m));
                if (src.is_device()) {
                    m.srcDevice = (CUdeviceptr) src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.srcPitch = src.pitch();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;

                m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                m.dstArray = dst.get_handle();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;

                CUB_CHECK(cuMemcpy2DAsync(&m, stream._->stream));
            }};
        }
    };

    template<typename T>
    struct cu_buffer_transfer<T, 3> {
        static cu_task transfer(const buffer_view<T, 3> &src, const buffer_view<T, 3> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY3D m;
                std::memset(&m, 0, sizeof(m));
                if (src.is_device()) {
                    m.srcDevice = (CUdeviceptr) src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;
                m.srcZ = info.src_z;
                m.srcPitch = src.pitch();
                m.srcHeight = src.height();

                if (dst.is_device()) {
                    m.dstDevice = (CUdeviceptr) dst.data();
                    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.dstHost = dst.data();
                    m.dstMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;
                m.dstZ = info.dst_z;
                m.dstPitch = dst.pitch();
                m.dstHeight = dst.height();

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;
                m.Depth = info.depth;

                CUB_CHECK(cuMemcpy3DAsync(&m, stream._->stream));
            }};
        }

        static cu_task transfer(const buffer_view<T, 3> &src, const cu_array<T, 3> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY3D m;
                std::memset(&m, 0, sizeof(m));
                if (src.is_device()) {
                    m.srcDevice = (CUdeviceptr) src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = src.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;
                m.srcZ = info.src_z;
                m.srcPitch = src.pitch();
                m.srcHeight = src.height();

                m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                m.dstArray = dst.get_handle();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;
                m.dstZ = info.dst_z;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;
                m.Depth = info.depth;

                CUB_CHECK(cuMemcpy3DAsync(&m, stream._->stream));
            }};
        }
    };

    template<typename T, int N>
    struct cu_array_transfer;

    template<typename T>
    struct cu_array_transfer<T, 1> {
        static cu_task transfer(const cu_array<T, 1> &src, const buffer_view<T, 1> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                auto dst_p = reinterpret_cast<unsigned char *>(dst.data()) + info.dst_x_bytes;
                if (dst.is_device()) {
                    CUB_CHECK(cuMemcpyAtoD(dst_p, src.get_handle(), info.src_x_bytes, info.width_bytes));
                } else {
                    CUB_CHECK(cuMemcpyAtoHAsync(dst_p, src.get_handle(), info.src_x_bytes, info.width_bytes,
                                                stream._->stream));
                }
            }};
        }

        static cu_task transfer(const cu_array<T, 1> &src, const cu_array<T, 1> &dst, const memory_transfer_info &info) {
            return [=](cu_stream &stream) {
                CUB_CHECK(cuMemcpyAtoA(dst.get_handle(), info.dst_x_bytes, src.get_handle(), info.src_x_bytes,
                                       info.width_bytes));
            };
        }
    };

    template<typename T>
    struct cu_array_transfer<T, 2> {
        static cu_task transfer(const cu_array<T, 2> &src, const buffer_view<T, 2> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY2D m;
                std::memset(&m, 0, sizeof(m));

                m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                m.srcArray = dst.get_handle();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;

                if (dst.is_device()) {
                    m.srcDevice = (CUdeviceptr) dst.data();
                    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.srcHost = dst.data();
                    m.srcMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.dstPitch = dst.pitch();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;

                CUB_CHECK(cuMemcpy2DAsync(&m, stream._->stream));
            }};
        }

        static cu_task transfer(const cu_array<T, 2> &src, const cu_array<T, 2> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY2D m;
                std::memset(&m, 0, sizeof(m));

                m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                m.srcArray = src.get_handle();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;

                m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                m.dstArray = dst.get_handle();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;

                CUB_CHECK(cuMemcpy2DAsync(&m, stream._->stream));
            }};
        }
    };

    template<typename T>
    struct cu_array_transfer<T, 3> {
        static cu_task transfer(const cu_array<T, 3> &src, const buffer_view<T, 3> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY3D m;
                std::memset(&m, 0, sizeof(m));
                m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                m.srcArray = src.get_handle();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;
                m.srcZ = info.src_z;

                m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                m.dstArray = dst.get_handle();
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;
                m.dstZ = info.dst_z;

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;
                m.Depth = info.depth;

                CUB_CHECK(cuMemcpy3DAsync(&m, stream._->stream));
            }};
        }

        static cu_task transfer(const cu_array<T, 3> &src, const cu_array<T, 3> &dst, const memory_transfer_info &info) {
            return {[=](cu_stream &stream) {
                CUDA_MEMCPY3D m;
                std::memset(&m, 0, sizeof(m));
                m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                m.srcArray = src.get_handle();
                m.srcXInBytes = info.src_x_bytes;
                m.srcY = info.src_y;
                m.srcZ = info.src_z;

                if (dst.is_device()) {
                    m.dstDevice = (CUdeviceptr) dst.data();
                    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                } else {
                    m.dstHost = dst.data();
                    m.dstMemoryType = CU_MEMORYTYPE_HOST;
                }
                m.dstXInBytes = info.dst_x_bytes;
                m.dstY = info.dst_y;
                m.dstZ = info.dst_z;
                m.dstPitch = dst.pitch();
                m.dstHeight = dst.height();

                m.WidthInBytes = info.width_bytes;
                m.Height = info.height;
                m.Depth = info.depth;

                CUB_CHECK(cuMemcpy3DAsync(&m, stream._->stream));
            }};
        }
    };
}

template<typename T, int N>
cu_task cu_memory_transfer(const cu_array<T, N>& src, const buffer_view<T, N>& dst, const memory_transfer_info& info){
    CHECK_CTX_SAME(src, dst)
    return detail::cu_array_transfer<T, N>::transfer(src, dst, info);
}

template<typename T, int N>
cu_task cu_memory_transfer(const cu_array<T, N>& src, const cu_array<T, N>& dst, const memory_transfer_info& info){
    CHECK_CTX_SAME(src, dst)
    return detail::cu_array_transfer<T, N>::transfer(src, dst, info);
}

template<typename T, int N>
cu_task cu_memory_transfer(const buffer_view<T, N>& src, const buffer_view<T, N>& dst, const memory_transfer_info& info){
    CHECK_CTX_SAME(src, dst)
    return detail::cu_buffer_transfer<T, N>::transfer(src, dst, info);
}

template<typename T, int N>
cu_task cu_memory_transfer(const buffer_view<T, N>& src, const cu_array<T, N>& dst, const memory_transfer_info& info){
    CHECK_CTX_SAME(src, dst)
    return detail::cu_buffer_transfer<T, N>::transfer(src, dst, info);
}


inline cu_task cu_memory_transfer(const cu_buffer<false>& src, const cu_texture_wrap& dst, const memory_transfer_info& info){
    CHECK_CTX_SAME(src, dst)
    return cu_task{[&](cu_stream& stream){
        CUDA_MEMCPY3D m;
        std::memset(&m, 0, sizeof(m));

        if(src.get_type() == e_cu_host)
            m.srcMemoryType = CU_MEMORYTYPE_HOST;
        else
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcXInBytes = info.src_x_bytes;
        m.srcY = info.src_y;
        m.srcZ = info.src_z;
        m.srcHost = src.get_data();

        m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        m.dstArray = (CUarray)dst._get_array_handle();
        m.dstXInBytes = info.dst_x_bytes;
        m.dstY = info.dst_y;
        m.dstZ = info.dst_z;

        m.WidthInBytes = info.width_bytes;
        m.Height = info.height;
        m.Depth = info.depth;

        CUB_CHECK(cuMemcpy3DAsync(&m, stream.lock().get()));
    }};
}

CUB_END