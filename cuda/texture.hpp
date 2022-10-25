#pragma once

#include "array.hpp"


CUB_BEGIN

//同时存储array和texture对象
class cu_texture_wrap{
public:
    enum address_mode {
        e_wrap = 0,
        e_clamp = 1,
        e_mirror = 2,
        e_border = 3
    };

    enum filter_mode {
        e_nearest = 0,
        e_linear = 1
    };

    enum read_mode{
        e_raw = 0,
        e_normalized_float = 1
    };

    enum tex_format{
        e_uint8,
        e_uint16,
        e_float
    };

    struct texture_info{
        tex_format format = e_uint8;
        uint32_t channels = 1;
        address_mode address = e_wrap;
        filter_mode filter = e_linear;
        read_mode read = e_normalized_float;
        cu_extent extent;
        cu_array3d_type type;
        bool normalized_coords = true;

        size_t alloc_bytes() const{
            size_t bytes = 0;
            switch (format) {
                case e_uint8: bytes = 1; break;
                case e_uint16: bytes = 2; break;
                case e_float: bytes = 4; break;
            }
            bytes *= channels;
            bytes *= extent.width * extent.height * extent.depth;
            return bytes;
        }
    };

    cu_texture_wrap(const texture_info& info, cu_context ctx){

    }
    cu_context get_context() const {
        return ctx;
    }
protected:

    cudaArray_t array;
    cudaTextureObject_t tex;
    cu_context ctx;

};

class cu_texture{
public:
    enum address_mode {
        e_wrap = 0,
        e_clamp = 1,
        e_mirror = 2,
        e_border = 3
    };

    enum filter_mode {
        e_nearest = 0,
        e_linear = 1
    };

    enum read_mode{
        e_raw = 0,
        e_normalized_float = 1
    };

    struct texture_info{
        address_mode address = e_wrap;
        filter_mode filter = e_linear;
        read_mode read = e_normalized_float;
        bool normalized_coords = true;
    };

    static texture_info as_array(){
        return {
            e_wrap, e_nearest, e_raw, false
        };
    }

    template<typename E, int N>
    cu_texture(const cu_array<E, N>& array, const texture_info& info, cu_context ctx){

        GET_CTX_SCOPE_SET(array)

        CUDA_RESOURCE_DESC resc_desc{CU_RESOURCE_TYPE_ARRAY};
        resc_desc.res.array.hArray = array.get_handle();

        CUDA_TEXTURE_DESC tex_desc;
        std::memset(&tex_desc, 0, sizeof(tex_desc));
        switch (info.address) {
            case e_wrap : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP; break;
            case e_clamp : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; break;
            case e_mirror : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_MIRROR; break;
            case e_border : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER; break;
        }
        tex_desc.addressMode[1] = tex_desc.addressMode[2] = tex_desc.addressMode[0];
        switch (info.filter) {
            case e_nearest : tex_desc.filterMode = CU_TR_FILTER_MODE_POINT; break;
            case e_linear : tex_desc.filterMode = CU_TR_FILTER_MODE_LINEAR; break;
        }
        if(info.normalized_coords)
            tex_desc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
        if(info.read == e_raw)
            tex_desc.flags |= CU_TRSF_READ_AS_INTEGER;

        CUB_CHECK(cuTexObjectCreate(&tex, &resc_desc, &tex_desc, nullptr));

    }

    CUB_CPU_GPU auto get_handle() const {
        return tex;
    }

    ~cu_texture(){
        CUB_CHECK(cuTexObjectDestroy(tex));
    }

private:
    CUtexObject tex;
};


CUB_END