#pragma once

#include "array.hpp"


CUB_BEGIN

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
    cu_texture(const cu_array<E, N>& array, const texture_info& info){

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