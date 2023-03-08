#pragma once

#include "array.hpp"

CUB_BEGIN

enum address_mode {
    e_wrap   = 0,
    e_clamp  = 1,
    e_mirror = 2,
    e_border = 3
};

enum filter_mode {
    e_nearest = 0,
    e_linear  = 1
};

enum read_mode{
    e_raw              = 0,
    e_normalized_float = 1
};

enum tex_format{
    e_uint8,
    e_uint16,
    e_float
};

struct texture_resc_info{
    tex_format format = e_uint8;
    uint32_t channels = 1;
    cu_extent extent;
    bool read_write = false;

    size_t alloc_bytes() const{
        size_t bytes = 0;
        switch (format) {
            case e_uint8 : bytes = 1; break;
            case e_uint16: bytes = 2; break;
            case e_float : bytes = 4; break;
        }
        bytes *= channels;
        bytes *= extent.width * extent.height * extent.depth;
        return bytes;
    }
};

struct texture_view_info{
    address_mode address = e_border;
    filter_mode  filter  = e_linear;
    read_mode    read    = e_normalized_float;
    bool normalized_coords = true;
};

inline int get_fmt_bits(tex_format fmt) {
    int bits = 0;
    switch (fmt) {
        case e_uint8 : bits = 8 ; break;
        case e_uint16: bits = 16; break;
        case e_float : bits = 32; break;
    }
    return bits;
}

inline bool is_float(tex_format fmt) {
    return fmt == e_float;
}

class cu_texture_wrap{
public:
    cu_texture_wrap(const texture_resc_info& resc_info, const texture_view_info& view_info, cu_context_handle_t ctx){
        assert(ctx->is_valid());
        this->ctx = ctx;
        auto _ = ctx->temp_ctx();

        cudaChannelFormatDesc desc;
        std::memset(&desc, 0, sizeof(desc));
        int* p = reinterpret_cast<int*>(&desc);
        int bits = get_fmt_bits(resc_info.format);
        for(int i = 0; i < resc_info.channels; ++i)
            p[i] = bits;
        if(is_float(resc_info.format)){
            desc.f = cudaChannelFormatKindFloat;
        }
        else{
            desc.f = cudaChannelFormatKindUnsigned;
        }
        CUB_CHECK(cudaMalloc3DArray(&array, &desc, {resc_info.extent.width, resc_info.extent.height, resc_info.extent.depth},
                                    resc_info.read_write ? cudaArraySurfaceLoadStore : 0));

        cudaResourceDesc resc_desc;
        resc_desc.resType = cudaResourceTypeArray;
        resc_desc.res.array.array = array;

        cudaTextureDesc tex_desc;
        std::memset(&tex_desc, 0, sizeof(tex_desc));
        switch (view_info.address) {
            case e_wrap   : tex_desc.addressMode[0] = cudaAddressModeWrap  ; break;
            case e_clamp  : tex_desc.addressMode[0] = cudaAddressModeClamp ; break;
            case e_mirror : tex_desc.addressMode[0] = cudaAddressModeMirror; break;
            case e_border : tex_desc.addressMode[0] = cudaAddressModeBorder; break;
        }
        tex_desc.addressMode[1] = tex_desc.addressMode[2] = tex_desc.addressMode[0];
        switch (view_info.filter) {
            case e_nearest : tex_desc.filterMode = cudaFilterModePoint ; break;
            case e_linear  : tex_desc.filterMode = cudaFilterModeLinear; break;
        }
        switch(view_info.read){
            case e_raw              : tex_desc.readMode = cudaReadModeElementType    ; break;
            case e_normalized_float : tex_desc.readMode = cudaReadModeNormalizedFloat; break;
        }
        tex_desc.normalizedCoords = view_info.normalized_coords;
        CUB_CHECK(cudaCreateTextureObject(&tex, &resc_desc, &tex_desc, nullptr));

    }

    auto get_context() const {
        return ctx;
    }

    auto _get_tex_handle() const{
        return tex;
    }

    auto _get_array_handle() const{
        return array;
    }

    cudaTextureObject_t view_as(const texture_view_info& view_info) const {
        auto _ = ctx->temp_ctx();

        cudaTextureObject_t ret;
        cudaResourceDesc resc_desc;
        resc_desc.resType = cudaResourceTypeArray;
        resc_desc.res.array.array = array;

        cudaTextureDesc tex_desc;
        std::memset(&tex_desc, 0, sizeof(tex_desc));
        switch (view_info.address) {
            case e_wrap   : tex_desc.addressMode[0] = cudaAddressModeWrap  ; break;
            case e_clamp  : tex_desc.addressMode[0] = cudaAddressModeClamp ; break;
            case e_mirror : tex_desc.addressMode[0] = cudaAddressModeMirror; break;
            case e_border : tex_desc.addressMode[0] = cudaAddressModeBorder; break;
        }
        tex_desc.addressMode[1] = tex_desc.addressMode[2] = tex_desc.addressMode[0];
        switch (view_info.filter) {
            case e_nearest : tex_desc.filterMode = cudaFilterModePoint ; break;
            case e_linear  : tex_desc.filterMode = cudaFilterModeLinear; break;
        }
        switch(view_info.read){
            case e_raw              : tex_desc.readMode = cudaReadModeElementType    ; break;
            case e_normalized_float : tex_desc.readMode = cudaReadModeNormalizedFloat; break;
        }
        tex_desc.normalizedCoords = view_info.normalized_coords;
        CUB_CHECK(cudaCreateTextureObject(&ret, &resc_desc, &tex_desc, nullptr));
        return ret;
    }

    cudaSurfaceObject_t as_surface() const {
        auto _ = ctx->temp_ctx();

        unsigned int flags = 0;
        CUB_CHECK(cudaArrayGetInfo(nullptr, nullptr, &flags, array));
        if((flags & cudaArraySurfaceLoadStore) == 0){
            std::cerr << "cuda texture as surface but create without load/store flag" << std::endl;
            assert(false);
            return 0;
        }
        cudaSurfaceObject_t surf;
        cudaResourceDesc resc_desc;
        resc_desc.resType = cudaResourceTypeArray;
        resc_desc.res.array.array = array;
        CUB_CHECK(cudaCreateSurfaceObject(&surf, &resc_desc));
        return surf;
    }

protected:
    cudaArray_t array;
    cudaTextureObject_t tex;
    cu_context_handle_t ctx;

};

inline cu_texture_wrap cu_context::alloc_texture(const texture_resc_info &resc_info, const texture_view_info &view_info) {
    return cu_texture_wrap(resc_info, view_info, this->ref_from_this());
}

class cu_surface{
  public:
    cu_surface(cudaArray_t array)
    {
        cudaResourceDesc desc;
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = array;
        CUB_CHECK(cudaCreateSurfaceObject(&surface, &desc));
    }

    ~cu_surface(){
        CUB_CHECK(cudaDestroySurfaceObject(surface));
    }

    auto _get_handle() const noexcept {
        return surface;
    }
  private:
    cudaSurfaceObject_t surface;
};

class [[deprecated]] cu_texture{
public:
    template<typename E, int N>
    cu_texture(const cu_array<E, N>& array, const texture_view_info& info){

        CUDA_RESOURCE_DESC resc_desc{CU_RESOURCE_TYPE_ARRAY};
        resc_desc.res.array.hArray = array.get_handle();

        CUDA_TEXTURE_DESC tex_desc;
        std::memset(&tex_desc, 0, sizeof(tex_desc));
        switch (info.address) {
            case e_wrap   : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP  ; break;
            case e_clamp  : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP ; break;
            case e_mirror : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_MIRROR; break;
            case e_border : tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER; break;
        }
        tex_desc.addressMode[1] = tex_desc.addressMode[2] = tex_desc.addressMode[0];
        switch (info.filter) {
            case e_nearest : tex_desc.filterMode = CU_TR_FILTER_MODE_POINT ; break;
            case e_linear  : tex_desc.filterMode = CU_TR_FILTER_MODE_LINEAR; break;
        }
        if(info.normalized_coords)
            tex_desc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
        if(info.read == e_raw)
            tex_desc.flags |= CU_TRSF_READ_AS_INTEGER;

        CUB_CHECK(cuTexObjectCreate(&tex, &resc_desc, &tex_desc, nullptr));

        f = [ctx = array.get_ctx(), this](){
            ctx.set_ctx();
            CUB_CHECK(cuTexObjectDestroy(tex));
        };
    }

    CUB_CPU_GPU auto get_handle() const {
        return tex;
    }

    ~cu_texture(){
        if(f)
            f();
    }

private:
    CUtexObject tex;
    std::function<void()> f;
};

CUB_END