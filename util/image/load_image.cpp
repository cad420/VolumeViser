#include "load_image.hpp"
#include "file/raw_file_io.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

VUTIL_BEGIN

    std::vector<byte> load_bytes_from_file(const std::string& filename,int* width,int* height,int* ncomp){
        auto bytes = read_raw_file_bytes(filename);
        return load_bytes_from_memory_file(bytes.data(),bytes.size(),width,height,ncomp);
    }

    tensor_t<byte,2> load_gray_from_file(const std::string& filename){
        auto bytes = read_raw_file_bytes(filename);
        return load_gray_from_memory_file(bytes.data(),bytes.size());
    }

    tensor_t<color2b,2> load_gray_alpha_from_file(const std::string& filename){
        auto bytes = read_raw_file_bytes(filename);
        return load_gray_alpha_from_memory_file(bytes.data(),bytes.size());
    }

    tensor_t<color3b,2> load_rgb_from_file(const std::string& filename){
        auto bytes = read_raw_file_bytes(filename);
        return load_rgb_from_memory_file(bytes.data(),bytes.size());
    }

    tensor_t<color4b,2> load_rgba_from_file(const std::string& filename){
        auto bytes = read_raw_file_bytes(filename);
        return load_rgba_from_memory_file(bytes.data(),bytes.size());
    }

    tensor_t<color3f,2> load_rgb_from_hdr_file(const std::string& filename){
        auto bytes = read_raw_file_bytes(filename);
        return load_rgb_from_hdr_memory_file(bytes.data(),bytes.size());
    }

    std::vector<byte> load_bytes_from_memory_file(const void* data,size_t bytes_count,int* width,int* height,int* comp){
        if(!data) return {};
        assert(width && height && comp);
        auto bytes = stbi_load_from_memory(static_cast<const stbi_uc*>(data),bytes_count,width,height,comp,0);
        if(!bytes) return {};
        std::vector<byte> ret(bytes,bytes + (*width) * (*height) * (*comp));
        stbi_image_free(bytes);
        return ret;
    }

    tensor_t<byte,2> load_gray_from_memory_file(const void* data,size_t bytes_count){
        int w,h,comp;
        auto bytes = stbi_load_from_memory(static_cast<const stbi_uc*>(data),bytes_count,&w,&h,&comp,STBI_grey);
        if(!bytes) return {};
        auto ret = tensor_t<byte,2>::from_linear_array({w,h},static_cast<const byte*>(data));
        stbi_image_free(bytes);
        return ret;
    }
    template <typename T,int C,int STBchannel>
    static tensor_t<T,2> load_from_memory(const void* data,size_t bytes_count){
        int w,h,comp;
        auto bytes = stbi_load_from_memory(static_cast<const stbi_uc*>(data),bytes_count,&w,&h,&comp,STBchannel);
        if(!bytes) return {};
        auto image = tensor_t<T,2>::from_linear_index_func({w,h},[&](int i){
            T ret;
            for(int j = 0; j < C; ++j)
                ret[j] = bytes[i * C + j];
            return ret;
        });
        stbi_image_free(bytes);
        return image;
    }
    tensor_t<color2b,2> load_gray_alpha_from_memory_file(const void* data,size_t bytes_count){
        return load_from_memory<color2b,2,STBI_grey_alpha>(data,bytes_count);
    }

    tensor_t<color3b,2> load_rgb_from_memory_file(const void* data,size_t bytes_count){
        return load_from_memory<color3b,3,STBI_rgb>(data,bytes_count);
    }

    tensor_t<color4b,2> load_rgba_from_memory_file(const void* data,size_t bytes_count){
        return load_from_memory<color4b,4,STBI_rgb_alpha>(data,bytes_count);
    }

    tensor_t<color3f,2> load_rgb_from_hdr_memory_file(const void* data,size_t bytes_count){
        int w, h, channels;
        float *d = stbi_loadf_from_memory(static_cast<const stbi_uc *>(data),bytes_count,&w, &h, &channels, STBI_rgb);
        if(!d)
            return tensor_t<color3f,2>();


        auto ret =  tensor_t<color3f,2>::from_linear_index_func({w,h},[&](int i){
            color3f ret;
            for(int j = 0; j < 3; ++j)
                ret[j] = d[i * 3 + j];
            return ret;
        });
        stbi_image_free(d);
        return ret;
    }

VUTIL_END