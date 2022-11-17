#pragma once

#include "../math.hpp"

VUTIL_BEGIN

    std::vector<byte> load_bytes_from_file(const std::string& filename,int* width,int* height,int* ncomp);

    tensor_t<byte,2> load_gray_from_file(const std::string& filename);

    tensor_t<color2b,2> load_gray_alpha_from_file(const std::string& filename);

    tensor_t<color3b,2> load_rgb_from_file(const std::string& filename);

    tensor_t<color4b,2> load_rgba_from_file(const std::string& filename);

    tensor_t<color3f,2> load_rgb_from_hdr_file(const std::string& filename);

    std::vector<byte> load_bytes_from_memory_file(const void* data,size_t bytes_count,int* width,int* height,int* comp);

    tensor_t<byte,2> load_gray_from_memory_file(const void* data,size_t bytes_count);

    tensor_t<color2b,2> load_gray_alpha_from_memory_file(const void* data,size_t bytes_count);

    tensor_t<color3b,2> load_rgb_from_memory_file(const void* data,size_t bytes_count);

    tensor_t<color4b,2> load_rgba_from_memory_file(const void* data,size_t bytes_count);

    tensor_t<color3f,2> load_rgb_from_hdr_memory_file(const void* data,size_t bytes_count);


VUTIL_END