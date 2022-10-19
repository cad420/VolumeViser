#include "save_image.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image_write.h"

VUTIL_BEGIN

    void save_rgb_to_png_file( const std::string &filename, const tensor_t<color3b, 2> &data )
    {
        stbi_write_png( filename.c_str(), data.shape()[ 0 ], data.shape()[ 1 ], 3, data.raw_data(), 0 );
    }

VUTIL_END