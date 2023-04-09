#pragma once

#include "../math.hpp"

VUTIL_BEGIN

    void save_rgb_to_png_file(const std::string& filename,const tensor_t<color3b,2>& data);

    void save_rgba_to_png_file(const std::string& filename, const tensor_t<color4b, 2>& data);

VUTIL_END