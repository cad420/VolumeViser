#pragma once

#include "math/tensor.hpp"
#include "math/color3.hpp"

VUTIL_BEGIN

    void save_rgb_to_png_file(const std::string& filename,const tensor_t<color3b,2>& data);

VUTIL_END