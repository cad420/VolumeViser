#pragma once

#include "../common.hpp"

VUTIL_BEGIN

inline float intBitsToFloat(int x){
    float ret;
    std::memcpy(&ret, &x, sizeof(x));
    return ret;
}


VUTIL_END