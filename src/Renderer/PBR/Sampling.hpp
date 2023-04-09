#pragma once

#include "PBR.hpp"

VISER_BEGIN

namespace cuda{

    CUB_CPU_GPU inline Vec3 UniformSampleSphere(float u, float v){
        float z = 1.f - 2.f * u;
        float r = sqrtf(max(0.f, 1.f - z*z));
        float phi = 2.f * PI_F * v;
        float x = r * cosf(phi);
        float y = r * sinf(phi);
        return Vec3{x, y, z};
    }

    CUB_CPU_GPU inline Vec3 UniformSampleSphere(Vec2 uv){
        return UniformSampleSphere(uv.x, uv.y);
    }


}

VISER_END