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

    CUB_CPU_GPU inline float SphericalTheta(float y){
        return acosf(clamp(y, -1.f, 1.f));
    }

    CUB_CPU_GPU inline float SphericalTheta(Vec3 wl){
        return SphericalTheta(wl.y);
    }

    CUB_CPU_GPU inline float SphericalPhi(float x, float z){
        float p = atan2f(z, x);
        return p < 0.f ? p + 2.f * PI_F : p;
    }

    CUB_CPU_GPU inline float SphericalPhi(Vec3 wl){
        return SphericalPhi(wl.x, wl.z);
    }

    CUB_CPU_GPU inline Vec2 ConcentricSampleDisk(float u, float v){
        float r, theta;
        // Map uniform random numbers to $[-1,1]^2$
        float sx = 2 * u - 1;
        float sy = 2 * v - 1;
        // Map square to $(r,\theta)$
        // Handle degeneracy at the origin

        if (sx == 0.0 && sy == 0.0)
        {
            return Vec2{0.f, 0.f};
        }

        if (sx >= -sy)
        {
            if (sx > sy)
            {
                // Handle first region of disk
                r = sx;
                if (sy > 0.0)
                    theta = sy / r;
                else
                    theta = 8.0f + sy / r;
            }
            else
            {
                // Handle second region of disk
                r = sy;
                theta = 2.0f - sx / r;
            }
        }
        else
        {
            if (sx <= sy)
            {
                // Handle third region of disk
                r = -sx;
                theta = 4.0f - sy / r;
            }
            else
            {
                // Handle fourth region of disk
                r = -sy;
                theta = 6.0f + sx / r;
            }
        }

        theta *= PI_F / 4.f;

        return Vec2{r * cosf(theta), r * sinf(theta)};
    }

    CUB_CPU_GPU inline Vec2 ConcentricSampleDisk(Vec2 uv){
        return ConcentricSampleDisk(uv.x, uv.y);
    }

    CUB_CPU_GPU inline Vec3 CosineWeightedHemisphere(float u, float v){
        const auto ret = ConcentricSampleDisk(u, v);
        return Vec3{ret.x ,ret.y, sqrtf(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y))};
    }

    CUB_CPU_GPU inline Vec3 CosineWeightedHemisphere(Vec2 uv){
        return CosineWeightedHemisphere(uv.x, uv.y);
    }


}

VISER_END