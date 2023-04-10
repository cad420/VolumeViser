#pragma once

#include "../PBR.hpp"

VISER_BEGIN

namespace cuda::microfacet{

    CUB_CPU_GPU inline float anisotropic_gtr2(float sin_phi_h, float cos_phi_h,
                                              float sin_theta_h, float cos_theta_h,
                                              float ax, float ay){

    }

    CUB_CPU_GPU inline float smith_anisotropic_gtr2(float cos_phi, float sin_phi, float ax, float ay, float tan_theta){

    }

    CUB_CPU_GPU inline void ggx(Vec3 lwi, Vec3 lwo, Spectrum* eval, float* pdf){

    }
}

VISER_END