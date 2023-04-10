#pragma once

#include "../PBR.hpp"

VISER_BEGIN

namespace cuda{


    CUB_CPU_GPU inline float DielectricFresnel(float etaI, float etaO, float cosThetaI) {
        if(cosThetaI < 0.f){
            swap(etaI, etaO);
            cosThetaI = -cosThetaI;
        }

        const float sin_theta_i = sqrtf(max(0.f, 1.f - cosThetaI * cosThetaI));
        const float sin_theta_t = etaO / etaI * sin_theta_i;

        if(sin_theta_t >= 1.f) return 1.f;

        const float cos_theta_t = sqrtf(max(0.f, 1.f - sin_theta_t * sin_theta_t));
        const float para = (etaI * cosThetaI - etaO * cos_theta_t) / (etaI * cosThetaI + etaO * cos_theta_t);
        const float perp = (etaO * cosThetaI - etaI * cos_theta_t) / (etaO * cosThetaI + etaI * cos_theta_t);
        return 0.5f * (para * para + perp * perp);
    }


    class FresnelPoint{
      public:
        CUB_CPU_GPU virtual ~FresnelPoint() = default;

        CUB_CPU_GPU virtual Spectrum eval(float cosThetaI) const = 0;
    };

    class DielectricFresnelPoint : public FresnelPoint{
      public:
        CUB_CPU_GPU DielectricFresnelPoint(float etaIn, float etaOut)
        : eta_i(etaIn), eta_o(etaOut)
        {

        }

        CUB_CPU_GPU Spectrum eval(float cosThetaI) const override{
            auto ret = DielectricFresnel(eta_i, eta_o, cosThetaI);
            return Spectrum{ret, ret, ret};
        }

      private:
        float eta_i;
        float eta_o;
    };


}

VISER_END