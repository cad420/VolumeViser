#pragma once

#include "PBR.hpp"

#include "Utility/Fresnel.hpp"
#include "Utility/Microfacet.hpp"

VISER_BEGIN

namespace cuda{

    struct BXDFSampleResult{

    };

    class BXDF{
      public:

        CUB_CPU_GPU virtual ~BXDF() = default;

        CUB_CPU_GPU virtual Spectrum eval(Vec3 lwi, Vec3 lwo) = 0;

        CUB_CPU_GPU virtual BXDFSampleResult sample(Vec3 lwo, Vec2 uv) = 0;

        CUB_CPU_GPU virtual float pdf(Vec3 lwi, Vec3 lwo) = 0;
    };

    class GGXMicrofacetReflectionBXDF : public BXDF{
      public:
        CUB_CPU_GPU GGXMicrofacetReflectionBXDF(const FresnelPoint* fresnelPoint,
                                                float roughness,
                                                float anisotropic);

        CUB_CPU_GPU ~GGXMicrofacetReflectionBXDF() override = default;

        CUB_CPU_GPU Spectrum eval(Vec3 lwi, Vec3 lwo) override;

        CUB_CPU_GPU BXDFSampleResult sample(Vec3 lwo, Vec2 uv) override;

        CUB_CPU_GPU float pdf(Vec3 lwi, Vec3 lwo) override;

      private:
        const FresnelPoint* fresnel;

        float ax;
        float ay;
    };

    class DiffuseBXDF : public BXDF{
      public:
        CUB_CPU_GPU explicit DiffuseBXDF(float3 albedo);

        CUB_CPU_GPU ~DiffuseBXDF() override = default;

        CUB_CPU_GPU Spectrum eval(Vec3 lwi, Vec3 lwo) override;

        CUB_CPU_GPU BXDFSampleResult sample(Vec3 lwo, Vec2 uv) override;

        CUB_CPU_GPU float pdf(Vec3 lwi, Vec3 lwo) override;

      private:
        Spectrum coef;// albedo / PI
    };

    class BSDF{
      public:

        CUB_CPU_GPU Spectrum eval(){

        }
        CUB_CPU_GPU float3 sample(){

        }
        CUB_CPU_GPU float pdf(){

        }

    };

    template<int MAX_CNT>
    class AggregateBSDF : public BSDF{
      public:

    };



}

VISER_END