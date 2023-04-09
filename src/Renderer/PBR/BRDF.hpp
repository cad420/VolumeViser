#pragma once

#include "PBR.hpp"

#include "Utility/Fresnel.hpp"
#include "Utility/Microfacet.hpp"

VISER_BEGIN

namespace cuda{

    class BXDF{
      public:
        Spectrum eval(){

        }
        float3 sample(){

        }
        float pdf(){

        }
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


    class GGXMicrofacetReflectionBXDF : public BXDF{
      public:

    };

    class DiffuseBXDF : public BXDF{
      public:

    };


}

VISER_END