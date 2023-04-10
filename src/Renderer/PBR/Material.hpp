#pragma once

#include "PBR.hpp"
#include "PBR/Utility/Fresnel.hpp"


VISER_BEGIN

namespace cuda{

    class Material{
      public:

    };

    class PBRSurface : public Material{
      public:


      private:

    };


    class PBRVolumeModel{
      public:
        enum EType{
            BRDF,
            PHASE
        };


        EType type;
        Material* material;
        Medium* medium;

    };


}

VISER_END