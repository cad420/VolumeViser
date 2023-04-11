#pragma once

#include "PBR.hpp"

VISER_BEGIN

namespace cuda{

    class Light{
      public:
        CUB_CPU_GPU virtual ~Light() = default;

        CUB_CPU_GPU virtual Spectrum Sample_Li() const = 0;

        CUB_CPU_GPU virtual Spectrum Le() const = 0;

        CUB_CPU_GPU virtual float Pdf_Li() const = 0;

    };

    class AreaLight : public Light{
      public:
    };

    class DiffuseLight : public AreaLight{

    };

    class InfiniteAreaLight : public Light{

    };

}

VISER_END