#pragma once

#include "PBR.hpp"

VISER_BEGIN

namespace cuda{

    class RNG{
      public :
        CUB_CPU_GPU RNG(uint32_t* pSeed0, uint32_t* pSeed1)
        :_p_seed0(pSeed0), _p_seed1(pSeed1)
        {

        }
        CUB_CPU_GPU float Get1(){
            *_p_seed0 = 36969 * ((*_p_seed0) & 65535) + ((*_p_seed0) >> 16);
            *_p_seed1 = 18000 * ((*_p_seed1) & 65535) + ((*_p_seed1) >> 16);

            unsigned int ires = ((*_p_seed0) << 16) + (*_p_seed1);

            union
            {
                float f;
                unsigned int ui;
            } res;

            res.ui = (ires & 0x007fffff) | 0x40000000;

            return (res.f - 2.f) / 2.f;
        }

        CUB_CPU_GPU Vec2 Get2(){
            return Vec2{Get1(), Get1()};
        }
        CUB_CPU_GPU Vec3 Get3(){
            return Vec3{Get1(), Get1(), Get1()};
        }
      private:
        uint32_t* _p_seed0;
        uint32_t* _p_seed1;
    };

}

VISER_END