#pragma once

#include "../Common.hpp"

VISER_BEGIN

namespace cuda{

#define KRNL_SS_BLOCK_W		16
#define KRNL_SS_BLOCK_H		8
#define KRNL_SS_BLOCK_SIZE	KRNL_SS_BLOCK_W * KRNL_SS_BLOCK_H
#define PI_F												3.141592654f
#define INV_PI_F											0.31830988618379067154f
#define INV_TWO_PI_F										0.15915494309189533577f
#define FOUR_PI_F											4.0f * PI_F
#define INV_4_PI_F											1.0f / FOUR_PI_F
#define FLT_MAX                                             3.402823466e+38F        // max value
#define	INF_MAX												FLT_MAX
#define MAX_NO_LIGHTS                                       4

using Spectrum = float3;

using Vec3 = typename TVec3Helper<float>::Type;
using Vec2 = float2;

class Material;

class BSDF;

class Medium;




}


VISER_END