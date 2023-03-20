#pragma once

#include <Common/Common.hpp>
#include "../Common/helper_math.h"
VISER_BEGIN

namespace cuda{

    struct AABB{
        float3 low;
        float3 high;
    };

    struct AABB_UI{
        uint3 low;
        uint3 high;
    };

    struct AABB_I{
        int3 low;
        int3 high;
    };

    struct Ray{
        float3 o;
        float3 d;
    };

    //返回光线与AABB相交的两个时刻
    //相交的话满足 t1 < t2 and t2 > 0
    CUB_CPU_GPU inline float2 RayIntersectAABB(const float3& low, const float3& high, const float3& o, const float3& inv_dir){
        float t_min_x = (low.x - o.x) * inv_dir.x;
        float t_max_x = (high.x - o.x) * inv_dir.x;
        if(inv_dir.x < 0.f){
            float t = t_min_x;
            t_min_x = t_max_x;
            t_max_x = t;
        }

        float t_min_y = (low.y - o.y) * inv_dir.y;
        float t_max_y = (high.y - o.y) * inv_dir.y;
        if(inv_dir.y < 0.f){
            float t = t_min_y;
            t_min_y = t_max_y;
            t_max_y = t;
        }

        float t_min_z = (low.z - o.z) * inv_dir.z;
        float t_max_z = (high.z - o.z) * inv_dir.z;
        if(inv_dir.z < 0.f){
            float t = t_min_z;
            t_min_z = t_max_z;
            t_max_z = t;
        }
        float2 ret;
        ret.x = max(t_min_x, max(t_min_y, t_min_z));
        ret.y = min(t_max_x, min(t_max_y, t_max_z));
        return ret;
    }

    CUB_CPU_GPU inline float2 RayIntersectAABB(const float3& low, const float3 high, const Ray& ray){
        return RayIntersectAABB(low, high, ray.o, 1.f / ray.d);
    }

    CUB_CPU_GPU inline float2 RayIntersectAABB(const AABB& box, const Ray& ray){
        return RayIntersectAABB(box.low, box.high, ray);
    }


}


VISER_END

