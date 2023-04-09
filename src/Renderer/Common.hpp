#pragma once

#include <Common/Common.hpp>
#include "../Common/helper_math.h"
VISER_BEGIN

namespace cuda{

    template<typename T>
    struct TVec3Helper;

    template<>
    struct TVec3Helper<float>{
        using Type = float3;
    };

    template<>
    struct TVec3Helper<int>{
        using Type = int3;
    };

    template<>
    struct TVec3Helper<uint32_t>{
        using Type = uint3;
    };

    template<typename T>
    struct TAABB{
        using Vec3 = typename TVec3Helper<T>::Type;
        Vec3 low;
        Vec3 high;
    };

    using AABB = TAABB<float>;

    using AABB_I = TAABB<int>;

    using AABB_UI = TAABB<uint32_t>;

    struct Ray{
        float3 o;
        float3 d;

        CUB_CPU_GPU Ray() = default;

        CUB_CPU_GPU Ray(float3 origin, float3 dir)
        :o(origin), d(dir)
        {

        }
    };

    struct RayExt : Ray{
        float min_t;
        float max_t;
        size_t pixel_id;

        CUB_CPU_GPU RayExt() = default;

        CUB_CPU_GPU RayExt(float3 origin, float3 dir, float minT, float maxT, size_t id = 0)
        :Ray(origin, dir), min_t(minT), max_t(maxT), pixel_id(id)
        {

        }

        CUB_CPU_GPU RayExt(float3 origin, float3 dir)
        : RayExt(origin, dir, 0.f, std::numeric_limits<float>::max(), 0)
        {

        }

        CUB_CPU_GPU RayExt& operator=(const RayExt& other){
            o = other.o;
            d = other.d;
            min_t = other.min_t;
            max_t = other.max_t;
            pixel_id = other.pixel_id;

            return *this;
        }
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

