#pragma once

#include <Common/Common.hpp>
#include <limits>

VISER_BEGIN

// 渲染需要的参数，可以一开始设置后一直保持不变
// 主要包括光照参数、LOD计算参数、传输函数

//使用统一的标准单位meter作为世界坐标系的度量单位

struct LevelOfDist{
    static constexpr int MaxLevelCount = 16;
    float LOD[MaxLevelCount] = {std::numeric_limits<float>::max()};
};
inline static const Float3 WorldUp = Float3{0.f, 1.f, 0.f};
struct Camera{
    Float3 pos;
    Float3 target;
    Float3 up;
    float near;
    float far;
    float fov;
    int width;
    int height;


    Mat4 GetViewMatrix() const{
        return Transform::look_at(pos, target, up);
    }

    Mat4 GetProjMatrix() const{
        return Transform::perspective(vutil::deg2rad(fov * 0.5f),
                                      static_cast<float>(width) / height,
                                      near, far);
    }

    Mat4 GetProjViewMatrix() const{
        return GetProjMatrix() * GetViewMatrix();
    }

};

struct RenderParams{
    struct {
        bool updated = false;
    }light;
    struct {
        bool updated = false;
        LevelOfDist leve_of_dist;
    }lod;
    struct {
        bool updated = false;
    }tf;
};

// 每一帧需要更新的参数，主要是相机
struct PerFrameParams{

};




VISER_END