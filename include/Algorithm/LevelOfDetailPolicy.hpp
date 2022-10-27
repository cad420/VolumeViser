#pragma once

#include <Common/Common.hpp>
#include <Core/RenderParams.hpp>
#include <Model/Volume.hpp>

VISER_BEGIN

//计算视锥体相交的块，其实不需要八叉树也可以
//先确定视锥体的包围盒AABB，根据包围盒的范围遍历虚拟节点，计算其是否与视锥体相交，而且可以并行


void ComputeIntersectedBlocksWithViewFrustum(std::vector<GridVolume::BlockUID>& blocks,
                                             const GridVolume& volume,
                                             const Frustum& funstum,
                                             const Float3& view,
                                             std::function<int(float)> lod);

template<typename ViewSpace>
void ComputeIntersectedBlocksWithViewSpace(std::vector<GridVolume::BlockUID>& blocks,
                                             const GridVolume& volume,
                                             const ViewSpace& space,
                                             const Float3& block);


/**
 * @brief 计算理论上不造成画面失真、闪烁、摩尔纹时每一个lod对应的最远距离
 * @param base_space 一个体素表示的最小space，对应采样频率最高
 * @param width 渲染帧的宽
 * @param height 渲染帧的高
 * @param fov 相机y方向的缩放角(radians)
 */
void ComputeUpBoundLOD(LevelOfDist& lod, float base_space, int width, int height, float fov);


VISER_END