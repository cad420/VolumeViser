#pragma once

#include <Common/Common.hpp>
#include <Core/Renderer.hpp>
VISER_BEGIN

/**
 * @param ptrs 一条SWC的所有节点，不能存在分叉，即只能是一条线，数组中依次存放根节点到末尾节点，
 * 每一个节点为一个float4数据，即(pos + radius)，
 * pos应该是该节点在这个volume里的offset，范围为(0,1)
 * radius以体素为单位
 */
 //todo change to template Voxel
inline void CPUVoxelizeSWC(CUDABufferView3D<uint8_t>& volume, const std::vector<Float4>& ptrs){
    uint32_t vol_x = volume.width(), vol_y = volume.height(), vol_z = volume.depth();
    assert(ptrs.size() > 1);

    int n = ptrs.size();

    BoundingBox3D vol_box(Float3(0), Float3(1));

    auto remap = [](float x, float n){
        return 0.5f / n + x * (1.f - 1.f / n);
    };
    auto remap2 = [](float x, float n){
        return (x - 0.5f / n) / (1.f - 1.f / n);
    };
    auto convert = [&](Float3 p, UInt3 shape){
        return UInt3(remap(p.x, shape.x) * shape.x,
                     remap(p.y, shape.y) * shape.y,
                     remap(p.z, shape.z) * shape.z);
    };
    auto reconvert = [&](Float3 coord, UInt3 shape){
        return Float3(remap2(coord.x / shape.x, shape.x),
                      remap2(coord.y / shape.y, shape.y),
                      remap2(coord.z / shape.z, shape.z));
    };

    for(int i = 1; i < n; i++){
        auto& pre = ptrs[i - 1];
        auto& cur = ptrs[i];
        //计算出直线方程
        Float3 A = Float3(pre.x, pre.y, pre.z);
        float radius_a = pre.w;
        Float3 B = Float3(cur.x, cur.y, cur.z);
        float radius_b = cur.w;
        //计算出该段的包围盒，最后要扩张半径大小
        BoundingBox3D box;
        box |= A;
        box |= B;
        box.low -= (std::max)(radius_a, radius_b);
        box.high += (std::max)(radius_a, radius_b);
        box &= vol_box;

        UInt3 low_coord = convert(box.low, UInt3(vol_x, vol_y, vol_z));
        UInt3 high_coord = convert(box.high, UInt3(vol_x, vol_y, vol_z));


        Float3 A2B = B - A;
        Float3 AB = A2B.normalized();
        float AB_dist = dot(A2B, AB);

        auto dist_to_line = [&](Float3 C){
            return cross(C - A, AB).length();
        };

        for(auto z = low_coord.z; z <= high_coord.z; z++){
            for(auto y = low_coord.y; y <= high_coord.y; y++){
                for(auto x = low_coord.x; x <= high_coord.x; x++){
                    auto C = reconvert(Float3(x, y, z), UInt3(vol_x, vol_y, vol_z));
                    float proj_A2C = dot((C - A), AB);
                    bool inner = proj_A2C > 0 && proj_A2C < AB_dist;
                    float R;
                    float C2AB_dist;
                    if(inner){
                        float u = proj_A2C / AB_dist;
                        R = (1.f - u) * radius_a + u * radius_b;
                        C2AB_dist = dist_to_line(C);
                    }
                    else{
                        R = proj_A2C <= 0 ? radius_a : radius_b;
                        C2AB_dist = proj_A2C <= 0 ? (C - A).length() : (C - B).length();
                    }
                    if(C2AB_dist <= R){
                        //fill voxel
                        volume.at(x, y, z) = 255;
                    }
                }
            }
        }
    }

}

//因为swc体素化后的值都是一样的，因此默认使用uint8类型，体素值为255
constexpr uint8_t SWCVoxelVal = 255;
constexpr size_t MaxSegmentCount = 1ull << 20;
using SWCSegment = std::pair<Float4, Float4>;
class SWCVoxelizerPrivate;
class SWCVoxelizer : public UnifiedRescBase{
public:
    struct VoxelizerCreateInfo{
        mutable Ref<GPUMemMgr> gpu_mem_mgr;
        mutable Ref<HostMemMgr> host_mem_mgr;
        size_t max_segment_count = MaxSegmentCount;
    };
    SWCVoxelizer(const VoxelizerCreateInfo& info);

    ~SWCVoxelizer();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void SetVolume(const VolumeParams& volume_params);

    void BindVTexture(VTextureHandle handle, TextureUnit unit);

    void BindPTBuffer(PTBufferHandle handle);

    struct SWCVoxelizeAlgoParams{

//        std::vector<SWCSegment> ptrs;
        CUDABufferView1D<SWCSegment> ptrs;//host buffer
        uint32_t lod = 0;
    };
    void Run(const SWCVoxelizeAlgoParams& params);

private:
    std::unique_ptr<SWCVoxelizerPrivate> _;
};

VISER_END