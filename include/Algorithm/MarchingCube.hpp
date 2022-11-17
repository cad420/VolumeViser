#pragma once

#include <Common/Common.hpp>
#include <Core/Renderer.hpp>

VISER_BEGIN


//大规模体数据的MarchingCube算法
//每次运行算法的时候，不需要将所需要范围的数据拷贝到单独的内存中
//只需要指定数据的范围(原点和大小)，利用虚拟纹理和页表管理，GPU可以读取到指定范围内部的体素
//因此前提是数据全部上传到虚拟纹理中，页表也相应更新，这一步骤和大规模体数据渲染相同。
class MarchingCubeAlgoPrivate;
class MarchingCubeAlgo : public UnifiedRescBase{
public:
    struct MarchingCubeAlgoCreateInfo{
        mutable Ref<GPUMemMgr> gpu_mem_mgr;
        mutable Ref<HostMemMgr> host_mem_mgr;
        //最大为 128mb 个体素， 对应尺寸可以是 512 x 512 x 512
        //因为算法内部会生成中间数据，大小为 体素个数 x 8 字节，因此最大体素个数结合GPU可用内存谨慎设置
        size_t max_voxel_num = 1ull << 27;
    };
    MarchingCubeAlgo(const MarchingCubeAlgoCreateInfo& info);

    ~MarchingCubeAlgo();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void SetVolume(const VolumeParams& volume_params);

    //传入的Texture任意的view设置都可以，内部会重新设置view info然后创建新的tex，但是resc都是一致的
    //可以只需要绑定一次
    void BindVTexture(VTextureHandle handle, TextureUnit unit);

    //其实也可以绑定一次即可 但是每次绑定一下也很方面 只需要一行代码
    void BindPTBuffer(PTBufferHandle handle);

    struct MarchingCubeAlgoParams{
        UInt3 shape;//基于体素的尺寸
        UInt3 origin;//基于体素的起点坐标
        float isovalue;//0-1
        uint32_t lod = 0;
        //这个存放计算结果，是device内存
        CUDABufferView1D<Float3> gen_dev_vertices_ret;
    };
    int Run(MarchingCubeAlgoParams& params);

private:
    std::unique_ptr<MarchingCubeAlgoPrivate> _;
};




VISER_END