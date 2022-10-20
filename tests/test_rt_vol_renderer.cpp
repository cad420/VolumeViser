#include <Core/Renderer.hpp>
#include <Algorithm/LevelOfDetailPolicy.hpp>


using namespace viser;

int main(int argc, char** argv){
    auto& resc_ins = ResourceMgr::GetInstance();

    // resource manager
    auto host_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                       .MaxMemBytes = 32ull << 30,
                                                       .DeviceIndex = -1});
    auto gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                      .MaxMemBytes = 16ull << 30,
                                                      .DeviceIndex = 0});

    // volume
    GridVolume::GridVolumeCreateInfo volume_info{};
    auto volume_uid = resc_ins.GetHostRef(host_resc_uid)->RegisterGridVolume(volume_info);

    FixedHostMemMgr::FixedHostMemMgrCreateInfo block_pool_info{};
    auto block_pool_uid = resc_ins.GetHostRef(host_resc_uid)->RegisterFixedHostMemMgr(block_pool_info);

    // v-texture
    GPUMemMgr::GPUVTexMgrCreateInfo tex_info{};
    auto vtex_uid = resc_ins.GetGPURef(gpu_resc_uid)->RegisterGPUVTexMgr(tex_info);


    // async queue for load block from disk to cpu memory

    // volume oct-tree

    // render params
    // 0. camera
    Camera camera;
    // 1. light
    // 2. lod
    // 3. tf

    // renderer



    // render loop
    {
        auto volume_ref = resc_ins.GetHostRef(host_resc_uid)->GetGridVolumeRef(volume_uid);
        // handle events

        // config per-frame params

        // compute blocks in current view frustum
        // 0. quad-tree query and refine
        auto camera_proj_view = camera.GetProjViewMatrix();
        Frustum camera_view_frustum;
        ExtractFrustumFromMatrix(camera_proj_view, camera_view_frustum);
        static std::vector<GridVolume::BlockUID> intersect_blocks;

        ComputeIntersectedBlocksWithViewFrustum(intersect_blocks,
                                                *volume_ref,
                                                camera_view_frustum,
                                                camera.pos,
                                                [](float)->int{
           //...
            return 0;
        });


        // load to v-texture
        // 0. page-table
        //
        auto gpu_vtex_ref = resc_ins.GetGPURef(gpu_resc_uid)->GetGPUVTexMgrRef(vtex_uid);
        auto page_table = gpu_vtex_ref->GetGPUPageTableMgrRef();
        std::vector<GPUPageTableMgr::PageTableItem> missed_blocks;
        page_table->GetAndLock(intersect_blocks, missed_blocks);


        auto host_block_pool = resc_ins.GetHostRef(host_resc_uid)->GetFixedHostMemMgrRef(block_pool_uid);
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        for(auto& missed_block : missed_blocks){
            auto block_handle = host_block_pool->GetBlock(missed_block.first.ToUnifiedRescUID(), true);
            if(missed_block.first.IsSame(block_handle.UID()))
                host_blocks[missed_block.first] = block_handle;
            else
                missed_host_blocks[missed_block.first] = block_handle;
        }
        // 1. decode blocks
        // if sync then wait for decode finished, else just append task and move on

        // 2. upload blocks
        for(auto& missed_block : missed_blocks){
            gpu_vtex_ref->UploadBlockToGPUTex(host_blocks[missed_block.first], missed_block.second);
        }

        // bind resource and render
        if(!host_blocks.empty()){

        }


        // get result

        // release resource

        page_table->Release();

        // swap buffer

    }

    return 0;
}