#include <Core/Renderer.hpp>



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

    // render params
    // 0. camera
    // 1. light
    // 2. lod
    // 3. tf

    // renderer


    // render loop
    {
        // handle events

        // config per-frame params

        // compute blocks in current view frustum
        // 0. quad-tree query and refine

        // load to v-texture
        // 0. page-table
        // 1. decode blocks
        // 2. upload blocks

        // bind resource and render

        // get result

        // swap buffer

    }

    return 0;
}