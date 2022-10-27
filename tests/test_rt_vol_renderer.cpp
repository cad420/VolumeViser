
#include <Core/Renderer.hpp>
#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Core/HashPageTable.hpp>

using namespace viser;
using namespace vutil::gl;
class rt_renderer: public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    void initialize() override {
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));


    }

    void frame() override {
        handle_events();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 7.f);
        if(ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);
            if (ImGui::Checkbox("VSync", &vsync)) {
                window->set_vsync(vsync);
            }

        }





        ImGui::End();
        ImGui::PopStyleVar();
    }

    void destroy() override {

    }

private:
    bool vsync = true;
};

int main(int argc, char** argv){
    {
        rt_renderer(window_desc_t{.size = {1200, 900},
                                  .title = "rt test",
                                  .resizeable = false,
                                  .multisamples = 1}).run();
        return 0;
    }
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
    auto volume = NewHandle<GridVolume>(RescAccess::Shared, volume_info);

    FixedHostMemMgr::FixedHostMemMgrCreateInfo block_pool_info{};
    auto block_pool_uid = resc_ins.GetHostRef(host_resc_uid)->RegisterFixedHostMemMgr(block_pool_info);

    // v-texture
    GPUMemMgr::GPUVTexMgrCreateInfo tex_info{};
    auto vtex_uid = resc_ins.GetGPURef(gpu_resc_uid)->RegisterGPUVTexMgr(tex_info);


    // async queue for load block from disk to cpu memory
    vutil::thread_group_t thread_group;
    thread_group.start(2);

    // volume oct-tree

    // render params
    // 0. camera
    Camera camera;
    // 1. light
    // 2. lod
    // 3. tf

    // renderer
    bool render_async = false;

    CRTVolumeRenderer rt_vol_renderer({});

    // bind v-textures
    {
        auto gpu_vtex_ref = resc_ins.GetGPURef(gpu_resc_uid)->GetGPUVTexMgrRef(vtex_uid);
        auto vtex = gpu_vtex_ref->GetAllTextures();
        for(auto& vt : vtex){
            rt_vol_renderer.BindVTexture(vt.second, vt.first);
        }
    }

    // framebuffer
    FrameBuffer framebuffer{1200, 900};

    framebuffer.color = resc_ins.GetGPURef(gpu_resc_uid)->AllocPitchedBuffer(RescAccess::Unique,
                                                                             framebuffer.frame_width * sizeof(uint32_t),
                                                                             framebuffer.frame_height, sizeof(uint32_t));
    framebuffer.color = resc_ins.GetGPURef(gpu_resc_uid)->AllocPitchedBuffer(RescAccess::Unique,
                                                                             framebuffer.frame_width * sizeof(float),
                                                                             framebuffer.frame_height, sizeof(float));


    // render loop
    {
        // handle events

        // config per-frame params

        // compute blocks in current view frustum
        // 0. quad-tree query and refine
        auto camera_proj_view = camera.GetProjViewMatrix();
        Frustum camera_view_frustum;
        ExtractFrustumFromMatrix(camera_proj_view, camera_view_frustum);
        static std::vector<GridVolume::BlockUID> intersect_blocks;
        intersect_blocks.clear();

        ComputeIntersectedBlocksWithViewFrustum(intersect_blocks,
                                                *volume,
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
        std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
        page_table->GetAndLock(intersect_blocks, blocks_info);


        auto host_block_pool = resc_ins.GetHostRef(host_resc_uid)->GetFixedHostMemMgrRef(block_pool_uid);
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;//在循环结束会释放Handle
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        for(auto& missed_block : blocks_info){
            if(!missed_block.second.Missed()) continue;
            //应该获取那些没有位于异步解压队列中的block
            auto block_handle = host_block_pool->GetBlock(missed_block.first.ToUnifiedRescUID());
            if(!block_handle.IsWriteLocked()){
                //说明这个block的handle被别人获取了还没有释放
                continue;
            }
            if(missed_block.first.IsSame(block_handle.GetUID())){
                block_handle.AddReadLock();
                host_blocks[missed_block.first] = std::move(block_handle);//move
            }
            else{
                block_handle.AddWriteLock();
                block_handle.SetUID(missed_block.first.ToUnifiedRescUID());
                missed_host_blocks[missed_block.first] = std::move(block_handle);
            }
        }

        // 1. decode blocks
        // if sync then wait for decode finished, else just append task and move on
        // 把解压任务加入到异步队列中，甚至可以细分任务的优先级，比如对于lod较小的数据优先执行
        // 加入异步队列的Handle只有在该任务被完成后才会被释放，不会在一次循环结束时释放
        std::map<int, std::vector<std::function<void()>>> task_mp;
        for(auto& missed_block : missed_host_blocks){
            int lod = missed_block.first.GetLOD();
            task_mp[lod].emplace_back([&,block = missed_block.first,
                                       block_handle = std::move(missed_block.second)]()mutable{
                volume->ReadBlock(block, *block_handle);
                block_handle.SetUID(block.ToUnifiedRescUID());
                if(!render_async)//note!!!
                {
                    block_handle.ConvertWriteToReadLock();
                    host_blocks[block] = std::move(block_handle);
                }
                else{
                    //异步加载的数据只是加载到host mem pool中，然后释放写锁
                    block_handle.ReleaseWriteLock();
                }
            });
        }
        std::map<int, vutil::task_group_handle_t> task_groups;
        std::vector<int> lods;
        for(auto& task : task_mp){
            int count = task.second.size();
            int lod = task.first;
            auto& tasks = task.second;
            assert(count > 0);
            lods.emplace_back(lod);
            auto task_group = thread_group.create_task(std::move(tasks.front()));
            for(int i = 1; i < count; i++){
                thread_group.enqueue_task(*task_group, std::move(tasks[i]));
            }
            task_groups[lod] = std::move(task_group);
        }
        int lod_count = lods.size();
        for(int i = 0; i < lod_count - 1; i++){
            int first = lods[i], second = lods[i + 1];
            thread_group.add_dependency(*task_groups[second], *task_groups[first]);
        }
        for(auto& [_, task_group] : task_groups){
            thread_group.submit(task_group);
        }
        if(!render_async){
            thread_group.wait_idle();
        }

        // 2. upload blocks
        for(auto& missed_block : blocks_info){
            if(!missed_block.second.Missed()) continue;
            auto& handle = host_blocks[missed_block.first];
            //这部分已经在CPU的数据块，调用异步memcpy到GPU
            gpu_vtex_ref->UploadBlockToGPUTexAsync(handle, missed_block.second);
            handle.ReleaseReadLock();
        }

        // bind resource and render
        if(!host_blocks.empty()){
            rt_vol_renderer.BindPTBuffer(page_table->GetPageTable().GetHandle());

        }
        // flush transfer before render start
        gpu_vtex_ref->Flush();

        // get result
        rt_vol_renderer.Render(framebuffer);

        // release resource

        page_table->Release(intersect_blocks);

        // swap buffer

    }

    return 0;
}