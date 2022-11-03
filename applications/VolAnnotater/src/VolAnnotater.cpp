#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Core/HashPageTable.hpp>
#include "VolAnnotater.hpp"
#include "Common.hpp"
#include <cuda_gl_interop.h>
#include <json.hpp>
#include <fstream>
//标注系统的窗口绘制任务交给OpenGL，如果有多个显卡，其余的显卡可以用于网格重建任务
class VolAnnotaterApp : public gl_app_t{
    // *.lod.desc.json
    void loadLODVolumeData(const std::string& lod_filename){
        LOG_DEBUG("Load LOD Volume Data : {}", lod_filename);
        std::ifstream in(lod_filename);
        if(!in.is_open()){
            throw std::runtime_error("Open LOD Volume Data File failed : " + lod_filename);
        }
        nlohmann::json j;
        in >> j;

        uint32_t levels = j.at("levels");
        if(levels > LevelOfDist::MaxLevelCount){
            throw std::runtime_error("Invalid levels for lod volume : " + std::to_string(levels));
        }
        this->max_lod = levels - 1;
        GridVolume::GridVolumeCreateInfo vol_info{
            .host_mem_mgr_uid = host_mem_mgr_ref->GetUID(),
            .gpu_mem_mgr_uid = render_gpu_mem_mgr_ref->GetUID(),
            .levels = levels
        };
        for(uint32_t lod = 0; lod < levels; lod++){
            std::string lod_path = j.at("lod" + std::to_string(lod));
            LOG_DEBUG("Load LOD({}) : {}", lod, lod_path);
            vol_info.lod_vol_file_io[lod] = Handle<VolumeIOInterface>(RescAccess::Shared, std::make_shared<EBVolumeFile>(lod_path));
        }
        volume = NewHandle<GridVolume>(RescAccess::Unique, vol_info);
        LOG_DEBUG("Load LOD Volume({}) successfully", volume->GetDesc().volume_name);
        VISER_WHEN_DEBUG(std::cout << volume->GetDesc() << std::endl)
    }
    //全局资源的初始化
    void initGlobalResource(){
        LOG_DEBUG("Start Global Resource Init...");
        auto& resc_ins = ResourceMgr::GetInstance();

        auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                                     .MaxMemBytes = create_info.max_host_mem_bytes,
                                                                     .DeviceIndex = -1});
        this->host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

        auto render_gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                                        .MaxMemBytes = create_info.max_render_gpu_mem_bytes,
                                                                        .DeviceIndex = create_info.render_gpu_index});
        this->render_gpu_mem_mgr_ref = resc_ins.GetGPURef(render_gpu_resc_uid);


        LOG_DEBUG("Successfully Finish Global Resource Init...");
    }
    void initVolDataResource(){
        LOG_DEBUG("Start Vol Data Resource Init...");

        volume_desc = volume->GetDesc();

        size_t block_size = (size_t)(volume_desc.block_length + volume_desc.padding * 2) * volume_desc.bits_per_sample
                            * volume_desc.samples_per_voxel / 8;
        block_size *= block_size * block_size;
        if(block_size == 0){
            throw std::runtime_error("Invalid block size equal to zero : " + std::to_string(block_size));
        }
        size_t block_num = create_info.max_fixed_host_mem_bytes / block_size;
        if(block_num == 0){
            throw std::runtime_error("Invalid block size: " + std::to_string(block_size));
        }

        FixedHostMemMgr::FixedHostMemMgrCreateInfo host_pool_info{
            .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
                    .fixed_block_size = block_size,
                    .fixed_block_num = block_num
        };
        auto host_pool_uid = host_mem_mgr_ref->RegisterFixedHostMemMgr(host_pool_info);
        host_block_pool_ref = host_mem_mgr_ref->GetFixedHostMemMgrRef(host_pool_uid);
        LOG_DEBUG("Successfully Create FixedHostMemMgr...");

        thread_group.start(create_info.threads_count);
        LOG_DEBUG("Successfully Start ThreadGroup with Count : {}", create_info.threads_count);

        GPUMemMgr::GPUVTexMgrCreateInfo vtex_info{
                .gpu_mem_mgr = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false),
                .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
                .vtex_count = create_info.vtex_count,
                .vtex_shape = {create_info.vtex_shape_x, create_info.vtex_shape_y, create_info.vtex_shape_z},
                .bits_per_sample = volume_desc.bits_per_sample,
                .samples_per_channel = volume_desc.samples_per_voxel,
                .vtex_block_length = (int)(volume_desc.block_length + volume_desc.padding * 2),
                .is_float = volume_desc.is_float, .exclusive = true
        };
        auto vtex_uid = render_gpu_mem_mgr_ref->RegisterGPUVTexMgr(vtex_info);
        gpu_vtex_mgr_ref = render_gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);
        gpu_pt_mgr_ref = gpu_vtex_mgr_ref->GetGPUPageTableMgrRef();

        {
            float volume_base_space = std::min({volume_desc.voxel_space.x,
                                                volume_desc.voxel_space.y,
                                                volume_desc.voxel_space.z});
            if(volume_base_space == 0)
                volume_space_ratio = Float3(1.f);
            else
                volume_space_ratio = Float3(volume_desc.voxel_space.x / volume_base_space,
                                            volume_desc.voxel_space.y / volume_base_space,
                                            volume_desc.voxel_space.z / volume_base_space);
            volume_box = {
                    Float3(0.f, 0.f, 0.f),
                    Float3(volume_desc.shape.x * volume_space_ratio.x * render_base_space,
                           volume_desc.shape.y * volume_space_ratio.y * render_base_space,
                           volume_desc.shape.z * volume_space_ratio.z * render_base_space)
            };
        }


        LOG_DEBUG("Successfully Finish Vol Data Resource Init...");
    }
    //与渲染器相关资源的初始化
    void initRenderResource(){
        LOG_DEBUG("Start Render Resource Init...");
        CRTVolumeRenderer::CRTVolumeRendererCreateInfo renderer_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false)
        };
        crt_vol_renderer = NewHandle<CRTVolumeRenderer>(RescAccess::Unique, renderer_info);


        ComputeUpBoundLOD(lod, render_base_space, offscreen.frame_width, offscreen.frame_height,
                          vutil::deg2rad(45.f));
        lod.LOD[max_lod] = std::numeric_limits<float>::max();

        LOG_DEBUG("Successfully Finish Render Resource Init...");
    }

    void registerCUDAGLInteropResource(){
        //注册cudaGL资源
        {

            cudaGL_interop.color_pbo.initialize_handle();
            cudaGL_interop.color_pbo.reinitialize_buffer_data(nullptr, framebuffer->frame_width * framebuffer->frame_height * sizeof(uint32_t), GL_DYNAMIC_COPY);
            CUB_CHECK(cudaGraphicsGLRegisterBuffer(&cudaGL_interop.cuda_frame_color_resc, cudaGL_interop.color_pbo.handle(), cudaGraphicsRegisterFlagsWriteDiscard));
            cudaGL_interop.depth_pbo.initialize_handle();
            cudaGL_interop.depth_pbo.reinitialize_buffer_data(nullptr, framebuffer->frame_width * framebuffer->frame_height * sizeof(float), GL_DYNAMIC_COPY);
            CUB_CHECK(cudaGraphicsGLRegisterBuffer(&cudaGL_interop.cuda_frame_depth_resc, cudaGL_interop.depth_pbo.handle(), cudaGraphicsRegisterFlagsWriteDiscard));

        }
    }
public:
    using gl_app_t::gl_app_t;

    void initialize(const VolAnnotater::VolAnnotaterCreateInfo& info){
        //tmp assert
        assert(info.render_compute_same_gpu);

        this->create_info = info;

        initGlobalResource();

        const std::string lod_vol_filename = "test_kingsnake.lod.desc.json";

        loadLODVolumeData(lod_vol_filename);

        initVolDataResource();

        initRenderResource();


    }

    void initialize() override {


        //todo resize event
        framebuffer = NewGeneralHandle<FrameBuffer>(RescAccess::Unique);
        framebuffer->frame_width = offscreen.frame_width;
        framebuffer->frame_height = offscreen.frame_height;

        //todo
        offscreen.fbo.initialize_handle();
        offscreen.rbo.initialize_handle();

        offscreen.color.initialize_handle();
        offscreen.color.initialize_texture(1, GL_RGBA8, offscreen.frame_width, offscreen.frame_height);
        offscreen.depth.initialize_handle();
        offscreen.depth.initialize_texture(1, GL_R32F,  offscreen.frame_width, offscreen.frame_height);


        registerCUDAGLInteropResource();

        {
            debug.host_color.resize(framebuffer->frame_height * framebuffer->frame_width);
            debug.host_depth.resize(framebuffer->frame_height * framebuffer->frame_width);
        }

        camera.set_position({1.012f, 1.012f, 1.6f});
        camera.set_perspective(45.f, 0.01f, 10.f);
        camera.set_direction(vutil::deg2rad(-90.f), 0.f);
    }

    void frame() override {
        handle_events();

        // map cudaGL interop资源
        cudaGraphicsResource_t rescs[2] = {cudaGL_interop.cuda_frame_color_resc, cudaGL_interop.cuda_frame_depth_resc};
        CUB_CHECK(cudaGraphicsMapResources(2, rescs));
        void* color_mapping_ptr = nullptr;
        size_t color_mapping_size = 0;
        CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&color_mapping_ptr, &color_mapping_size, cudaGL_interop.cuda_frame_color_resc));
        assert(color_mapping_ptr && color_mapping_size == framebuffer->frame_width * framebuffer->frame_height * sizeof(uint32_t));
        void* depth_mapping_ptr = nullptr;
        size_t depth_mapping_size = 0;
        CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&depth_mapping_ptr, &depth_mapping_size, cudaGL_interop.cuda_frame_depth_resc));
        assert(depth_mapping_ptr && depth_mapping_size == framebuffer->frame_width * framebuffer->frame_height * sizeof(float));

        framebuffer->color = CUDABufferView2D<uint32_t>(color_mapping_ptr, {.pitch = framebuffer->frame_width * sizeof(uint32_t),
                                                                            .xsize = (size_t) framebuffer->frame_width,
                                                                            .ysize = (size_t) framebuffer->frame_height});
        framebuffer->depth = CUDABufferView2D<float>(depth_mapping_ptr, {.pitch = framebuffer->frame_width * sizeof(float),
                                                                         .xsize = (size_t) framebuffer->frame_width,
                                                                         .ysize = (size_t) framebuffer->frame_height});

        //=========================================
        // 设置体渲染参数并进行CUDA绘制

        render_volume();

        CUB_CHECK(cudaGraphicsUnmapResources(2, rescs));

        //=========================================

        //将cudaGL interop资源拷贝到OpenGL纹理中
        cudaGL_interop.color_pbo.bind();
        offscreen.color.set_texture_data<color4b>(offscreen.frame_width, offscreen.frame_height, nullptr);
        cudaGL_interop.color_pbo.unbind();

        cudaGL_interop.depth_pbo.bind();
        offscreen.depth.set_texture_data<GLfloat>(offscreen.frame_width, offscreen.frame_height, nullptr);
        cudaGL_interop.depth_pbo.unbind();

        GL_EXPR(glFinish());



        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();

        frame_vol();


    }

    void destroy() override{

    }
private:
    void update_vol_camera(){
        vol_camera.width = offscreen.frame_width;
        vol_camera.height = offscreen.frame_height;
        vol_camera.fov = 45.f;
        vol_camera.near = camera.get_near_z();
        vol_camera.far = camera.get_far_z();
        vol_camera.pos = camera.get_position();
        vol_camera.target = camera.get_xyz_direction() + vol_camera.pos;
        static Float3 world_up = Float3(0, 1, 0);
        Float3 right = cross(camera.get_xyz_direction(), world_up);
        vol_camera.up = cross(right, camera.get_xyz_direction());
    }
    void render_volume(){

        // 计算视锥体内的数据块
        update_vol_camera();
        auto camera_proj_view = vol_camera.GetProjViewMatrix();
        Frustum camera_view_frustum;
        ExtractFrustumFromMatrix(camera_proj_view, camera_view_frustum);
        static std::vector<GridVolume::BlockUID> intersect_blocks;
        intersect_blocks.clear();
        ComputeIntersectedBlocksWithViewFrustum(intersect_blocks,
                                                (float)volume_desc.block_length * volume_space_ratio * render_base_space,
                                                volume_desc.blocked_dim,
                                                volume_box,
                                                camera_view_frustum,
                                                [this, pos = vol_camera.pos]
                                                (const BoundingBox3D& box)->int{
            auto center = (box.low + box.high) * 0.5f;
            float dist = (center - pos).length();
            for(int i = 0; i <= max_lod; i++){
                if(dist < this->lod.LOD[i])
                    return i;
            }
            return max_lod;
        });
        VISER_WHEN_DEBUG(
                LOG_DEBUG("total intersect block count : {}", intersect_blocks.size());
                for(auto& b : intersect_blocks){
                    LOG_DEBUG("intersect lod {} block : {} {} {}", b.GetLOD(), b.x, b.y, b.z);
                }
                )
        // 加载缺失的数据块到虚拟纹理中
        static std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
        blocks_info.clear();
        gpu_pt_mgr_ref->GetAndLock(intersect_blocks, blocks_info);

        //因为是单机同步的，不需要加任何锁
        // 暂时使用同步等待加载完数据块
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;//在循环结束会释放Handle
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        for(auto& block : blocks_info){
            if(!block.second.Missed()) continue;
            auto block_hd = host_block_pool_ref->GetBlock(block.first.ToUnifiedRescUID());
            if(block.first.IsSame(block_hd.GetUID())){
                host_blocks[block.first] = std::move(block_hd);
            }
            else{
                block_hd.SetUID(block.first.ToUnifiedRescUID());
                missed_host_blocks[block.first] = std::move(block_hd);
            }
        }

        // 解压数据块
        // 这里先加载lod小的数据块，但是在同步下是没关系的，只有在异步加载的时候有意义
        static std::map<int, std::vector<std::function<void()>>> task_mp;
        task_mp.clear();
        for(auto& missed_block : missed_host_blocks){
            int lod = missed_block.first.GetLOD();
            task_mp[lod].emplace_back(
                    [&, block = missed_block.first,
                     block_handle = std::move(missed_block.second)
                     ]()mutable{
                        volume->ReadBlock(block, *block_handle);
                        block_handle.SetUID(block.ToUnifiedRescUID());
                        host_blocks[block] = std::move(block_handle);
                        LOG_DEBUG("finish lod {} block ({}, {}, {}) loading...",
                                  block.GetLOD(), block.x, block.y, block.z);
                    });
        }
        static std::map<int, vutil::task_group_handle_t> task_groups; task_groups.clear();
        static std::vector<int> lods; lods.clear();
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
        //同步，等待所有解压任务完成
        thread_group.wait_idle();

        //将数据块上传到虚拟纹理
        for(auto& missed_block : blocks_info){
            if(!missed_block.second.Missed()) continue;
            auto& handle = host_blocks[missed_block.first];
            //这部分已经在CPU的数据块，调用异步memcpy到GPU
            gpu_vtex_mgr_ref->UploadBlockToGPUTexAsync(handle, missed_block.second);
        }

        gpu_vtex_mgr_ref->Flush();


//        crt_vol_renderer->BindPTBuffer(gpu_pt_mgr_ref->GetPageTable().GetHandle());

        //更新每一帧绘制的参数
        static PerFrameParams per_frame_params{};
        updatePerFrameParams(per_frame_params);
        crt_vol_renderer->SetPerFrameParams(per_frame_params);

        crt_vol_renderer->Render(framebuffer);

        gpu_pt_mgr_ref->Release(intersect_blocks);
    }

private:
    void updatePerFrameParams(PerFrameParams& params){
        params.frame_width = framebuffer->frame_width;
        params.frame_height = framebuffer->frame_height;
        params.frame_w_over_h = (float)framebuffer->frame_width / (float)framebuffer->frame_height;
    }
private:
    //将体绘制的结果画到ImGui窗口上
    void frame_vol(){
        ImGui::Begin("Volume Render Frame", 0, ImGuiWindowFlags_NoResize);

        ImGui::Image((void*)(intptr_t)(offscreen.color.handle()),
                ImVec2(offscreen.frame_width, offscreen.frame_height));


        ImGui::End();
    }
private:
    struct{
        std::vector<uint32_t> host_color;
        std::vector<float> host_depth;
    }debug;

    VolAnnotater::VolAnnotaterCreateInfo create_info;

    //标注系统不考虑并行，直接保存加了锁的Ref就好
    Ref<HostMemMgr> host_mem_mgr_ref;
    Ref<GPUMemMgr> render_gpu_mem_mgr_ref;
//    Ref<GPUMemMgr> compute_gpu_mem_mgr_ref;

    //每次标注只针对一个体数据，可以运行过程中动态加载更换体数据对象
    Handle<GridVolume> volume;

    Ref<FixedHostMemMgr> host_block_pool_ref;

    //只有一个渲染器，直接保存加锁Ref
    Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
    Ref<GPUPageTableMgr> gpu_pt_mgr_ref;

    vutil::thread_group_t thread_group;

    //由OpenGL资源注册得到的CUDA资源句柄

    Handle<FrameBuffer> framebuffer;


    Handle<CRTVolumeRenderer> crt_vol_renderer;

    viser::Camera vol_camera;

    float render_base_space = 0.002f;
    BoundingBox3D volume_box;
    GridVolume::GridVolumeDesc volume_desc;
    int max_lod;
    viser::LevelOfDist lod;
    Float3 volume_space_ratio;


    // OpenGL资源
    // CUDA渲染器先渲染到离屏帧后，再输出到屏幕或ImGui
    struct{
        int frame_width = 900;
        int frame_height = 600;
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t color;
        texture2d_t depth;
    }offscreen;

    // CUDA OpenGL interop
    struct{
        cudaGraphicsResource_t cuda_frame_color_resc;
        cudaGraphicsResource_t cuda_frame_depth_resc;
        pixel_unpack_buffer color_pbo;
        pixel_unpack_buffer depth_pbo;
    }cudaGL_interop;
};

class VolAnnotaterPrivate{
public:

    VolAnnotater::VolAnnotaterCreateInfo create_info;
};

VolAnnotater::VolAnnotater(const VolAnnotaterCreateInfo &info) {
    _ = std::make_unique<VolAnnotaterPrivate>();
    _->create_info = info;
}

VolAnnotater::~VolAnnotater() {

}

void VolAnnotater::run() {
    SET_LOG_LEVEL_DEBUG
    auto app = std::make_unique<VolAnnotaterApp>(window_desc_t{
        .size = {1920, 1080}, .title = "VolAnnotater"
    });
    try {
        app->initialize(_->create_info);
    }
    catch (const std::exception& err) {
        std::cerr << "Program exit with exception : " << err.what() << std::endl;
        return;
    }
    LOG_DEBUG("start render loop");
    try {
        app->run();
    }
    catch (const std::exception& err) {
        std::cerr << "Program exit with exception : " << err.what() << std::endl;
    }
}
