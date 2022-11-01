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

        auto vol_desc = volume->GetDesc();

        size_t block_size = (size_t)(vol_desc.block_length + vol_desc.padding * 2) * vol_desc.bits_per_sample
                            * vol_desc.samples_per_voxel / 8;
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
                .bits_per_sample = vol_desc.bits_per_sample,
                .samples_per_channel = vol_desc.samples_per_voxel,
                .vtex_block_length = (int)(vol_desc.block_length + vol_desc.padding * 2),
                .is_float = vol_desc.is_float, .exclusive = true
        };
        auto vtex_uid = render_gpu_mem_mgr_ref->RegisterGPUVTexMgr(vtex_info);
        gpu_vtex_mgr_ref = render_gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);
        gpu_pt_mgr_ref = gpu_vtex_mgr_ref->GetGPUPageTableMgrRef();

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

        LOG_DEBUG("Successfully Finish Render Resource Init...");
    }

    void registerCUDAGLInteropResource(){
        auto [window_w, window_h] = window->get_window_size();
        //注册cudaGL资源
        {

            cudaGL_interop.color_pbo.initialize_handle();
            cudaGL_interop.color_pbo.reinitialize_buffer_data(nullptr, window_w * window_h * sizeof(uint32_t), GL_DYNAMIC_COPY);
            CUB_CHECK(cudaGraphicsGLRegisterBuffer(&cudaGL_interop.cuda_frame_color_resc, cudaGL_interop.color_pbo.handle(), cudaGraphicsRegisterFlagsWriteDiscard));
            cudaGL_interop.depth_pbo.initialize_handle();
            cudaGL_interop.depth_pbo.reinitialize_buffer_data(nullptr, window_w * window_h * sizeof(float), GL_DYNAMIC_COPY);
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

        const std::string lod_vol_filename = "test_foot.lod.desc.json";

        loadLODVolumeData(lod_vol_filename);

        initVolDataResource();

        initRenderResource();


    }

    void initialize() override {
        registerCUDAGLInteropResource();

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
    }

    void frame() override {
        handle_events();
        auto [window_w, window_h] = window->get_window_size();
        // map cudaGL interop资源
        cudaGraphicsResource_t rescs[2] = {cudaGL_interop.cuda_frame_color_resc, cudaGL_interop.cuda_frame_depth_resc};
        CUB_CHECK(cudaGraphicsMapResources(2, rescs));
        void* color_mapping_ptr = nullptr;
        size_t color_mapping_size = 0;
        CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&color_mapping_ptr, &color_mapping_size, cudaGL_interop.cuda_frame_color_resc));
        assert(color_mapping_ptr && color_mapping_size == window_w * window_h * sizeof(uint32_t));
        void* depth_mapping_ptr = nullptr;
        size_t depth_mapping_size = 0;
        CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&depth_mapping_ptr, &depth_mapping_size, cudaGL_interop.cuda_frame_depth_resc));
        assert(depth_mapping_ptr && depth_mapping_size == window_w * window_h * sizeof(float));

        framebuffer->color = CUDABufferView2D<uint32_t>(color_mapping_ptr, {.pitch = window_w * sizeof(uint32_t), .xsize = (size_t) window_w, .ysize = (size_t) window_h});
        framebuffer->depth = CUDABufferView2D<float>(depth_mapping_ptr, {.pitch = window_w * sizeof(float), .xsize = (size_t) window_w, .ysize = (size_t) window_h});

        //=========================================
        // 设置体渲染参数并进行CUDA绘制
        static PerFrameParams per_frame_params{};
        updatePerFrameParams(per_frame_params);
        crt_vol_renderer->SetPerFrameParams(per_frame_params);
        crt_vol_renderer->Render(framebuffer);

        CUB_CHECK(cudaGraphicsUnmapResources(2, rescs));

        //=========================================

        //将cudaGL interop资源拷贝到OpenGL纹理中
        cudaGL_interop.color_pbo.bind();
        offscreen.color.set_texture_data<color4b>(offscreen.frame_width, offscreen.frame_height, nullptr);
        cudaGL_interop.color_pbo.unbind();

        cudaGL_interop.depth_pbo.bind();
        offscreen.color.set_texture_data<GLfloat>(offscreen.frame_width, offscreen.frame_height, nullptr);
        cudaGL_interop.depth_pbo.unbind();

        GL_EXPR(glFinish());



        frame_vol();


    }

    void destroy() override{

    }
private:
    void updatePerFrameParams(PerFrameParams& params){

    }
private:
    //将体绘制的结果画到ImGui窗口上
    void frame_vol(){
        ImGui::Begin("Volume Render Frame");

        ImGui::Image((void*)(intptr_t)(offscreen.color.handle()),
                ImVec2(offscreen.frame_width, offscreen.frame_height));


        ImGui::End();
    }
private:
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
        .size = {1200, 900}, .title = "VolAnnotater"
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
