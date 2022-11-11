#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Core/HashPageTable.hpp>
#include <Model/SWC.hpp>
#include "VolAnnotater.hpp"
#include "Common.hpp"
#include <cuda_gl_interop.h>
#include <json.hpp>
#include <fstream>
#include <Model/SWC.hpp>
#include <IO/SWCIO.hpp>
#include "SWCRenderer.hpp"
#include <Model/Mesh.hpp>

//标注系统的窗口绘制任务交给OpenGL，如果有多个显卡，其余的显卡可以用于网格重建任务
#define FOV 30.f
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

    // *.swc
    void loadSWCFile(const std::string& filename){
            swc_file.Open(filename, viser::SWCFile::Read);
            auto all_swc_pts = swc_file.GetAllPoints();
            swc_file.Close();

            for(auto& pt : all_swc_pts){
                swc.InsertNodeLeaf(pt);
            }

            swc.PrintInfo();

            assert(swc_renderer);

            auto lines = swc.PackLines();

            vol_tag_priv_data.patch_id = 0;
            vec3f _space = vec3f(volume_space_ratio * render_base_space);
            vec4f space = vec4f(_space.x, _space.y, _space.z, 1.f);
            for(auto& line : lines){
                std::unordered_map<vec4f, uint32_t> mp;
                uint32_t idx = 0;
                int n = line.size();
                assert(n % 2 == 0);
                std::vector<vec4f> vertices;
                std::vector<uint32_t> indices;
                for(int i = 0; i < n / 2; i++){
                    auto& a = line[i * 2];
                    auto& b = line[i * 2 + 1];
                    vec4f va = vec4f(a.x, a.y, a.z, a.radius);
                    vec4f vb = vec4f(b.x, b.y, b.z, b.radius);
                    va *= space;
                    vb *= space;
                    if(mp.count(va) == 0){
                        vertices.push_back(va);
                        mp[va] = idx++;
                    }
                    if(mp.count(vb) == 0){
                        vertices.push_back(vb);
                        mp[vb] = idx++;
                    }

                    indices.push_back(mp[va]);
                    indices.push_back(mp[vb]);
                }
                swc_renderer->InitLine(vertices, indices, vol_tag_priv_data.patch_id++);
            }
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


//        ComputeUpBoundLOD(lod, render_base_space, offscreen.frame_width, offscreen.frame_height,
//                          vutil::deg2rad(FOV));
        ComputeDefaultLOD(lod, (float)volume_desc.block_length * volume_space_ratio * render_base_space);
        lod.LOD[max_lod] = std::numeric_limits<float>::max();

        VolumeParams vol_params;
        vol_params.block_length = volume_desc.block_length;
        vol_params.padding = volume_desc.padding;
        vol_params.voxel_dim = volume_desc.shape;
        vol_params.bound = {
                {0.f, 0.f, 0.f},
                Float3(vol_params.voxel_dim) * render_base_space * volume_space_ratio
        };
        crt_vol_renderer->SetVolume(vol_params);

        RenderParams render_params;
        render_params.lod.updated = true;
        render_params.lod.leve_of_dist = lod;
        render_params.tf.updated = true;
        render_params.tf.tf_pts.pts[0.f] = Float4(0.f);
        render_params.tf.tf_pts.pts[0.32f] = Float4(0.f, 1.f, 0.5f, 0.f);
        render_params.tf.tf_pts.pts[0.4f] = Float4(1.f, 0.5f, 0.f, 1.f);
        render_params.tf.tf_pts.pts[0.96f] = Float4(1.f, 0.5f, 0.f, 1.f);
        render_params.tf.tf_pts.pts[1.f] = Float4(0.f);
        render_params.other.ray_step = render_base_space * 0.5f;
        render_params.other.max_ray_dist = 10.f;
        render_params.other.inv_tex_shape = Float3(1.f / create_info.vtex_shape_x,
                                                   1.f / create_info.vtex_shape_y,
                                                   1.f / create_info.vtex_shape_z);
        crt_vol_renderer->SetRenderParams(render_params);

        auto vtexs = gpu_vtex_mgr_ref->GetAllTextures();
        for(auto& [unit, handle] : vtexs){
            crt_vol_renderer->BindVTexture(handle, unit);
        }

        vol_tag_priv_data.query_info = NewGeneralHandle<CUDAHostBuffer>(RescAccess::Unique,
                                                          sizeof(float) * 8,
                                                          cub::memory_type::e_cu_host,
                                                          render_gpu_mem_mgr_ref->_get_cuda_context());
        vol_tag_priv_data.query_info_view = vol_tag_priv_data.query_info->view_1d<float>(sizeof(float)*8);

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

    void initSWCRendererResource(){
        SWCRenderer::SWCRendererCreateInfo info;
        swc_renderer = std::make_unique<SWCRenderer>(info);


    }

public:
    using gl_app_t::gl_app_t;

    void initialize(const VolAnnotater::VolAnnotaterCreateInfo& info){
        //tmp assert
        assert(info.render_compute_same_gpu);

        this->create_info = info;

        initGlobalResource();

        const std::string lod_vol_filename = "test_mouse.lod.desc.json";
        loadLODVolumeData(lod_vol_filename);

        initVolDataResource();

        initRenderResource();


    }

    void initialize() override {
        GL_EXPR(glClearColor(0.f, 0.f, 0.f, 0.f));
        GL_EXPR(glClearDepthf(1.f));
        GL_EXPR(glEnable(GL_DEPTH_TEST));

        //todo resize event
        framebuffer = NewGeneralHandle<FrameBuffer>(RescAccess::Unique);
        framebuffer->frame_width = offscreen.frame_width;
        framebuffer->frame_height = offscreen.frame_height;

        //todo
        offscreen.fbo.initialize_handle();
        offscreen.rbo.initialize_handle();
        offscreen.rbo.set_format(GL_DEPTH32F_STENCIL8, offscreen.frame_width, offscreen.frame_height);
        offscreen.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, offscreen.rbo);
        assert(offscreen.fbo.is_complete());

        offscreen.color.initialize_handle();
        offscreen.color.initialize_texture(1, GL_RGBA8, offscreen.frame_width, offscreen.frame_height);
        offscreen.depth.initialize_handle();
        offscreen.depth.initialize_texture(1, GL_R32F,  offscreen.frame_width, offscreen.frame_height);


        registerCUDAGLInteropResource();

        {
            debug.host_color.resize(framebuffer->frame_height * framebuffer->frame_width);
            debug.host_depth.resize(framebuffer->frame_height * framebuffer->frame_width);
        }
        auto default_pos = Float3(volume_desc.shape) * Float3(0.5f, 0.5f, 1.2f) * render_base_space * volume_space_ratio;
        camera.set_position(default_pos);
        camera.set_perspective(FOV, 0.01f, 10.f);
        camera.set_direction(vutil::deg2rad(-90.f), 0.f);
        camera.set_move_speed(render_base_space);
        camera.set_view_rotation_speed(0.001f);

        app_settings.camera_move_speed = render_base_space;

        initSWCRendererResource();

        const std::string swc_filename = "C:/Users/wyz/projects/VolumeViser/test_data/swc/N001.swc";
//        loadSWCFile(swc_filename);

        quad_vao.initialize_handle();

        view_to_proj_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("C:/Users/wyz/projects/VolumeViser/applications/VolAnnotater/asset/glsl/quad.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("C:/Users/wyz/projects/VolumeViser/applications/VolAnnotater/asset/glsl/view_to_proj_depth.frag")
                );
        v2p_params.fov = vutil::deg2rad(FOV);
        v2p_params.w_over_h = (float)offscreen.frame_width / offscreen.frame_height;
        v2p_params_buffer.initialize_handle();
        v2p_params_buffer.reinitialize_buffer_data(nullptr, GL_DYNAMIC_DRAW);
    }

    void frame() override {
        camera.set_w_over_h((float)offscreen.frame_width / offscreen.frame_height);
        handle_events();

        vol_tag_priv_data.clicked = ImGui::GetIO().MouseClicked[0];
        auto click_pos = ImGui::GetIO().MouseClickedPos[0];
        vol_tag_priv_data.clicked_pos = vec2i(click_pos.x - vol_renderer_priv_data.window_pos.x - 8,
                                              click_pos.y - vol_renderer_priv_data.window_pos.y - 27);

        update_vol_camera();
        if(app_settings.draw_volume){
            // map cudaGL interop资源
            cudaGraphicsResource_t rescs[2] = {cudaGL_interop.cuda_frame_color_resc,
                                               cudaGL_interop.cuda_frame_depth_resc};
            CUB_CHECK(cudaGraphicsMapResources(2, rescs));
            void *color_mapping_ptr = nullptr;
            size_t color_mapping_size = 0;
            CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&color_mapping_ptr, &color_mapping_size,
                                                           cudaGL_interop.cuda_frame_color_resc));
            assert(color_mapping_ptr &&
                   color_mapping_size == framebuffer->frame_width * framebuffer->frame_height * sizeof(uint32_t));
            void *depth_mapping_ptr = nullptr;
            size_t depth_mapping_size = 0;
            CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&depth_mapping_ptr, &depth_mapping_size,
                                                           cudaGL_interop.cuda_frame_depth_resc));
            assert(depth_mapping_ptr &&
                   depth_mapping_size == framebuffer->frame_width * framebuffer->frame_height * sizeof(float));

            framebuffer->color = CUDABufferView2D<uint32_t>(color_mapping_ptr,
                                                            {.pitch = framebuffer->frame_width * sizeof(uint32_t),
                                                                    .xsize = (size_t) framebuffer->frame_width,
                                                                    .ysize = (size_t) framebuffer->frame_height});
            framebuffer->depth = CUDABufferView2D<float>(depth_mapping_ptr,
                                                         {.pitch = framebuffer->frame_width * sizeof(float),
                                                                 .xsize = (size_t) framebuffer->frame_width,
                                                                 .ysize = (size_t) framebuffer->frame_height});

            //=========================================
            // 设置体渲染参数并进行CUDA绘制
            vol_render_timer.start();
            if(!mouse->is_cursor_visible())
                render_volume();
            else if(vol_tag_priv_data.clicked && app_settings.annotating){
                query_volume();
            }
            vol_render_timer.stop();
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

            //将view depth转换为proj depth并写入到rbo中
            offscreen.fbo.bind();
            GL_EXPR(glViewport(0, 0, offscreen.frame_width, offscreen.frame_height));
//            offscreen.fbo.attach(GL_COLOR_ATTACHMENT0, offscreen.color);
//            vutil::gl::framebuffer_t::clear_color_depth_buffer();
            vutil::gl::framebuffer_t::clear_buffer(GL_DEPTH_BUFFER_BIT);

            offscreen.depth.bind(0);
            view_to_proj_shader.bind();

            v2p_params.proj = camera.get_proj();
            v2p_params_buffer.set_buffer_data(&v2p_params);
            v2p_params_buffer.bind(0);

            quad_vao.bind();
//            GL_EXPR(glDisable(GL_DEPTH_TEST));
            GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
//            GL_EXPR(glEnable(GL_DEPTH_TEST));
            quad_vao.unbind();
            view_to_proj_shader.unbind();
            offscreen.fbo.unbind();

            render_swc();
        }
        //测试只画swc线
        if constexpr(false){
            offscreen.fbo.bind();
            GL_EXPR(glViewport(0, 0, offscreen.frame_width, offscreen.frame_height));
            offscreen.fbo.attach(GL_COLOR_ATTACHMENT0, offscreen.color);
            vutil::gl::framebuffer_t::clear_color_depth_buffer();


            swc_renderer->Draw(camera.get_view(), camera.get_proj());


            offscreen.fbo.unbind();
        }


        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();

        frame_app_settings();


        frame_vol();

        frame_timer();

        frame_debug();

        ImGui::ShowDemoWindow();
    }

    void destroy() override{

    }
private:
    void update_vol_camera(){
        vol_camera.width = offscreen.frame_width;
        vol_camera.height = offscreen.frame_height;
        vol_camera.fov = FOV;
        vol_camera.near = camera.get_near_z();
        vol_camera.far = camera.get_far_z();
        vol_camera.pos = camera.get_position();
        vol_camera.target = camera.get_xyz_direction() + vol_camera.pos;
        static Float3 world_up = Float3(0, 1, 0);
        Float3 right = cross(camera.get_xyz_direction(), world_up);
        vol_camera.up = cross(right, camera.get_xyz_direction());
    }
    //画swc线，可以选择和volume render的结果进行混合绘制
    void render_swc(){
        //在vol render后的depth buffer上画线
        offscreen.fbo.bind();
        GL_EXPR(glViewport(0, 0, offscreen.frame_width, offscreen.frame_height));
        offscreen.fbo.attach(GL_COLOR_ATTACHMENT0, offscreen.color);
        if(debug.clear_vol_color)
            vutil::gl::framebuffer_t::clear_color_depth_buffer();
        swc_renderer->Draw(camera.get_view(), camera.get_proj());

        offscreen.fbo.unbind();
    }
    //进行大规模体绘制
    void render_volume(){

        // 计算视锥体内的数据块
        update_vol_camera();
        auto camera_proj_view = vol_camera.GetProjViewMatrix();
        Frustum camera_view_frustum;
        ExtractFrustumFromMatrix(camera_proj_view, camera_view_frustum);
        auto& intersect_blocks = vol_renderer_priv_data.intersect_blocks;
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

        // 加载缺失的数据块到虚拟纹理中
        auto& blocks_info = vol_renderer_priv_data.blocks_info;
        blocks_info.clear();
        gpu_pt_mgr_ref->GetAndLock(intersect_blocks, blocks_info);

        //因为是单机同步的，不需要加任何锁
        // 暂时使用同步等待加载完数据块
        auto& host_blocks = vol_renderer_priv_data.host_blocks;//在循环结束会释放Handle
        host_blocks.clear();
        auto& missed_host_blocks = vol_renderer_priv_data.missed_host_blocks;
        missed_host_blocks.clear();
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
        auto& task_mp = vol_renderer_priv_data.task_mp;
        task_mp.clear();
        for(auto& missed_block : missed_host_blocks){
            int lod = missed_block.first.GetLOD();
            task_mp[lod].emplace_back(
                    [&, block = missed_block.first,
                     block_handle = std::move(missed_block.second)
                     ]()mutable{
                        volume->ReadBlock(block, *block_handle);
                        if constexpr (false){
                            std::vector<int> table(256, 0);
                            auto v = block_handle->view_1d<uint8_t>(256 * 256 * 256);
                            for(int i = 0; i < 1 << 24; i++){
                                table[v.at(i)]++;
                            }
                            for(int i = 0; i < 256; i++)
                                std::cout << "(" << i << "," << table[i] << ")  ";
                            std::cout << std::endl;
                        }
                        block_handle.SetUID(block.ToUnifiedRescUID());
                        host_blocks[block] = std::move(block_handle);
                        LOG_DEBUG("finish lod {} block ({}, {}, {}) loading...",
                                  block.GetLOD(), block.x, block.y, block.z);
                    });
        }
        auto& task_groups = vol_renderer_priv_data.task_groups;
        task_groups.clear();
        auto& lods = vol_renderer_priv_data.lods;
        lods.clear();
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


        crt_vol_renderer->BindPTBuffer(gpu_pt_mgr_ref->GetPageTable().GetHandle());


        //更新每一帧绘制的参数
        auto& per_frame_params = vol_renderer_priv_data.per_frame_params;
        updatePerFrameParams(per_frame_params);
        crt_vol_renderer->SetPerFrameParams(per_frame_params);


        crt_vol_renderer->Render(framebuffer);


        gpu_pt_mgr_ref->Release(intersect_blocks);
    }
    //查询点对应在体数据中的位置和体素值
    void query_volume(){
        LOG_DEBUG("start query volume");

        LOG_DEBUG("query pos: {} {}", vol_tag_priv_data.clicked_pos.x,
                  vol_tag_priv_data.clicked_pos.y);

        if(vol_tag_priv_data.clicked_pos.x < 0 || vol_tag_priv_data.clicked_pos.x >= offscreen.frame_width
        || vol_tag_priv_data.clicked_pos.y < 0 || vol_tag_priv_data.clicked_pos.y >= offscreen.frame_height){
            return;
        }

        crt_vol_renderer->Query(vol_tag_priv_data.clicked_pos.x,
                                vol_tag_priv_data.clicked_pos.y,
                                vol_tag_priv_data.query_info_view, 0);


        SWC::SWCPoint pt;
        pt.id = ++vol_tag_priv_data.swc_id;
        if(vol_tag_priv_data.swc_id == 1){
            vol_tag_priv_data.swc_patch_mp[pt.id] = vol_tag_priv_data.patch_id++;
        }
        pt.pid = vol_tag_priv_data.prev_swc_id == 0 ? -1 : vol_tag_priv_data.prev_swc_id;
        pt.x = vol_tag_priv_data.query_info_view.at(0);
        pt.y = vol_tag_priv_data.query_info_view.at(1);
        pt.z = vol_tag_priv_data.query_info_view.at(2);
        pt.radius = vol_tag_priv_data.query_info_view.at(3);
        swc.InsertNodeLeaf(pt);

        if(pt.id > 1){
            vec4f cur_vtx = vec4f(pt.x, pt.y, pt.z, pt.radius);
            auto prev_node = swc.GetNode(vol_tag_priv_data.prev_swc_id);
            vec4f prev_vtx = vec4f(prev_node.x, prev_node.y, prev_node.z, prev_node.radius);
            swc_renderer->AddLine(prev_vtx, cur_vtx,
                                  vol_tag_priv_data.swc_patch_mp.at(swc.GetNodeRoot(pt.id)));
        }
        vol_tag_priv_data.prev_swc_id = pt.id;
        LOG_DEBUG("finish query volume, pos: {} {} {}, depth: {}, color: {} {} {} {}",
                  vol_tag_priv_data.query_info_view.at(0),
                  vol_tag_priv_data.query_info_view.at(1),
                  vol_tag_priv_data.query_info_view.at(2),
                  vol_tag_priv_data.query_info_view.at(3),
                  vol_tag_priv_data.query_info_view.at(4),
                  vol_tag_priv_data.query_info_view.at(5),
                  vol_tag_priv_data.query_info_view.at(6),
                  vol_tag_priv_data.query_info_view.at(7));
    }

    //
    void run_mc_algo(){

    }
private:
    void updatePerFrameParams(PerFrameParams& params){
        params.frame_width = framebuffer->frame_width;
        params.frame_height = framebuffer->frame_height;
        params.frame_w_over_h = (float)framebuffer->frame_width / (float)framebuffer->frame_height;
        params.fov = vutil::deg2rad(FOV);
        params.cam_pos = vol_camera.pos;
        params.cam_dir = (vol_camera.target - vol_camera.pos).normalized();
        params.cam_up = vol_camera.up;
        params.cam_right = vutil::cross(params.cam_dir, params.cam_up).normalized();
        params.debug_mode = debug.debug_mode;
    }
private:
    //将体绘制的结果画到ImGui窗口上
    void frame_vol(){
        ImGui::Begin("Volume Render Frame", 0, ImGuiWindowFlags_NoResize);

        auto pos = ImGui::GetWindowPos();
        vol_renderer_priv_data.window_pos = vec2i(pos.x, pos.y);

        ImGui::Image((void*)(intptr_t)(offscreen.color.handle()),
                ImVec2(offscreen.frame_width, offscreen.frame_height));

        ImGui::End();
    }
    void frame_swc_test(){

    }
    void frame_swc_settings(){
        ImGui::Begin("SWC", 0, ImGuiWindowFlags_NoResize);

        ImGui::Selectable("SWC Files");

        ImGui::End();
    }

    void frame_timer(){
        ImGui::Begin("Time cost");

        ImGui::Text("%s", vol_render_timer.duration_str().c_str());

        ImGui::Text("Volume Bound: %f %f %f, %f %f %f", volume_box.low.x, volume_box.low.y, volume_box.low.z,
                    volume_box.high.x, volume_box.high.y, volume_box.high.z);

        ImGui::Text("Camera Pos: %f %f %f", vol_camera.pos.x, vol_camera.pos.y, vol_camera.pos.z);
        ImGui::Text("Camera Target: %f %f %f", vol_camera.target.x, vol_camera.target.y, vol_camera.target.z);
        ImGui::Text("Camera Dir: %f %f %f", vol_renderer_priv_data.per_frame_params.cam_dir.x,
                    vol_renderer_priv_data.per_frame_params.cam_dir.y,
                    vol_renderer_priv_data.per_frame_params.cam_dir.z);


        auto clicked_pos = ImGui::GetIO().MouseClickedPos[0];
        ImGui::Text("Vol Render Window Pos: %d %d", vol_renderer_priv_data.window_pos.x, vol_renderer_priv_data.window_pos.y);
        ImGui::Text("Left Mouse Button Clicked Pos: %f %f", clicked_pos.x, clicked_pos.y);

        if(ImGui::RadioButton("debug normal", debug.debug_mode == debug_mode_normal)){
            debug.debug_mode = debug_mode_normal;
        }
        if(ImGui::RadioButton("debug entry pos", debug.debug_mode == debug_mode_entry_pos)){
            debug.debug_mode = debug_mode_entry_pos;
        }
        if(ImGui::RadioButton("debug exit pos", debug.debug_mode == debug_mode_exit_pos)){
            debug.debug_mode = debug_mode_exit_pos;
        }
        if(ImGui::RadioButton("debug scalar", debug.debug_mode == debug_mode_scalar)){
            debug.debug_mode = debug_mode_scalar;
        }
        if(ImGui::RadioButton("debug no light shading", debug.debug_mode == debug_mode_no_light_shading)){
            debug.debug_mode = debug_mode_no_light_shading;
        }

        ImGui::Checkbox("Clear Vol Color", &debug.clear_vol_color);

        ImGui::End();
    }

    void frame_debug(){
        ImGui::Begin("Debug");

        if(ImGui::BeginTable("intersect blocks", 3)){
            ImGui::TableSetupColumn("BlockUID");
            ImGui::TableSetupColumn("TexCoord");
            ImGui::TableSetupColumn("Status");
            ImGui::TableHeadersRow();

            for(auto& b : vol_renderer_priv_data.blocks_info){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%d %d %d, %d", b.first.x, b.first.y, b.first.z, b.first.GetLOD());
                ImGui::TableNextColumn();
                ImGui::Text("%d, %d %d %d", b.second.tid, b.second.sx, b.second.sy, b.second.sz);
                ImGui::TableNextColumn();
                ImGui::Text("%s", b.second.Missed() ? "missed" : "existed");
            }

            ImGui::EndTable();

        }


        if(ImGui::BeginTable("Level of Detail", 2)){
            ImGui::TableSetupColumn("LOD");
            ImGui::TableSetupColumn("Dist");
            ImGui::TableHeadersRow();

            for(int l = 0; l <= max_lod; l++){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("lod %d", l);
                ImGui::TableNextColumn();
                ImGui::Text("%.5f", lod.LOD[l]);
            }

            ImGui::EndTable();
        }


        ImGui::End();
    }
    void frame_app_settings(){
        ImGui::Begin("Application Settings");

        ImGui::Checkbox("Draw Volume", &app_settings.draw_volume);

        ImGui::Checkbox("Annotate", &app_settings.annotating);

        if(ImGui::InputFloat("Camera Move Speed", &app_settings.camera_move_speed)){
            camera.set_move_speed(app_settings.camera_move_speed);
        }

        ImGui::End();
    }
private:
    enum debug_mode_enum : int{
        debug_mode_normal = 0,
        debug_mode_entry_pos = 1,
        debug_mode_exit_pos = 2,
        debug_mode_scalar = 3,
        debug_mode_no_light_shading = 4,
    };
    struct{
        std::vector<uint32_t> host_color;
        std::vector<float> host_depth;
        int debug_mode = 0;
        bool clear_vol_color = false;
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
    //mc和render公用一个gpu和vtex，因此在跑mc的时候不能进行渲染
    //由于跑mc可能替换了数据块，因此渲染的时候需要重新导入，而且两者同一数据块的数据是完全不同的
    Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
    Ref<GPUPageTableMgr> gpu_pt_mgr_ref;

    vutil::thread_group_t thread_group;

    //由OpenGL资源注册得到的CUDA资源句柄

    Handle<FrameBuffer> framebuffer;

    Timer vol_render_timer;

    Handle<CRTVolumeRenderer> crt_vol_renderer;

    struct{
        vec2i window_pos;
        std::vector<GridVolume::BlockUID> intersect_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;//在循环结束会释放Handle
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        std::map<int, std::vector<std::function<void()>>> task_mp;
        std::map<int, vutil::task_group_handle_t> task_groups;
        std::vector<int> lods;
        PerFrameParams per_frame_params;

    }vol_renderer_priv_data;

    viser::Camera vol_camera;

    float render_base_space = 0.00128f;
    BoundingBox3D volume_box;
    GridVolume::GridVolumeDesc volume_desc;
    int max_lod;
    viser::LevelOfDist lod;
    Float3 volume_space_ratio;

    vertex_array_t quad_vao;

    program_t view_to_proj_shader;
    struct alignas(16) ViewToProjParams{
        mat4 proj;
        float fov;
        float w_over_h;
    }v2p_params;
    std140_uniform_block_buffer_t<ViewToProjParams> v2p_params_buffer;


    struct{
        std::unordered_map<GridVolume::BlockUID, Handle<Mesh>> block_mesh_mp;
        //网格的patch和swc的patch是不一样的
        //swc的patch是指一条完整的神经元
        //网格的patch是指一个block对应的mesh
        std::unordered_map<GridVolume::BlockUID, size_t> block_patch_mp;

    }swc2mesh_priv_data;

    //标注相关
    SWCFile swc_file;
    SWC swc;
    std::unique_ptr<SWCRenderer> swc_renderer;
    struct{
        size_t patch_id = 0;
        //这里需要假设神经元是树型的，即只有一个根节点，但一个swc文件可以有多棵树，但不能是图
        //记录每条神经元在渲染器中对应的patch id，一条神经元由root代表
        std::unordered_map<SWC::SWCPointKey, size_t> swc_patch_mp;

        int swc_id = 0;
        SWC::SWCPointKey prev_swc_id = -1;


        Handle<CUDAHostBuffer> query_info;
        CUDABufferView1D<float> query_info_view;
        bool clicked;
        vec2i clicked_pos;


    }vol_tag_priv_data;

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

    //系统的控制参数
    struct{
        bool draw_volume = true;
        bool annotating = false;
        float camera_move_speed;
    }app_settings;
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
