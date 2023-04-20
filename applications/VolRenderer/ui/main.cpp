//
// Created by wyz on 2023/4/10.
//

//只用于渲染单帧 不能调整任何参数 除了相机移动 可用于debug

#include "../src/Common.hpp"
#include "Algorithm/LevelOfDetailPolicy.hpp"

#include <cuda_gl_interop.h>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

#define TEST_TIME(f, desc) test_time([this](){f;}, desc);

class UI : public gl_app_t{
  public:
    using gl_app_t::gl_app_t;

    UI& init(int argc, char** argv){
       // todo resize window for frame size in config file

       cmdline::parser cmd;

       cmd.add<std::string>("config-file", 'c', "config json filename");

       cmd.parse_check(argc, argv);

       auto filename = cmd.get<std::string>("config-file");

       SET_LOG_LEVEL_DEBUG

       LoadFromJsonFile(params, filename);

        return *this;
    }

  private:

    void initialize() override {
//        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glDisable(GL_DEPTH_TEST));
        int w = WINDOW_WIDTH;
        int h = WINDOW_HEIGHT;

        //
        fbo.initialize_handle();
        rbo.initialize_handle();
        rbo.set_format(GL_DEPTH24_STENCIL8, w, h);
        fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, rbo);
        frame_color.initialize_handle();
        frame_color.initialize_texture(1, GL_RGBA32F, w, h);
        accumulate_color.initialize_handle();
        accumulate_color.initialize_texture(1, GL_RGBA32F, w, h);
        display_color.initialize_handle();
        display_color.initialize_texture(1, GL_RGBA8, w, h);

        accumulator = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/accumulate.comp")
            );

        post_processor = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/post_process.comp")
            );

        screen = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/quad.frag")
            );

        accu_params_buffer.initialize_handle();
        accu_params_buffer.reinitialize_buffer_data(&accu_params, GL_STATIC_DRAW);
        post_params_buffer.initialize_handle();
        post_params_buffer.reinitialize_buffer_data(&post_params, GL_STATIC_DRAW);

        quad_vao.initialize_handle();

        framebuffer = NewHandle<FrameBuffer>(ResourceType::Object);
        framebuffer->frame_width = w;
        framebuffer->frame_height = h;


        // renderer resource
        auto& resc_ins = ResourceMgr::GetInstance();
        auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                              .MaxMemBytes = params.memory.max_host_mem_bytes,
                                                              .DeviceIndex = -1});
        auto host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

        auto gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                          .MaxMemBytes = params.memory.max_gpu_mem_bytes,
                                                          .DeviceIndex = (int)params.memory.gpu_index});
        auto gpu_mem_mgr_ref = resc_ins.GetGPURef(gpu_resc_uid);

        gpu_mem_mgr_ref->_get_cuda_context()->set_ctx();

        PBVolumeRenderer::PBVolumeRendererCreateInfo pb_info{
            .host_mem_mgr_ref = host_mem_mgr_ref, //todo: just single thread ?
            .gpu_mem_mgr_ref = gpu_mem_mgr_ref
        };

        pb_vol_renderer = NewHandle<PBVolumeRenderer>(ResourceType::Object, pb_info);

        GridVolume::GridVolumeCreateInfo vol_info;
        SetupVolumeIO(vol_info, params.data);
        auto volume = NewHandle<GridVolume>(ResourceType::Object, vol_info);

        auto volume_desc = volume->GetDesc();

        pb_vol_renderer->BindGridVolume(std::move(volume));



        auto render_space = (std::min)({volume_desc.voxel_space.x, volume_desc.voxel_space.y, volume_desc.voxel_space.z});

        {
            RenderParams render_params{};
            ComputeUpBoundLOD(render_params.lod.leve_of_dist, render_space,
                              w, h, vutil::deg2rad(40.f));
            render_params.lod.updated = true;
            for(int i = 0; i < vol_info.levels; i++)
                render_params.lod.leve_of_dist.LOD[i] *= 1.5f;

            render_params.tf.updated = true;
            render_params.tf.tf_pts.pts.push_back({119.f/255.f, Float4(0.f, 0.f, 0.f, 0.f)});
            render_params.tf.tf_pts.pts.push_back({142.f/255.f, Float4(0.5f, 0.48443f, 0.36765f, 0.3412f)});
            render_params.tf.tf_pts.pts.push_back({238.f/255.f, Float4(0.853f, 0.338f, 0.092f, 0.73333f)});


            render_params.raycast.updated = true;
            render_params.raycast.ray_step = render_space * 0.5f;
            render_params.raycast.max_ray_dist = 6.f;

            render_params.other.updated = true;
            render_params.other.output_depth = false;

            pb_vol_renderer->SetRenderParams(render_params);
        }

        register_cuda_gl_resource();

        Float3 default_pos = {3.60977, 2.882, 9.3109};//8.06206f
        camera.set_position(default_pos);
        camera.set_perspective(40.f, 0.001f, 10.f);
        camera.set_direction(vutil::deg2rad(-90.f), 0.f);
        camera.set_move_speed(0.005);
        camera.set_view_rotation_speed(0.0003f);
        //global camera for get proj view
        camera.set_w_over_h((float)w / h);
    }

    void frame() override {
        before_render();

        handle_events();

        render_scene();

        render_ui();

        after_render();
    }

    void destroy() override {

    }

    void handle_events() override {
        gl_app_t::handle_events();
    }

  private:
    void register_cuda_gl_resource(){
        CUB_CHECK(cudaGraphicsGLRegisterImage(&cuda_frame_color_resc,
                                              frame_color.handle(),
                                              GL_TEXTURE_2D, 0));
    }


    void before_render(){
        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();


    }

    void clear_framebuffer(){
        fbo.bind();
        fbo.attach(GL_COLOR_ATTACHMENT0, accumulate_color);
        fbo.attach(GL_COLOR_ATTACHMENT1, display_color);
        GLenum cls[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        GL_EXPR(glDrawBuffers(2, cls));
        fbo.clear_buffer(GL_COLOR_BUFFER_BIT);
        fbo.unbind();

        cur_frame_index = 0;
        accu_params.frame_index = cur_frame_index;
        accu_params_buffer.set_buffer_data(&accu_params);
    }

    void update_per_frame(){
        auto view_not_changed = [&](const PerFrameParams& cur){
            bool pos_same = cur.cam_pos == per_frame_params.cam_pos;
            bool dir_same = cur.cam_dir == per_frame_params.cam_dir;
            bool fov_same = cur.fov     == per_frame_params.fov;
            return pos_same && dir_same && fov_same;
        };

        auto get_cur_params = [&](){
            PerFrameParams cur_params;
            cur_params.fov = vutil::deg2rad(camera.get_fov_deg());
            cur_params.cam_dir = camera.get_xyz_direction();
            cur_params.cam_pos = camera.get_position();
            cur_params.frame_width = WINDOW_WIDTH;
            cur_params.frame_height = WINDOW_HEIGHT;
            cur_params.frame_w_over_h = camera.get_w_over_h();
            cur_params.proj_view = camera.get_view_proj();
            static Float3 WorldUp = {0.f, 1.f, 0.f};
            cur_params.cam_right = vutil::cross(camera.get_xyz_direction(), WorldUp).normalized();
            cur_params.cam_up = vutil::cross(cur_params.cam_right, cur_params.cam_dir);
            return cur_params;
        };

        auto cur_params = get_cur_params();
        clear_history = !view_not_changed(cur_params);
        static bool first = true;
        if(first) clear_history = first = false;
        if(clear_history){
            clear_framebuffer();
        }
        pb_vol_renderer->SetPerFrameParams(cur_params);

        per_frame_params = cur_params;
    }

    void render_scene(){
        // mapping

        auto w = framebuffer->frame_width;
        auto h = framebuffer->frame_height;

        CUB_CHECK(cudaGraphicsMapResources(1, &cuda_frame_color_resc));
        cudaMipmappedArray_t mip_array;
        CUB_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&mip_array, cuda_frame_color_resc));
        cudaArray_t array;
        CUB_CHECK(cudaGetMipmappedArrayLevel(&array, mip_array, 0));
        framebuffer->_color = NewHandle<CUDASurface>(ResourceType::Buffer, array);

        update_per_frame();


//        pb_vol_renderer->Render(framebuffer);

        TEST_TIME(pb_vol_renderer->Render(framebuffer), "vol-render");


        cudaGraphicsUnmapResources(1, &cuda_frame_color_resc);

        auto x = (w + 16 - 1) / 16;
        auto y = (h + 16 - 1) / 16;

        fbo.bind();
        accumulator.bind();
        accu_params_buffer.bind(0);
        frame_color.bind_image(0, 0, GL_READ_ONLY, GL_RGBA32F);
        accumulate_color.bind_image(1, 0, GL_READ_WRITE, GL_RGBA32F);
        GL_EXPR(glDispatchCompute(x, y, 1));
        accumulator.unbind();

        post_processor.bind();
        post_params_buffer.bind(0);
        accumulate_color.bind_image(0, 0, GL_READ_ONLY, GL_RGBA32F);
        display_color.bind_image(1, 0, GL_WRITE_ONLY, GL_RGBA8);
        GL_EXPR(glDispatchCompute(x, y, 1));
        post_processor.unbind();

        framebuffer_t::bind_to_default();
        screen.bind();
        quad_vao.bind();
        display_color.bind(0);
        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        display_color.unbind(0);
        quad_vao.unbind();
        screen.unbind();
    }

    void render_ui(){
        if(ImGui::Begin("VolRenderer Settings")){
            ImGui::Text("Cur Frame Index: %d", cur_frame_index);

            for(auto& [desc, time] : time_records){
                ImGui::Text("%s cost time: %s ms", desc.c_str(), time.ms().fmt().c_str());
            }

        }

        ImGui::End();
    }

    void after_render(){
        cur_frame_index++;
//        if(clear_history) cur_frame_index = 0;

    }


    void test_time(std::function<void()> f, const std::string& desc){
        timer.start();
        f();
        timer.stop();
        time_records[desc] = timer.duration();
    }
  private:
    VolRendererConfigParams params;

    Timer timer;
    std::map<std::string, Duration> time_records;

    struct{
        int cur_frame_index = 0;
        bool clear_history;

    };

    struct{
        cudaGraphicsResource_t cuda_frame_color_resc = nullptr;// float4

    };

    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t frame_color; //rgba32f
        texture2d_t accumulate_color; // rgba32f
        texture2d_t display_color; // same with accumulate_color except different format with rgba8
    };

    PerFrameParams per_frame_params;

    Handle<FrameBuffer> framebuffer;

    Handle<PBVolumeRenderer> pb_vol_renderer;

    struct{
        struct alignas(16) AccumulateParams{
            int frame_index = 0;
        }accu_params;
        std140_uniform_block_buffer_t<AccumulateParams> accu_params_buffer;
        program_t accumulator;
        struct alignas(16) PostProcessParams{
            int tone_mapping = 0;
        }post_params;
        std140_uniform_block_buffer_t<PostProcessParams> post_params_buffer;
        program_t post_processor;
        program_t screen;
        vertex_array_t quad_vao;
    };
};

int main(int argc, char** argv){
    try{
        UI(window_desc_t{.size = {1200, 720}, .title = "VolRendererUI"}).init(argc, argv).run();
    }
    catch (const std::exception& err){
        std::cerr<< "VolRendererUI exited with error: " << err.what() << std::endl;
    }
}