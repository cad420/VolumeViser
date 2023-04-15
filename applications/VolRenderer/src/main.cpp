//
// Created by wyz on 2023/4/6.
//

#include "Common.hpp"
#include "VideoMaker.hpp"






class ImageProcessor{
  public:
    ImageProcessor(int width, int height, CUDAContext ctx){

    }

    Handle<FrameBuffer> GetFrameBuffer(){
        return framebuffer;
    }

    void Flush(){

    }

    void SaveImage(std::string_view filename){
        if(vutil::ends_with(filename, ".hdr")){
            throw std::runtime_error("Can't save image for hdr format");
        }
        auto ret = radiance.map([](const Float4& r){
           return vutil::color4b(
                std::clamp<int>(r.x * 255, 0, 255),
                std::clamp<int>(r.y * 255, 0, 255),
                std::clamp<int>(r.z * 255, 0, 255),
                255);
        });
        vutil::save_rgba_to_png_file(filename.data(), ret.get_data());
    }

  private:
    CUDAContext ctx;
    Handle<FrameBuffer> framebuffer;// use _color for float4
    vutil::image2d_t<Float4> radiance;
};

class RenderOutput{
  public:
    RenderOutput(int width, int height, CUDAContext ctx){
        image_processor = std::make_unique<ImageProcessor>(width, height, ctx);


    }

    ~RenderOutput(){
        Close();
    }

    Handle<FrameBuffer> GetFrameBuffer(){
        return image_processor->GetFrameBuffer();
    }

    // indicates read buffer from framebuffer, should call before call on GetFrameBuffer()
    void Flush(){

    }

    void ExportFrame(std::string_view filename){
        image_processor->SaveImage(filename);
    }

    void Close(){

    }
  private:
    CUDAContext ctx;
    std::unique_ptr<ImageProcessor> image_processor;
    std::unique_ptr<VideoMaker> video_maker;
};



void Run(Handle<PBVolumeRenderer> renderer, std::unique_ptr<RenderOutput> rop, const VolRendererConfigParams& params){
    // create and bind render resource


    // render loop

    for(int frame_index = 0; frame_index < params.animation.frame_count; frame_index++){


        renderer->Render(rop->GetFrameBuffer());

        rop->Flush();
    }

}



void Run(const VolRendererConfigParams& params){
    // create system resource
    auto& resc_ins = ResourceMgr::GetInstance();
    auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                          .MaxMemBytes = params.memory.max_host_mem_bytes,
                                                          .DeviceIndex = -1});
    auto host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

    auto gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                      .MaxMemBytes = params.memory.max_gpu_mem_bytes,
                                                      .DeviceIndex = (int)params.memory.gpu_index});
    auto gpu_mem_mgr_ref = resc_ins.GetGPURef(gpu_resc_uid);

    // create renderer
    PBVolumeRenderer::PBVolumeRendererCreateInfo info{
        .host_mem_mgr_ref = host_mem_mgr_ref, //todo: just single thread ?
        .gpu_mem_mgr_ref = gpu_mem_mgr_ref
    };

    auto pb_renderer = NewHandle<PBVolumeRenderer>(ResourceType::Object, info);

    // load volume data
    GridVolume::GridVolumeCreateInfo vol_info;
    SetupVolumeIO(vol_info, params.data);
    auto volume = NewHandle<GridVolume>(ResourceType::Object, vol_info);

    pb_renderer->BindGridVolume(std::move(volume));

    // create render output resource
    std::unique_ptr<RenderOutput> rop;

    // run render task
    Run(std::move(pb_renderer), std::move(rop), params);
}

int main(int argc, char** argv){
    try{
        cmdline::parser cmd;

        cmd.add<std::string>("config-file", 'c', "config json filename");

        cmd.add<int>("log-level", 'l', "log level, 0 for debug, 1 for info, 2 for error(default)", false, 2);

        cmd.parse_check(argc, argv);

        auto filename = cmd.get<std::string>("config-file");

        auto log_level = cmd.get<int>("log-level");

        if(log_level == 0){
            SET_LOG_LEVEL_DEBUG
        }
        else if(log_level == 1){
            SET_LOG_LEVEL_INFO
        }
        else{
            SET_LOG_LEVEL_ERROR
        }

        VolRendererConfigParams params;

        LoadFromJsonFile(params, filename);

        Run(params);

    }
    catch (const std::exception& err)
    {
        LOG_ERROR("VolRenderer exit with exception: {}", err.what());
    }

    return 0;
}