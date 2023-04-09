//
// Created by wyz on 2023/4/6.
//

#include <Core/Renderer.hpp>
#include <json.hpp>

#include "VideoMaker.hpp"

using namespace viser;

namespace{
    enum LightType{
        Point,
        Spot,
        Direct
    };
}

struct VolRendererConfigParams{
    struct Settings{

    }settings;

    struct Camera{

    }camera;

    struct TransferFunc{

    }tf;

    struct Lights{

    }lights;

    struct Render{

    }render;

    struct Frame{
        Int2 resolution;

    }frame;

    struct Data{
        std::vector<std::string> lod_filenames;
    }data;

    struct Memory{
        size_t max_host_mem_bytes = 4ull << 30;
        size_t max_gpu_mem_bytes = 8ull << 30;
        uint32_t gpu_index = 0;
    }memory;

    struct Animation{
        int frame_count;
        std::vector<Mat4> transforms; // for camera
        // todo: other operations like change fov, near/far plane or field of depth
    }animation;
};

class ImageProcessor{
  public:
    ImageProcessor(int width, int height, CUDAContext ctx){

    }

    Handle<FrameBuffer> GetFrameBuffer(){
        return framebuffer;
    }

    void Flush(){

    }

  private:
    CUDAContext ctx;
    Handle<FrameBuffer> framebuffer;// use _color for float4
    vutil::image2d_t<Float4> radiance;
};

class RenderOutput{
  public:
    RenderOutput(int width, int height, CUDAContext ctx){

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

    void Close(){

    }
  private:
    CUDAContext ctx;
    std::unique_ptr<ImageProcessor> image_processor;
    std::unique_ptr<VideoMaker> video_maker;
};

void LoadFromJsonFile(VolRendererConfigParams& params, std::string_view filename){

}

void Run(Handle<PBVolumeRenderer> renderer, std::unique_ptr<RenderOutput> rop, const VolRendererConfigParams& params){
    // create and bind render resource


    // render loop

    for(int frame_index = 0; frame_index < params.animation.frame_count; frame_index++){


        renderer->Render(rop->GetFrameBuffer());

        rop->Flush();
    }

}

void SetupVolumeIO(GridVolume::GridVolumeCreateInfo& volInfo,
                   const VolRendererConfigParams::Data& data){
    if(data.lod_filenames.empty()){
        throw std::runtime_error("Empty lod filenames for loading volume data");
    }

    int lod = 0;
    for(auto& filename : data.lod_filenames){
        volInfo.lod_vol_file_io[lod++] = Handle<VolumeIOInterface>(ResourceType::Object,
                                                                   CreateVolumeFileByFileName(filename));
    }
    volInfo.levels = lod;
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