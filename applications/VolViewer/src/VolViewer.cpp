#include "VolViewer.hpp"
#include "NeuronRenderer.hpp"

#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Algorithm/MarchingCube.hpp>
#include <Algorithm/Voxelization.hpp>
#include <Core/Renderer.hpp>
#include <Core/HashPageTable.hpp>
#include <Core/Distribute.hpp>
#include <Model/SWC.hpp>
#include <Model/Mesh.hpp>
#include <Model/SWC.hpp>
#include <IO/SWCIO.hpp>

#include <cuda_gl_interop.h>
#include <json.hpp>

#include <SDL.h>

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

//一个主机只有一个节点，一个节点可以对应多个渲染器，一个渲染器对应一个窗口，一般一个窗口对应一个显示屏
//window不用记录相机参数 虽然每个节点有单独的相机 但是只会使用root节点记录的相机参数
// sdl2 + dx11
class VolViewWindow {
  public:
    struct VolViewWindowCreateInfo{
        bool control_window = false;

        int window_xpos;
        int window_ypos;
        int window_width;
        int window_height;
        int gpu_index;
    };

    explicit VolViewWindow(const VolViewWindowCreateInfo& info){

        uint32_t flags = 0
                                     | SDL_WINDOW_BORDERLESS
            ;
        window = SDL_CreateWindow("null", info.window_xpos,  info.window_ypos,
                                          info.window_width, info.window_height, flags);

        renderer = SDL_CreateRenderer(window, info.gpu_index , SDL_RENDERER_ACCELERATED);

        auto device = SDL_RenderGetD3D11Device(renderer);
    }

    ~VolViewWindow(){
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
    }

    void Initialize(){



        LOG_DEBUG("VolViewWindow initialize successfully...");
    }

    //调用gl指令绘制
    void Draw(){

    }

    void Draw(Handle<FrameBuffer>){

    }

    //统一提交后swap buffer 用于保持整体画面的一致性
    void Commit(){

        SDL_RenderPresent(renderer);
    }

    void HandleEvents(){

    }

  private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Surface* surface;
    SDL_Texture* texture;
};

enum RendererType{
    VOL,
    MESH
};

struct GlobalSettings{
    inline static UnifiedRescUID fixed_host_mem_mgr_uid = 0;
    inline static bool async_render = true;
    inline static vec2i node_window_size{1920, 1080};
};

int GetWorldRank(){
    return DistributeMgr::GetInstance().GetWorldRank();
}

class VolViewerPrivate{
public:
    struct WindowRescPack{
      std::unique_ptr<VolViewWindow> window;
      Handle<RTVolumeRenderer> rt_renderer;

    };
    std::map<int, WindowRescPack> window_resc_mp;

    vutil::thread_group_t render_group;

    int node_window_count;

    VolViewerCreateInfo info;

    struct{
        Ref<HostMemMgr> host_mem_mgr_ref;

        Ref<GPUMemMgr> gpu_mem_mgr_ref;

    };

  public:


};

std::vector<std::string> LoadVolumeInfos(const std::string& filename){

}

VolViewer::VolViewer(const VolViewerCreateInfo &info) {
    SET_LOG_LEVEL_DEBUG

    _ = std::make_unique<VolViewerPrivate>();

    _->info = info;

    DistributeMgr::GetInstance().SetWorldRank(info.root_rank);

    _->node_window_count = info.window_infos.size();

    //todo check node_window_count <= node gpu count

    _->render_group.start(_->node_window_count);

    for(int i = 0; i < _->node_window_count; i++){
        int world_rank = DistributeMgr::GetInstance().GetWorldRank();
        int idx = (i << 16) | world_rank;
        auto& resc = _->window_resc_mp[idx];
        auto& window_info = info.window_infos.at(i);
        resc.window =  std::make_unique<VolViewWindow>(VolViewWindow::VolViewWindowCreateInfo{
            .window_xpos = window_info[0], .window_ypos = window_info[1],
            .window_width = window_info[2], .window_height = window_info[3],
            .gpu_index = i
        });
        resc.window->Initialize();
    }

    //check memory settings
#define MEMORY_RATIO 0.8

    auto host_mem_mgr_uid = ResourceMgr::GetInstance().RegisterResourceMgr({
        .type = ResourceMgr::Host,
        .MaxMemBytes = info.MaxHostMemGB << 30
    });

    auto host_mem_mgr_ref = ResourceMgr::GetInstance().GetHostRef(host_mem_mgr_uid).LockRef();

    // load volume
    auto paths = LoadVolumeInfos(info.resource_path);

    GridVolume::GridVolumeCreateInfo vol_info;
    vol_info.levels = paths.size();
    vol_info.host_mem_mgr_uid = host_mem_mgr_uid;
    for(int lod = 0; lod < paths.size(); lod++){
        vol_info.lod_vol_file_io[lod] = Handle<VolumeIOInterface>(ResourceType::Object, CreateVolumeFileByFileName(paths[lod]));
    }

    auto volume = NewHandle<GridVolume>(ResourceType::Object, vol_info);

    auto volume_desc = volume->GetDesc();

    size_t block_size = (size_t)(volume_desc.block_length + volume_desc.padding * 2) * volume_desc.bits_per_sample
                        * volume_desc.samples_per_voxel / 8;

    size_t fixed_mem_size = (info.MaxHostMemGB << 30) * MEMORY_RATIO;
    size_t block_num = fixed_mem_size / block_size;
    //create fixed host mem mgr

    FixedHostMemMgr::FixedHostMemMgrCreateInfo fixed_info;
    fixed_info.fixed_block_size = block_size;
    fixed_info.fixed_block_num = block_num;
    fixed_info.host_mem_mgr = host_mem_mgr_ref;

    auto fixed_host_mem_mgr_uid = host_mem_mgr_ref->RegisterFixedHostMemMgr(fixed_info);
    auto fixed_host_mem_mgr_ref = host_mem_mgr_ref->GetFixedHostMemMgrRef(fixed_host_mem_mgr_uid);
    // create renderer





    for(int i = 0; i < _->node_window_count; i++){
        auto gpu_mem_mgr_uid = ResourceMgr::GetInstance().RegisterResourceMgr({
            .type = ResourceMgr::Device,
            .MaxMemBytes = info.MaxGPUMemGB << 30,
            .DeviceIndex = i
        });
        auto gpu_mem_mgr_ref = ResourceMgr::GetInstance().GetGPURef(gpu_mem_mgr_uid).LockRef();
        RTVolumeRenderer::RTVolumeRendererCreateInfo rt_info{
            .host_mem_mgr = host_mem_mgr_ref,
            .gpu_mem_mgr = gpu_mem_mgr_ref,
            .use_shared_host_mem = true,
            .shared_fixed_host_mem_mgr_ref = fixed_host_mem_mgr_ref
        };
        auto renderer = NewHandle<RTVolumeRenderer>(ResourceType::Object, rt_info);

        renderer->BindGridVolume(volume);
    }

}

VolViewer::~VolViewer() {

}

void VolViewer::run()
{
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0){
        throw std::runtime_error("SDL init failed!");
    }

    bool exit = false;
    auto should_close = [this, &exit]{

        return exit;
    };

    auto process_input = [this, &exit]{
        static SDL_Event event;
        while(SDL_PollEvent(&event)){
            //imgui

            switch (event.type)
            {
                case SDL_QUIT : {
                    exit = true;
                } break;
                case SDL_KEYDOWN : {
                    switch(event.key.keysym.sym){
                    case SDLK_ESCAPE:{
                        exit = true;
                    } break;
                    case SDLK_a:{
                        LOG_DEBUG("key a down...");
                    } break;
                    }
                } break;
                case SDL_MOUSEBUTTONDOWN:{
                    LOG_DEBUG("mouse button down...");
                } break;
            }
        }
    };



    while(!should_close()){
        vutil::AutoTimer frame_timer("render frame");
        process_input();

        auto render_task = _->render_group.create_task();

        for(auto& [idx, resc] : _->window_resc_mp){
            auto& window = resc.window;
            render_task->enqueue_task([&]{
                window->Draw();
                window->Commit();
            });
        }
        _->render_group.submit(render_task);
    }

    SDL_Quit();
}
