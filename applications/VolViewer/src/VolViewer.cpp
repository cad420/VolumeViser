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

#include <fstream>


#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <wrl/client.h>
#include <d3d11_1.h> //use 11.0 is ok

//	d3d11.lib dxgi.lib d3dcompiler.lib dxguid.lib

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

template <class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

namespace {
    void glfw_close_callback(GLFWwindow * glfw_window);

    void glfw_focus_callback(GLFWwindow * glfw_window, int focus);

    void glfw_key_callback(GLFWwindow * glfw_window, int key, int scancode, int action, int mods);

    void glfw_char_callback(GLFWwindow * glfw_window, uint32_t c);

    void glfw_mouse_button_callback(GLFWwindow * glfw_window, int button, int action, int mods);

    void glfw_cursor_callback(GLFWwindow * glfw_window, double x, double y);

    void glfw_scroll_callback(GLFWwindow * glfw_window, double xoffset, double yoffset);
}


enum RendererType{
    VOL,
    MESH
};

class D3D11Exception : public std::runtime_error
{
  public:

    using runtime_error::runtime_error;
};

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

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

        glfw_window = glfwCreateWindow(info.window_width, info.window_height, "", nullptr, nullptr);
        glfwSetWindowPos(glfw_window, info.window_xpos, info.window_ypos);
        auto hwnd = glfwGetWin32Window(glfw_window);

        glfwSetWindowCloseCallback(glfw_window, glfw_close_callback);
        glfwSetWindowFocusCallback(glfw_window, glfw_focus_callback);
        glfwSetKeyCallback(glfw_window, glfw_key_callback);
        glfwSetCharCallback(glfw_window, glfw_char_callback);
        glfwSetMouseButtonCallback(glfw_window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(glfw_window, glfw_cursor_callback);
        glfwSetScrollCallback(glfw_window, glfw_scroll_callback);

        glfwShowWindow(glfw_window);

        //init d3d11
        const D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_1;
        uint32_t create_dev_flag = 0;
#ifdef VISER_DEBUG
        create_dev_flag |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        IDXGIFactory1* factory = nullptr;
        if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory),(void**)&factory))){
            throw D3D11Exception("Create DXGIFactory failed");
        }
        IDXGIAdapter* adapter = nullptr;
        if(factory->EnumAdapters(info.gpu_index, &adapter) == DXGI_ERROR_NOT_FOUND){
            throw D3D11Exception("Create DXGIAdapter failed");
        }
        DXGI_ADAPTER_DESC adapter_desc;
        adapter->GetDesc(&adapter_desc);



      if(FAILED(D3D11CreateDevice(
                adapter, D3D_DRIVER_TYPE_UNKNOWN , nullptr,
                create_dev_flag, &feature_level, 1,
                D3D11_SDK_VERSION,
                dev.GetAddressOf(), nullptr, dev_ctx.GetAddressOf()
                ))){
          throw D3D11Exception("Create d3d11 device failed");
      }

      DXGI_MODE_DESC back_buf_desc;
      back_buf_desc.Width = info.window_width;
      back_buf_desc.Height = info.window_height;
      back_buf_desc.RefreshRate.Numerator = 60;
      back_buf_desc.RefreshRate.Denominator = 1;
      back_buf_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      back_buf_desc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
      back_buf_desc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

      DXGI_SWAP_CHAIN_DESC swap_chain_desc;
      swap_chain_desc.BufferDesc = back_buf_desc;
      swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      swap_chain_desc.BufferCount = 1;
      swap_chain_desc.OutputWindow = hwnd;
      swap_chain_desc.Windowed = true;
      swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
      swap_chain_desc.Flags = 0;
      swap_chain_desc.SampleDesc.Count = 1;
      swap_chain_desc.SampleDesc.Quality = 0;

      if(FAILED(factory->CreateSwapChain(dev.Get(), &swap_chain_desc, swap_chain.GetAddressOf()))){
          throw D3D11Exception("Create swap chain failed");
      }
    }

    ~VolViewWindow(){

    }

    void Initialize(){



        LOG_DEBUG("VolViewWindow initialize successfully...");
    }

    void Draw(){

    }

    void Draw(ComPtr<ID3D11ShaderResourceView> frame){

    }

    void Draw(Handle<FrameBuffer>){

    }

    //统一提交后swap buffer 用于保持整体画面的一致性
    void Commit(){
        if(FAILED(swap_chain->Present(1, 0))){
            LOG_ERROR("Window Commit error");
        }
    }

    void HandleEvents(){

    }

  private:


    struct{
        GLFWwindow* glfw_window;

        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> dev_ctx;
        ComPtr<IDXGISwapChain> swap_chain;

        ComPtr<ID3D11RenderTargetView> rtv;
        ComPtr<ID3D11Texture2D> dsb;
        ComPtr<ID3D11DepthStencilView> dsv;
    };

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
        std::unique_ptr<NeuronRenderer> neuron_renderer;
    };
    std::map<int, WindowRescPack> window_resc_mp;

    vutil::thread_group_t render_group;

    int node_window_count;

    VolViewerCreateInfo info;

    struct{
        Ref<HostMemMgr> host_mem_mgr_ref;

        Ref<GPUMemMgr> gpu_mem_mgr_ref;

    };

    std::function<int(int)> get_window_idx;

  public:


};

std::vector<std::string> LoadVolumeInfos(const std::string& filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        LOG_ERROR("volume infos file open failed: " + filename);
        return {};
    }

    nlohmann::json j;
    in >> j;
    try{
        int levels = j.at("levels");
        std::vector<std::string> ret;
        for(int i = 0; i < levels; i++){
            auto name = "lod" + std::to_string(i);
            ret.push_back(j.at(name));
        }
        return ret;
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("volume infos format error: {}", err.what());
        return {};
    }
}

VolViewer::VolViewer(const VolViewerCreateInfo &info) {
    SET_LOG_LEVEL_DEBUG


    if(glfwInit() != GLFW_TRUE){
        throw std::runtime_error("GLFW init failed!");
    }


    _ = std::make_unique<VolViewerPrivate>();

    _->info = info;

    DistributeMgr::GetInstance().SetWorldRank(info.root_rank);

    _->node_window_count = info.window_infos.size();

    //todo check node_window_count <= node gpu count

    _->render_group.start(_->node_window_count);

    _->get_window_idx = [rank = DistributeMgr::GetInstance().GetWorldRank()](int i){
        return (i << 16) | rank;
    };

    for(int i = 0; i < _->node_window_count; i++){
        int idx = _->get_window_idx(i);
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
    block_size *= block_size * block_size;

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

//        renderer->BindGridVolume(volume);

        _->window_resc_mp.at(_->get_window_idx(i)).rt_renderer = std::move(renderer);
    }

}

VolViewer::~VolViewer() {

}



namespace{
    std::function<void(GLFWwindow*)> CloseCallback = [](GLFWwindow*){};
    std::function<void(GLFWwindow*, int)> FocusCallback = [](GLFWwindow*, int){};
    std::function<void(GLFWwindow*, int, int, int, int)> KeyCallback = [](GLFWwindow*, int, int, int, int){};
    std::function<void(GLFWwindow*, uint32_t)> CharCallback = [](GLFWwindow*, uint32_t){};
    std::function<void(GLFWwindow*, int, int, int)> MouseButtonCallback = [](GLFWwindow*, int, int, int){};
    std::function<void(GLFWwindow*, double, double)> CursorCallback = [](GLFWwindow*, double, double){};
    std::function<void(GLFWwindow*, double, double)> ScrollCallback = [](GLFWwindow*, double, double){};
}

namespace{
    void glfw_close_callback(GLFWwindow* glfw_window){
        CloseCallback(glfw_window);
    }

    void glfw_focus_callback(GLFWwindow* glfw_window,int focus){
        FocusCallback(glfw_window, focus);
    }

    void glfw_key_callback(GLFWwindow* glfw_window,int key,int scancode,int action,int mods){
        KeyCallback(glfw_window, key, scancode, action, mods);
    }

    void glfw_char_callback(GLFWwindow* glfw_window,uint32_t c){
        CharCallback(glfw_window, c);
    }

    void glfw_mouse_button_callback(GLFWwindow* glfw_window,int button,int action,int mods){
        MouseButtonCallback(glfw_window, button, action, mods);
    }

    void glfw_cursor_callback(GLFWwindow* glfw_window,double x,double y){
        CursorCallback(glfw_window, x, y);
    }

    void glfw_scroll_callback(GLFWwindow* glfw_window,double xoffset,double yoffset){
        ScrollCallback(glfw_window, xoffset, yoffset);
    }
}

void VolViewer::run()
{


    bool exit = false;
    auto should_close = [this, &exit]{

        return exit;
    };

    auto process_input = [this, &exit]{

        glfwPollEvents();

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

        _->render_group.wait_idle();
    }

}
