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

#include "VolViewer.hpp"
#include "NeuronRenderer.hpp"

#include <cuda_d3d11_interop.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_dx11.h>

#define SHARED_FIXED_HOST_MEMORY


//	d3d11.lib dxgi.lib d3dcompiler.lib dxguid.lib

using namespace viser;
using namespace vutil;
using namespace vutil::gl;



namespace {
    void glfw_close_callback(GLFWwindow * glfw_window);

    void glfw_focus_callback(GLFWwindow * glfw_window, int focus);

    void glfw_key_callback(GLFWwindow * glfw_window, int key, int scancode, int action, int mods);

    void glfw_char_callback(GLFWwindow * glfw_window, uint32_t c);

    void glfw_mouse_button_callback(GLFWwindow * glfw_window, int button, int action, int mods);

    void glfw_cursor_callback(GLFWwindow * glfw_window, double x, double y);

    void glfw_scroll_callback(GLFWwindow * glfw_window, double xoffset, double yoffset);
}


enum class VolViewRenderType{
    VOL = 0,
    MESH = 1
};

class D3D11Exception : public std::runtime_error
{
  public:

    using runtime_error::runtime_error;
};

//一个主机只有一个节点，一个节点可以对应多个渲染器，一个渲染器对应一个窗口，一般一个窗口对应一个显示屏
//window不用记录相机参数 虽然每个节点有单独的相机 但是只会使用root节点记录的相机参数
// sdl2 + dx11

GLFWwindow* root_window = nullptr;

class VolViewWindow {
  public:
    struct VolViewWindowCreateInfo{
        bool control_window = false;

        int window_xpos;
        int window_ypos;
        int window_width;
        int window_height;
        int frame_width;
        int frame_height;
        int gpu_index;
        bool root = false;

        Ref<GPUMemMgr> gpu_mem_mgr_ref;
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

      //create rtv
      ComPtr<ID3D11Texture2D> background_buffer;
      if(FAILED(swap_chain->GetBuffer(0, IID_PPV_ARGS(background_buffer.GetAddressOf())))){
          throw D3D11Exception("Get dxgi swap chain buffer failed");
      }

      if(FAILED(dev->CreateRenderTargetView(background_buffer.Get(), nullptr, rtv.GetAddressOf()))){
          throw D3D11Exception("Create render target view failed");
      }

      //create depth-stencil buffer and view
      D3D11_TEXTURE2D_DESC dsb_desc;
      dsb_desc.ArraySize = 1;
      dsb_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
      dsb_desc.CPUAccessFlags = 0;
      dsb_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
      dsb_desc.Width = info.window_width;
      dsb_desc.Height = info.window_height;
      dsb_desc.MipLevels = 1;
      dsb_desc.MiscFlags = 0;
      dsb_desc.SampleDesc.Count = 1;
      dsb_desc.SampleDesc.Quality = 0;
      dsb_desc.Usage = D3D11_USAGE_DEFAULT;

      if(FAILED(dev->CreateTexture2D(&dsb_desc, nullptr, dsb.GetAddressOf()))){
          throw D3D11Exception("Create d3d11 depth stencil buffer failed");
      }

      D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc;
      dsv_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
      dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
      dsv_desc.Flags = 0;
      dsv_desc.Texture2D.MipSlice = 0;

      if(FAILED(dev->CreateDepthStencilView(dsb.Get(), &dsv_desc, dsv.GetAddressOf()))){
          throw D3D11Exception("Create d3d11 depth stencil view failed");
      }



      D3D11_VIEWPORT vp;
      vp.TopLeftX = 0;
      vp.TopLeftY = 0;
      vp.Width = info.window_width;
      vp.Height = info.window_height;
      vp.MaxDepth = 1.f;
      vp.MinDepth = 0.f;
      dev_ctx->RSSetViewports(1, &vp);

      dev_ctx->OMSetRenderTargets(1, rtv.GetAddressOf(), dsv.Get());

      // cuda d3d interop
      gpu_mem_mgr_ref = std::move(info.gpu_mem_mgr_ref);
      auto __ = gpu_mem_mgr_ref._get_ptr()->_get_cuda_context()->temp_ctx();
      D3D11_TEXTURE2D_DESC osf_desc;
      osf_desc.ArraySize = 1;
      osf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
      osf_desc.CPUAccessFlags = 0;
      osf_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      osf_desc.Width = info.frame_width;
      osf_desc.Height = info.frame_height;
      osf_desc.MipLevels = 1;
      osf_desc.MiscFlags = 0;
      osf_desc.SampleDesc.Count = 1;
      osf_desc.SampleDesc.Quality = 0;
      osf_desc.Usage = D3D11_USAGE_DEFAULT;
      if(FAILED(dev->CreateTexture2D(&osf_desc, nullptr, offscreen_frame.GetAddressOf()))){
          throw D3D11Exception("Create d3d11 offscreen frame texture2d failed");
      }

//      D3D11_BUFFER_DESC osb_desc;
//      osb_desc.Usage = D3D11_USAGE_DEFAULT;
//      osb_desc.CPUAccessFlags = 0;
//      osb_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
//      osb_desc.MiscFlags = 0;
//      osb_desc.ByteWidth = (size_t)info.window_width * info.window_height * sizeof(uint32_t);
//      osb_desc.StructureByteStride = 0;
//      if(FAILED(dev->CreateBuffer(&osb_desc, nullptr, offscreen_buffer.GetAddressOf()))){
//          throw D3D11Exception("Create d3d11 offscreen frame buffer failed");
//      }

      CUB_CHECK(cudaGraphicsD3D11RegisterResource(&cuda_frame_color_resc, offscreen_frame.Get(), cudaGraphicsRegisterFlagsSurfaceLoadStore));

      D3D11_SHADER_RESOURCE_VIEW_DESC osf_srv_desc;
      osf_srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      osf_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
      osf_srv_desc.Texture2D.MostDetailedMip = 0;
      osf_srv_desc.Texture2D.MipLevels = 1;

      if(FAILED(dev->CreateShaderResourceView(offscreen_frame.Get(), &osf_srv_desc, osf_srv.GetAddressOf()))){
          throw D3D11Exception("Create d3d11 offscreen frame shader resource view failed");
      }

      //create shader
      ComPtr<ID3DBlob> blob;
      auto compile_shader = [&]( const WCHAR* filename, LPCSTR entryPoint, LPCSTR shaderModel){
          ID3DBlob *errorBlob = nullptr;
          DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
          // 设置 D3DCOMPILE_DEBUG 标志用于获取着色器调试信息。该标志可以提升调试体验，
          // 但仍然允许着色器进行优化操作
          dwShaderFlags |= D3DCOMPILE_DEBUG;

          // 在Debug环境下禁用优化以避免出现一些不合理的情况
          dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
          D3DCompileFromFile(filename, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, shaderModel, dwShaderFlags,
                             0, blob.ReleaseAndGetAddressOf(), &errorBlob);

          if (errorBlob)
          {
              std::cerr<< (reinterpret_cast<const char *>(errorBlob->GetBufferPointer())) << std::endl;
          }
      };

      compile_shader(L"asset/hlsl/quad.hlsl", "VSMain", "vs_5_0");
      dev->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, quad_vs_shader.GetAddressOf());

      compile_shader(L"asset/hlsl/quad.hlsl", "PSMain", "ps_5_0");
      dev->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, quad_ps_shader.GetAddressOf());

      //imgui
      if(info.root){
          ImGui::CreateContext();
          ImGui_ImplGlfw_InitForOther(glfw_window, false);
          ImGui_ImplDX11_Init(dev.Get(), dev_ctx.Get());
      }

      // others
      window_w = info.window_width;
      window_h = info.window_height;

      root = info.root;

      vol_framebuffer = NewHandle<FrameBuffer>(ResourceType::Buffer);
      vol_framebuffer->frame_width = info.frame_width;
      vol_framebuffer->frame_height = info.frame_height;

      window_framebuffer_size = (size_t)info.frame_width * info.frame_height * 4;
      if(root){
          root_window = glfw_window;
      }
    }

    ~VolViewWindow(){
        CUB_CHECK(cudaGraphicsUnregisterResource(cuda_frame_color_resc));
    }

    void Initialize(){



        LOG_DEBUG("VolViewWindow initialize successfully...");
    }

    void PreRender(){
        static Float4 clear_color = {0.f, 0.f, 0.f, 0.f};

        dev_ctx->ClearRenderTargetView(rtv.Get(), &clear_color.x);

        dev_ctx->ClearDepthStencilView(dsv.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.f, 0);
    }

    void Draw(){

    }

    void Draw(VolViewRenderType type, bool onlyImGui = false){
        //std::functional is better hhh
        static void(VolViewWindow::*draw_funcs[2])()  = {&VolViewWindow::DrawVol, &VolViewWindow::DrawMesh};
        assert(type == VolViewRenderType::VOL || type == VolViewRenderType::MESH);
        if(!onlyImGui)
        (this->*draw_funcs[static_cast<int>(type)])();

        if(onlyImGui){
            Draw(osf_srv);
        }

        if(root && DistributeMgr::GetInstance().IsRoot()){
            DrawImGui();
        }

    }

    void DrawImGui(){
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Settings");
            if(ImGui::TreeNode("Vol Render Setting")){
                if(ImGui::TreeNode("PerFrameParams")){
                    ImGui::Text("Camera Pos: %.5f %.5f %.5f", per_frame_params->cam_pos.x,
                                per_frame_params->cam_pos.y, per_frame_params->cam_pos.z);
                    ImGui::Text("Camera Dir: %.5f %.5f %.5f", per_frame_params->cam_dir.x,
                                per_frame_params->cam_dir.y, per_frame_params->cam_dir.z);
                    ImGui::TreePop();
                }
                if(ImGui::TreeNode("Raycast")){
                    bool update = false;
                    update |= ImGui::InputFloat("Ray Step", &render_params->raycast.ray_step,
                                                0.00016f, 0.00016f, "%.5f");
                    update |= ImGui::InputFloat("Max Ray Dist", &render_params->raycast.max_ray_dist,
                                                0.1f, 0.1f);

                    render_params->raycast.updated = update;

                    ImGui::TreePop();
                }
                if(ImGui::TreeNode("TransferFunc")){
                    bool tf_update = false;

                    static std::map<int, Float4> pt_mp;

                    static Float3 color;
                    static bool selected_pt = false;
                    static int sel_pos;
                    if(selected_pt){
                        color = pt_mp.at(sel_pos).xyz();
                    }
                    if(ImGui::ColorEdit3("Point Color(RGBA)", &color.x)){
                        if(selected_pt){
                            auto& c = pt_mp.at(sel_pos);
                            c.x = color.x;
                            c.y = color.y;
                            c.z = color.z;

                            tf_update = true;
                        }
                    }



                    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
                    const int ysize = 255;
                    canvas_sz.y = ysize;
                    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
                    ImGui::InvisibleButton("tf", canvas_sz);


                    ImGuiIO& io = ImGui::GetIO();
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();

                    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
                    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(200, 200, 200, 255));

                    const bool is_hovered = ImGui::IsItemHovered(); // Hovered
                    const bool is_active = ImGui::IsItemActive();   // Held
                    const ImVec2 origin(canvas_p0.x, canvas_p0.y); // Lock scrolled origin
                    const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);
                    const ImVec2 tf_origin(canvas_p0.x, canvas_p0.y + canvas_sz.y);

                    bool check_add = false;
                    if(is_active && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)){
                        check_add = true;
                    }

                    auto canvas_y_to_alpha = [&](float y){
                        return (ysize - y) / float(ysize);
                    };
                    auto alpha_to_canvas_y = [&](float alpha){
                        return ysize - alpha * ysize;
                    };



                    if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                        bool pick = false;
                        for(auto& [x_pos, color] : pt_mp){
                            if(std::abs(x_pos - mouse_pos_in_canvas.x) < 5
                                && std::abs(alpha_to_canvas_y(color.w) - mouse_pos_in_canvas.y) < 5){
                                selected_pt = true;
                                sel_pos = x_pos;
                                pick = true;
                                break;
                            }
                        }
                        if(!pick) selected_pt = false;
                    }



                    if(!selected_pt && check_add){
                        auto it = pt_mp.upper_bound(mouse_pos_in_canvas.x);
                        Float4 rgba;
                        rgba.w = canvas_y_to_alpha(mouse_pos_in_canvas.y);
                        if(it == pt_mp.end()){
                            auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                            if(itt == pt_mp.begin()){
                                rgba.x = rgba.y = rgba.z = 0.f;
                            }
                            else{
                                itt = std::prev(itt);
                                rgba.x = itt->second.x;
                                rgba.y = itt->second.y;
                                rgba.z = itt->second.z;
                            }
                        }
                        else{
                            auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                            if(itt == pt_mp.begin()){
                                rgba.x = it->second.x;
                                rgba.y = it->second.y;
                                rgba.z = it->second.z;
                            }
                            else{
                                itt = std::prev(itt);
                                float u = (mouse_pos_in_canvas.x - itt->first) / (float)(it->first - itt->first);
                                rgba.x = itt->second.x * (1.f - u) + it->second.x * u;
                                rgba.y = itt->second.y * (1.f - u) + it->second.y * u;
                                rgba.z = itt->second.z * (1.f - u) + it->second.z * u;
                            }
                        }
                        pt_mp[mouse_pos_in_canvas.x] = rgba;
                        selected_pt = true;
                        sel_pos = mouse_pos_in_canvas.x;
                        tf_update = true;
                    }


                    //add
                    if(is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)
                        && selected_pt){
                        int nx = sel_pos + io.MouseDelta.x;
                        auto c = pt_mp.at(sel_pos);
                        int ny = alpha_to_canvas_y(c.w) + io.MouseDelta.y;
                        //                    LOG_DEBUG("ny : {}, delta y: {}", ny, io.MouseDelta.y);
                        ny = (std::min)(ny, ysize);
                        ny = (std::max)(ny, 0);
                        if(nx == sel_pos || pt_mp.count(nx) == 0){
                            c.w = canvas_y_to_alpha(ny);
                            pt_mp.erase(sel_pos);
                            sel_pos = nx;
                            pt_mp[nx] = c;
                            tf_update = true;
                        }
                    }

                    //delete
                    if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Right)
                        && selected_pt){
                        selected_pt = false;
                        pt_mp.erase(sel_pos);
                        tf_update = true;
                    }

                    draw_list->PushClipRect(canvas_p0, canvas_p1, true);
                    bool first = true;
                    ImVec2 prev;
                    if(!pt_mp.empty()){
                        auto it = pt_mp.begin();
                        ImVec2 p = ImVec2(it->first + origin.x, alpha_to_canvas_y(it->second.w) + origin.y);
                        draw_list->AddLine(ImVec2(origin.x, p.y), p, IM_COL32(0, 0, 0, 255));
                        auto itt = std::prev(pt_mp.end());
                        p = ImVec2(itt->first + origin.x, alpha_to_canvas_y(itt->second.w) + origin.y);
                        draw_list->AddLine(p, ImVec2(origin.x + canvas_sz.x, p.y), IM_COL32(0, 0, 0, 255));
                    }
                    for(auto& [x, c] : pt_mp){
                        ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                        if(first){
                            first = false;
                        }
                        else{
                            draw_list->AddLine(prev, cur, IM_COL32(0, 0, 0, 255));
                        }
                        prev = cur;
                    }
                    for(auto& [x, c] : pt_mp){
                        ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                        draw_list->AddCircleFilled(cur, 5.f,
                                                   IM_COL32(int(c.x * 255), int(c.y * 255), int(c.z * 255), 255));
                        if(x == sel_pos && selected_pt){
                            draw_list->AddCircle(cur, 6.f, IM_COL32(255, 127, 0, 255), 0, 2.f);
                        }
                    }
                    draw_list->PopClipRect();

                    ImGui::TreePop();

                    if(tf_update){
                        std::vector<std::pair<float, Float4>> pts;

                        for(auto& [x, c] : pt_mp){
                            pts.emplace_back((float)x / (float)canvas_sz.x, c);
                        }

                        render_params->tf.tf_pts.pts = std::move(pts);

                        render_params->tf.updated = true;
                    }
                }

                ImGui::TreePop();
            }



            ImGui::End();
        }
        ImGui::EndFrame();
        ImGui::Render();
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    }

    void DrawMesh(){


    }

    void DrawVol(){
        auto __ = gpu_mem_mgr_ref._get_ptr()->_get_cuda_context()->temp_ctx();
        CUB_CHECK(cudaGraphicsResourceSetMapFlags(cuda_frame_color_resc, cudaGraphicsMapFlagsWriteDiscard));
        CUB_CHECK(cudaGraphicsMapResources(1, &cuda_frame_color_resc));
        if constexpr(false){
            void *color_mapping_ptr = nullptr;
            size_t color_mapping_size = 0;
            CUB_CHECK(cudaGraphicsResourceGetMappedPointer(&color_mapping_ptr, &color_mapping_size, cuda_frame_color_resc));
            assert(color_mapping_ptr && color_mapping_size == window_framebuffer_size);

            vol_framebuffer->color = CUDABufferView2D<uint32_t>(
                color_mapping_ptr, {.pitch = window_w * sizeof(uint32_t), .xsize = window_w, .ysize = window_h});
        }
        {
            cudaMipmappedArray_t mip_array;
            CUB_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&mip_array, cuda_frame_color_resc));
            cudaArray_t array;
            CUB_CHECK(cudaGetMipmappedArrayLevel(&array, mip_array, 0));
            vol_framebuffer->_color = NewHandle<CUDASurface>(ResourceType::Buffer, array);
        }

        rt_renderer->Render(vol_framebuffer);

        CUB_CHECK(cudaGraphicsUnmapResources(1, &cuda_frame_color_resc));

        Draw(osf_srv);


    }


    //统一提交后swap buffer 用于保持整体画面的一致性
    void Commit(){
        if(FAILED(swap_chain->Present(0, 0))){
            LOG_ERROR("Window Commit error");
        }
    }


    auto GetD3D11Device() const {
        return dev.Get();
    }

    struct{
        Handle<RTVolumeRenderer> rt_renderer;

        std::shared_ptr<RenderParams> render_params;

        std::shared_ptr<PerFrameParams> per_frame_params;

        std::unique_ptr<NeuronRenderer> neuron_renderer;
    };
  private:
    void Draw(ComPtr<ID3D11ShaderResourceView> frame){

        dev_ctx->OMSetRenderTargets(1, rtv.GetAddressOf(), dsv.Get());

        dev_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        dev_ctx->VSSetShader(quad_vs_shader.Get(), nullptr, 0);
        dev_ctx->PSSetShader(quad_ps_shader.Get(), nullptr, 0);
        dev_ctx->PSSetShaderResources(0, 1, frame.GetAddressOf());

        dev_ctx->Draw(6, 0);
    }
  private:


    struct{
        ComPtr<ID3D11Texture2D> offscreen_frame;
        ComPtr<ID3D11Buffer> offscreen_buffer;

        ComPtr<ID3D11ShaderResourceView> osf_srv;

       Handle<FrameBuffer> vol_framebuffer;

        cudaGraphicsResource_t cuda_frame_color_resc = nullptr;
        //cudaGraphicsResource_t cuda_frame_depth_resc;// used to blend with mesh render result
    };

    struct{
        bool root;
        Ref<GPUMemMgr> gpu_mem_mgr_ref;

        GLFWwindow* glfw_window;
        uint32_t window_w, window_h;
        size_t window_framebuffer_size;


        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> dev_ctx;
        ComPtr<IDXGISwapChain> swap_chain;

        ComPtr<ID3D11RenderTargetView> rtv;
        ComPtr<ID3D11Texture2D> dsb;
        ComPtr<ID3D11DepthStencilView> dsv;


        ComPtr<ID3D11VertexShader> quad_vs_shader;
        ComPtr<ID3D11PixelShader> quad_ps_shader;

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
//        Handle<RTVolumeRenderer> rt_renderer;
//        std::unique_ptr<NeuronRenderer> neuron_renderer;
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

    VolViewRenderType view_render_type = VolViewRenderType::VOL;

    struct{
        std::shared_ptr<RenderParams> g_render_params;
        std::shared_ptr<PerFrameParams> g_per_frame_params;
        fps_camera_t camera;
    };

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
    SET_LOG_LEVEL_INFO


    if(glfwInit() != GLFW_TRUE){
        throw std::runtime_error("GLFW init failed!");
    }


    _ = std::make_unique<VolViewerPrivate>();

    _->info = info;

    auto& distr = DistributeMgr::GetInstance();
    DistributeMgr::GetInstance().SetRootRank(info.root_rank);


    _->node_window_count = info.window_infos.size();

    //todo check node_window_count <= node gpu count

    _->render_group.start(_->node_window_count);

    _->get_window_idx = [rank = DistributeMgr::GetInstance().GetWorldRank()](int i){
        return (i << 16) | rank;
    };

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

    auto render_space = (std::min)({volume_desc.voxel_space.x, volume_desc.voxel_space.y, volume_desc.voxel_space.z});

    size_t block_size = (size_t)(volume_desc.block_length + volume_desc.padding * 2) * volume_desc.bits_per_sample
                        * volume_desc.samples_per_voxel / 8;
    block_size *= block_size * block_size;

    size_t fixed_mem_size = (info.MaxHostMemGB << 30) * MEMORY_RATIO;
    size_t block_num = fixed_mem_size / block_size;
    //create fixed host mem mgr
#ifdef SHARED_FIXED_HOST_MEMORY
    FixedHostMemMgr::FixedHostMemMgrCreateInfo fixed_info;
    fixed_info.fixed_block_size = block_size;
    fixed_info.fixed_block_num = block_num;
    fixed_info.host_mem_mgr = host_mem_mgr_ref;

    auto fixed_host_mem_mgr_uid = host_mem_mgr_ref.Invoke(&HostMemMgr::RegisterFixedHostMemMgr, fixed_info);
    auto fixed_host_mem_mgr_ref = host_mem_mgr_ref.Invoke(&HostMemMgr::GetFixedHostMemMgrRef, fixed_host_mem_mgr_uid);
#endif
    //init render params
    _->g_render_params = std::make_shared<RenderParams>();
    ComputeUpBoundLOD(_->g_render_params->lod.leve_of_dist, render_space,
                      info.global_frame_width, info.global_frame_height, vutil::deg2rad(40.f));
    _->g_render_params->lod.updated = true;
    for(int i = 0; i < vol_info.levels; i++)
    _->g_render_params->lod.leve_of_dist.LOD[i] *= 1.5f;

    _->g_render_params->tf.updated = true;
    _->g_render_params->tf.tf_pts.pts.push_back({119.f/255.f, Float4(0.f, 0.f, 0.f, 0.f)});
    _->g_render_params->tf.tf_pts.pts.push_back({142.f/255.f, Float4(0.5f, 0.48443f, 0.36765f, 0.3412f)});
    _->g_render_params->tf.tf_pts.pts.push_back({238.f/255.f, Float4(0.853f, 0.338f, 0.092f, 0.73333f)});


    _->g_render_params->raycast.updated = true;
    _->g_render_params->raycast.ray_step = render_space * 0.5f;
    _->g_render_params->raycast.max_ray_dist = 6.f;

    _->g_render_params->other.updated = true;
    _->g_render_params->other.output_depth = false;

    _->g_render_params->distrib.world_row_count = info.global_window_rows;
    _->g_render_params->distrib.world_col_count = info.global_window_cols;

    _->g_per_frame_params = std::make_shared<PerFrameParams>();
    _->g_per_frame_params->frame_w_over_h = (float)info.global_frame_width / info.global_frame_height;

    // create renderer
    for(int i = 0; i < _->node_window_count; i++){
        auto gpu_mem_mgr_uid = ResourceMgr::GetInstance().RegisterResourceMgr({
            .type = ResourceMgr::Device,
            .MaxMemBytes = info.MaxGPUMemGB << 30,
            .DeviceIndex = i
        });
        auto gpu_mem_mgr_ref = ResourceMgr::GetInstance().GetGPURef(gpu_mem_mgr_uid).LockRef();

        int idx = _->get_window_idx(i);
        auto& resc = _->window_resc_mp[idx];
        auto& window_info = info.window_infos.at(i);
        resc.window =  std::make_unique<VolViewWindow>(VolViewWindow::VolViewWindowCreateInfo{
            .window_xpos = window_info.pos_x, .window_ypos = window_info.pos_y,
            .window_width = window_info.window_w, .window_height = window_info.window_h,
            .frame_width = info.node_frame_width, .frame_height = info.node_frame_height,
            .gpu_index = i, .root = ((idx >> 16) == 0),
            .gpu_mem_mgr_ref = gpu_mem_mgr_ref

        });
        resc.window->Initialize();
        resc.window->render_params = _->g_render_params;
        resc.window->per_frame_params = _->g_per_frame_params;

        RTVolumeRenderer::RTVolumeRendererCreateInfo rt_info{
            .host_mem_mgr = host_mem_mgr_ref.LockRef(),
            .gpu_mem_mgr = gpu_mem_mgr_ref.LockRef(),
#ifdef SHARED_FIXED_HOST_MEMORY
            .use_shared_host_mem = true,
            .shared_fixed_host_mem_mgr_ref = fixed_host_mem_mgr_ref.LockRef()
#else
            .use_shared_host_mem = false
#endif
        };
        auto renderer = NewHandle<RTVolumeRenderer>(ResourceType::Object, rt_info);

//        renderer->SetRenderMode(false);

        _->g_render_params->distrib.updated = true;

        float ox = (window_info.window_index_x - (info.global_window_cols * 0.5f - 0.5f)) * info.node_frame_width;
        float oy = (window_info.window_index_y - (info.global_window_rows * 0.5f - 0.5f)) * info.node_frame_height;
        LOG_INFO("window idx xy: {} {}, ox oy: {} {}",
                  window_info.window_index_x,
                  window_info.window_index_y,
                  ox, oy);
        _->g_render_params->distrib.node_x_offset = ox;
        _->g_render_params->distrib.node_y_offset = oy;
        _->g_render_params->distrib.node_x_index = window_info.window_index_x;
        _->g_render_params->distrib.node_y_index = window_info.window_index_y;


        renderer->SetRenderParams(*_->g_render_params);

        renderer->BindGridVolume(volume);

        LOG_DEBUG("BindGridVolume ok");

        _->window_resc_mp.at(_->get_window_idx(i)).window->rt_renderer = std::move(renderer);
    }

    _->g_render_params->Reset();

    Float3 default_pos = {3.60977, 2.882, 9.3109};//8.06206f
    _->camera.set_position(default_pos);
    _->camera.set_perspective(40.f, 0.001f, 10.f);
    _->camera.set_direction(vutil::deg2rad(-90.f), 0.f);
    _->camera.set_move_speed(0.005);
    _->camera.set_view_rotation_speed(0.0003f);
    //global camera for get proj view
    _->camera.set_w_over_h((float)info.global_frame_width / info.global_frame_height);

//    distr.WaitForSync();
    LOG_DEBUG("viewer create ok");
}

VolViewer::~VolViewer() {
    std::cerr << _->camera.get_position().x << " "
              << _->camera.get_position().y << " "
              << _->camera.get_position().z
              << std::endl;
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
    LOG_DEBUG("start run");

    int exit = false;
    bool show_mouse = false;
    auto should_close = [this, &exit]{

        return exit;
    };

    fps_camera_t::UpdateParams u_params;
    static double ox, oy;
    int render_params_updated = false;
    auto update_per_frame_params = [&]{
        _->camera.update(u_params);
        u_params = fps_camera_t::UpdateParams{};
        _->camera.recalculate_matrics();
        auto& params = *_->g_per_frame_params;
        params.frame_width = _->info.node_frame_width;
        params.frame_height = _->info.node_frame_height;
        //need to set global_window_w / global_window_h before
        //params.frame_w_over_h = (float)params.frame_width / (float)params.frame_height;
        params.fov = vutil::deg2rad(_->camera.get_fov_deg());
        params.cam_pos = _->camera.get_position();
        params.cam_dir = _->camera.get_xyz_direction();
        static Float3 WorldUp = {0.f, 1.f, 0.f};
        params.cam_right = vutil::cross(_->camera.get_xyz_direction(), WorldUp).normalized();
        params.cam_up = vutil::cross(params.cam_right, params.cam_dir);
        params.proj_view = _->camera.get_view_proj();

        //mpi
        auto& distr = DistributeMgr::GetInstance();
                distr.WaitForSync();
        distr.Bcast(reinterpret_cast<float*>(&params), 32);
        distr.Bcast(&_->g_render_params->raycast.ray_step, 3);
        distr.Bcast(&_->g_render_params->raycast.updated, 1);
//        distr.WaitForSync();

        for(auto& [idx, resc] : _->window_resc_mp){
            auto& window = resc.window;
            window->rt_renderer->SetRenderParams(*_->g_render_params);
            window->rt_renderer->SetPerFrameParams(params);
        }
        {
            render_params_updated |= _->g_render_params->tf.updated
                | _->g_render_params->lod.updated | _->g_render_params->light.updated | _->g_render_params->raycast.updated
                | _->g_render_params->distrib.updated | _->g_render_params->other.updated;
        }
        _->g_render_params->Reset();
    };

    CharCallback = [&](GLFWwindow* glfw_window, int c){
        ImGui_ImplGlfw_CharCallback(glfw_window, c);

    };
    FocusCallback = [&](GLFWwindow* glfw_window, int focus){
        if(focus){
            glfwSetInputMode(
                glfw_window, GLFW_CURSOR,
                show_mouse ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        }
    };

    ScrollCallback = [&](GLFWwindow* glfw_window,double x,double y){
        ImGui_ImplGlfw_ScrollCallback(glfw_window, x, y);
        if(show_mouse) return;

    };

    CursorCallback = [&](GLFWwindow* glfw_window,double x,double y){
        ImGui_ImplGlfw_CursorPosCallback(glfw_window, x, y);

        if(show_mouse) return;
        u_params.cursor_rel_x = x - ox;
        u_params.cursor_rel_y = y - oy;
        ox = x;
        oy = y;

    };

    MouseButtonCallback = [&](GLFWwindow* glfw_window, int button, int action, int mods){
        if(show_mouse){
            ImGui_ImplGlfw_MouseButtonCallback(glfw_window, button, action, mods);
            return;
        }


    };

    KeyCallback = [&](GLFWwindow* glfw_window,int key,int scancode,int action,int mods){
        ImGui_ImplGlfw_KeyCallback(glfw_window,key,scancode,action,mods);
        if(action == GLFW_RELEASE) return;
        if(key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS){
            show_mouse = !show_mouse;
            glfwSetInputMode(
                glfw_window, GLFW_CURSOR,
                show_mouse ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        }
        if(show_mouse) return;


        if(key == GLFW_KEY_W){
            u_params.front = true;
        }
        else if(key == GLFW_KEY_S){
            u_params.back = true;
        }
        else if(key == GLFW_KEY_A){
            u_params.left = true;
        }
        else if(key == GLFW_KEY_D){
            u_params.right = true;
        }
        else if(key == GLFW_KEY_SPACE){
            u_params.up = true;
        }
        else if(key == GLFW_KEY_LEFT_SHIFT){
            u_params.down = true;
        }
        else if(key == GLFW_KEY_ESCAPE){
            exit = true;
        }

    };

    auto update_cursor_pos = [&]{
        glfwGetCursorPos(root_window, &ox, &oy);
    };

    auto process_input = [&]{

        update_cursor_pos();

        glfwPollEvents();

        LOG_DEBUG("start update per frame params");
        update_per_frame_params();
        LOG_DEBUG("finish update per frame params");
        // mpi

    };

    auto should_redraw = [&]()->bool{
        if(show_mouse){
            if(render_params_updated) return true;
            return false;
        }
        else return true;
    };

    auto after_render = [&]{
        render_params_updated = false;


//        std::cerr << _->camera.get_position().x << " "
//                  << _->camera.get_position().y << " "
//                  << _->camera.get_position().z
//                  << std::endl;
    };

    while(!should_close()){
        vutil::AutoTimer frame_timer("render frame");
///        LOG_DEBUG("start process input");
        process_input();
//        LOG_DEBUG("finish process input");
        auto render_task = _->render_group.create_task();

        for(auto& [idx, resc] : _->window_resc_mp){
            auto& window = resc.window;
            render_task->enqueue_task([&, idx = idx]{
                vutil::AutoTimer _timer("render window " + std::to_string(idx >> 16));
//                if((idx >> 16) == 0) return;
                window->PreRender();
                window->Draw(VolViewRenderType::VOL, !should_redraw());
//                window->Commit();
            });
        }
        _->render_group.submit(render_task);

        _->render_group.wait_idle();

        auto draw_task = _->render_group.create_task();

        for(auto& [idx, resc] : _->window_resc_mp){
            auto& window = resc.window;
            draw_task->enqueue_task([&]{
                window->Commit();
            });
        }

        _->render_group.submit(draw_task);

        _->render_group.wait_idle();

        after_render();

    }

}
