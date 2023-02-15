#include "VolViewer.hpp"

#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Algorithm/MarchingCube.hpp>
#include <Algorithm/Voxelization.hpp>
#include <Core/Renderer.hpp>
#include <Core/HashPageTable.hpp>
#include <Model/SWC.hpp>
#include <Model/Mesh.hpp>
#include <Model/SWC.hpp>
#include <IO/SWCIO.hpp>

#include <cuda_gl_interop.h>
#include <json.hpp>

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

//一个主机只有一个节点，一个节点可以对应多个渲染器，一个渲染器对应一个窗口，一般一个窗口对应一个显示屏
//window不用记录相机参数 虽然每个节点有单独的相机 但是只会使用root节点记录的相机参数
class VolViewWindow {
  public:
    struct VolViewWindowCreateInfo{
        bool control_window = false;
    };

    void Initialize();

    //调用gl指令绘制
    void Draw(Handle<FrameBuffer> frame);

    //统一提交后swap buffer 用于保持整体画面的一致性
    void Commit();

  private:



};

struct GlobalSettings{
    inline static UnifiedRescUID fixed_host_mem_mgr_uid = 0;
    inline static bool async_render = true;
    inline static vec2i node_window_size{1920, 1080};
};

class VolViewerPrivate{
public:
  struct WindowRescPack{
      std::unique_ptr<VolViewWindow> window;
      Handle<RTVolumeRenderer> rt_renderer;
  };
  std::map<int, WindowRescPack> window_resc_mp;

};

VolViewer::VolViewer(const VolViewerCreateInfo &info) {

}

VolViewer::~VolViewer() {

}

void VolViewer::run()
{

}
