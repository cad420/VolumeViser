#pragma once

#include <memory>
#include <string>
#include <vector>

struct VolViewerCreateInfo{
    int global_frame_width = 0;
    int global_frame_height = 0;
    int global_window_rows = 1;
    int global_window_cols = 1;
    int node_frame_width = 0;
    int  node_frame_height = 0;
    int root_rank = 0;

    std::string resource_path;

    struct WindowInfos{
        int pos_x, pos_y;
        int window_w, window_h;
        int window_index_x, window_index_y;
    };
    std::vector<WindowInfos> window_infos;


    size_t MaxHostMemGB = 16;
    size_t MaxGPUMemGB = 20;
};

class VolViewerPrivate;
class VolViewer final{
public:

    explicit VolViewer(const VolViewerCreateInfo& info);

    ~VolViewer();

    void run();

private:
    std::unique_ptr<VolViewerPrivate> _;
};

