#pragma once

#include <memory>
#include <string>
#include <vector>

struct VolViewerCreateInfo{
    int frame_width = 0;
    int frame_height = 0;
    int root_rank = 0;

    std::string resource_path;
    std::vector<std::array<int,4>> window_infos;

    size_t MaxHostMemGB = 16;
    size_t MaxGPUMemGB = 16;
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

