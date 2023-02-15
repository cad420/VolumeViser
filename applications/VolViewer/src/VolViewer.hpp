#pragma once

#include <memory>

struct VolViewerCreateInfo{

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

