#pragma once

#include <memory>

class VolViewerPrivate;
class VolViewer{
public:
    struct VolViewerCreateInfo{

    };

    explicit VolViewer(const VolViewerCreateInfo& info);

    ~VolViewer();



private:
    std::unique_ptr<VolViewerPrivate> _;
};

