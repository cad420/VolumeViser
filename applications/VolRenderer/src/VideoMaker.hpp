#pragma once

#include <memory>

class VideoMakerPrivate;
class VideoMaker{
  public:
    struct VideoMakerCreateInfo{

    };
    explicit VideoMaker(const VideoMakerCreateInfo& info);

    ~VideoMaker();



  private:
    std::unique_ptr<VideoMakerPrivate> _;
};