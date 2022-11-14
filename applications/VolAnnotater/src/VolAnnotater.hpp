#pragma once

#include <memory>
#include <string>

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_ERROR 2

struct VolAnnotaterCreateInfo{
    size_t max_host_mem_bytes;
    bool render_compute_same_gpu = true;
    int render_gpu_index = 0;
    int compute_gpu_index = 0;
    size_t max_render_gpu_mem_bytes;
    size_t max_compute_gpu_mem_bytes;

    size_t max_fixed_host_mem_bytes;

    int threads_count = 2;
    // virtual texture
    int vtex_count;
    int vtex_shape_x, vtex_shape_y, vtex_shape_z;

    // window
    int window_width = 1920, window_height = 1080;
    // log
    int log_level = LOG_LEVEL_ERROR;
    bool save_log_file = false;
    std::string log_filename;
};

class VolAnnotaterPrivate;
class VolAnnotater final{
public:

    explicit VolAnnotater(const VolAnnotaterCreateInfo& info);

    ~VolAnnotater();

    void run();

private:
    std::unique_ptr<VolAnnotaterPrivate> _;
};