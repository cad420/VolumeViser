#include "VolAnnotater.hpp"

#include <console/cmdline.hpp>

#include <json.hpp>
#include <fstream>

void ParseFromJsonFile(const std::string& filename, VolAnnotaterCreateInfo& info){
    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::runtime_error("Failed to open json file : " + filename);
    }
    nlohmann::json j;
    in >> j;

    size_t max_host_mem_gb = j.at("max_host_mem_gb");
    info.max_host_mem_bytes = max_host_mem_gb << 30;

    size_t max_render_gpu_mem_gb = j.at("max_render_gpu_mem_gb");
    info.max_render_gpu_mem_bytes = max_render_gpu_mem_gb << 30;

    size_t max_compute_gpu_mem_gb = j.at("max_compute_gpu_mem_gb");
    info.max_compute_gpu_mem_bytes = max_compute_gpu_mem_gb << 30;

    info.render_gpu_index = j.at("render_gpu_index");

    info.compute_gpu_index = j.at("compute_gpu_index");

    info.render_compute_same_gpu = info.render_gpu_index == info.compute_gpu_index;

    size_t max_fixed_host_mem_gb = j.at("max_fixed_host_mem_gb");
    info.max_fixed_host_mem_bytes = max_fixed_host_mem_gb << 30;

    info.threads_count = j.at("async_decoding_threads_count");

    info.vtex_count = j.at("virtual_texture_count");

    std::array<int,3> shape = j.at("virtual_texture_shape");
    info.vtex_shape_x = shape[0], info.vtex_shape_y = shape[1], info.vtex_shape_z = shape[2];
}

int main(int argc, char** argv){
    try {
        cmdline::parser cmd;

        cmd.add("config", 'c', "use json file as program configure");
        cmd.add<std::string>("filename", 'j', "json config filename");

        cmd.add<int>("hmem", 'h', "max host memory usage(GB)", false, 16);

        cmd.add<int>("vtex_count", 0, "virtual texture count", false, 1);

        cmd.parse_check(argc, argv);

        VolAnnotaterCreateInfo info{};

        if(cmd.exist("config")){
            std::string json_filename = cmd.get<std::string>("filename");
            ParseFromJsonFile(json_filename, info);
        }
        else{

        }

        VolAnnotater(info).run();

    }
    catch (const std::exception& err) {
        std::cerr << "Process exit with error : " << err.what() << std::endl;
    }
}