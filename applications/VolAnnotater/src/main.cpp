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

    info.threads_count = j.at("cpu_decoding_threads_count");

    info.vtex_count = j.at("virtual_texture_count");

    std::array<int,3> shape = j.at("virtual_texture_shape");
    info.vtex_shape_x = shape[0], info.vtex_shape_y = shape[1], info.vtex_shape_z = shape[2];

    info.log_level = j.at("log_level");
}

int main(int argc, char** argv){
    try {
        cmdline::parser cmd;

        cmd.add("config", 'c', "use json file as program configure");
        cmd.add<std::string>("filename", 'j', "json config filename");

        cmd.add<int>("hmem", 'h', "max host memory size(GB)", false, 16);

        cmd.add<int>("vtex_count", 'v', "virtual texture count", false, 1);

        cmd.add<int>("fmem", 'f', "max fixed host memory size(GB)", false, 12);

        cmd.add<int>("render_gpu", 0, "render gpu index", false, 0);

        cmd.add<int>("compute_gpu", 0, "compute gpu index", false, 0);

        cmd.add<int>("render_mem", 0, "max render gpu memory size(GB)", false, 6);

        cmd.add<int>("compute_mem", 0, "max compute gpu memory size(GB)", false, 6);

        cmd.add<int>("cpu_decoding_threads_count", 0, "cpu decoding threads count", false, 2);

        cmd.add<int>("vtex_x", 'x', "virtual texture shape x", false, 1024);

        cmd.add<int>("vtex_y", 'z', "virtual texture shape x", false, 1024);

        cmd.add<int>("vtex_z", 'y', "virtual texture shape x", false, 1024);

        cmd.add<int>("log_level", 'l', "log level", false, 2);

        cmd.parse_check(argc, argv);

        VolAnnotaterCreateInfo info{};

        if(cmd.exist("config")){
            std::string json_filename = cmd.get<std::string>("filename");
            ParseFromJsonFile(json_filename, info);
        }
        else{
            info.max_host_mem_bytes = cmd.get<int>("hmem") * (1ull << 30);
            info.max_fixed_host_mem_bytes = cmd.get<int>("fmem") * (1ull << 30);
            info.max_render_gpu_mem_bytes = cmd.get<int>("render_mem") * (1ull << 30);
            info.max_compute_gpu_mem_bytes = cmd.get<int>("compute_mem") * (1ull << 30);
            info.render_gpu_index = cmd.get<int>("render_gpu");
            info.compute_gpu_index = cmd.get<int>("compute_gpu");
            info.render_compute_same_gpu = info.render_gpu_index == info.compute_gpu_index;
            info.threads_count = cmd.get<int>("cpu_decoding_threads_count");
            info.vtex_count = cmd.get<int>("vtex_count");
            info.vtex_shape_x = cmd.get<int>("vtex_x");
            info.vtex_shape_y = cmd.get<int>("vtex_y");
            info.vtex_shape_z = cmd.get<int>("vtex_z");
            info.log_level = cmd.get<int>("log_level");
        }

        VolAnnotater(info).run();

    }
    catch (const std::exception& err) {
        std::cerr << "Process exit with error : " << err.what() << std::endl;
    }
}