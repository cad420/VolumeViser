#include "VolViewer.hpp"

#include <console/cmdline.hpp>

#include <json.hpp>
#include <fstream>

extern int GetWorldRank();

void ParseFromJsonFile(const std::string& filename, VolViewerCreateInfo& info){
    nlohmann::json j;
    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::runtime_error("Open VolViewer config file failed with filename: " + filename);
    }
    in >> j;

    info.global_frame_width = j.at("global-frame-width");
    info.global_frame_height = j.at("global-frame-height");

    auto& nodes = j.at("node");
    info.root_rank = nodes.at("root-rank");
    info.global_window_rows = nodes.at("window-rows");
    info.global_window_cols = nodes.at("window-cols");
    int world_rank = GetWorldRank();
    int node_count = nodes.at("node-count");
    if(world_rank >= node_count){
        throw std::runtime_error("node-count in config file is wrong with exactly run node count");
    }
    info.node_frame_width = nodes.at("node-frame-width");
    info.node_frame_height = nodes.at("node-frame-height");
    auto node_name = std::string("node-") + std::to_string(world_rank);
    auto& node = nodes.at(node_name);
    info.resource_path = node.at("resource-path");
    int window_count = node.at("window-count");
    for(int i = 0; i < window_count; ++i){
        auto& window = node.at("window-" + std::to_string(i));
        auto& window_info = info.window_infos.emplace_back();
        window_info.pos_x = window.at("window-xpos");
        window_info.pos_y = window.at("window-ypos");
        window_info.window_w = window.at("window-w");
        window_info.window_h = window.at("window-h");
        window_info.window_index_x = window.at("window-index-x");
        window_info.window_index_y = window.at("window-index-y");
    }
}

int main(int argc, char** argv){
    try{
        cmdline::parser cmd;

        cmd.add<std::string>("config-file", 'c', "config json filename");

        cmd.parse_check(argc, argv);

        auto filename = cmd.get<std::string>("config-file");

        VolViewerCreateInfo info;

        ParseFromJsonFile(filename, info);

        VolViewer(info).run();
    }
    catch (const std::exception& err)
    {
        std::cerr << "Process exited with: " << err.what() << std::endl;
    }
}