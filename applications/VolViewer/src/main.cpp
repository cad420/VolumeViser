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

    auto& nodes = j.at("node");
    info.frame_width = nodes.at("frame-width");
    info.frame_height = nodes.at("frame-height");
    info.root_rank = nodes.at("root-rank");
    int world_rank = GetWorldRank();
    int node_count = nodes.at("node-count");
    if(world_rank >= node_count){
        throw std::runtime_error("node-count in config file is wrong with exactly run node count");
    }
    auto node_name = std::string("node-") + std::to_string(world_rank);
    auto& node = nodes.at(node_name);
    info.resource_path = node.at("resource-path");
    int window_count = node.at("window-count");
    for(int i = 0; i < window_count; ++i){
        auto& window = node.at("window-" + std::to_string(i));
        auto& shape = info.window_infos.emplace_back();
        shape[0] = window.at("window-xpos");
        shape[1] = window.at("window-ypos");
        shape[2] = window.at("window-w");
        shape[3] = window.at("window-h");
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
        std::cerr << "Process exited with: " << std::endl;
    }
}