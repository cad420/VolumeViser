#pragma once

#include <Core/Renderer.hpp>
#include <json.hpp>
#include <fstream>

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

enum LightType{
    Point,
    Spot,
    Direct
};


struct VolRendererConfigParams{
    struct Settings{

    }settings;

    struct Camera{

    }camera;

    struct TransferFunc{

    }tf;

    struct Lights{

    }lights;

    struct Render{

    }render;

    struct Frame{
        Int2 resolution;

    }frame;

    struct Data{
        std::vector<std::string> lod_filenames;
    }data;

    struct Memory{
        size_t max_host_mem_bytes = 8ull << 30;
        size_t max_gpu_mem_bytes = 10ull << 30;
        uint32_t gpu_index = 0;
    }memory;

    struct Animation{
        int frame_count;
        std::vector<Mat4> transforms; // for camera
        // todo: other operations like change fov, near/far plane or field of depth
    }animation;
};

inline void LoadFromJsonFile(VolRendererConfigParams& params, std::string_view filename){
    std::ifstream in(filename.data());
    if(!in.is_open()){
        throw std::runtime_error("Open vol renderer config json file failed");
    }
    nlohmann::json j;
    in >> j;
    in.close();

    auto must_have = [&](const std::string& item, auto& json)->decltype(auto){
        if(json.count(item) == 0){
            LOG_ERROR("Json config file missed item({})", item);
            exit(0);
        }
        return json.at(item);
    };
    bool ok;
    nlohmann::json::value_type empty;
    auto may_have = [&](const std::string& item, auto& json)-> nlohmann::json::value_type&{
        if(json.count(item) == 0){
            ok = false;
            return empty;
        }
        ok = true;
        return json.at(item);
    };

    // data
    {
        auto data = must_have("data", j);
        int levels = data.at("levels");
        for(int i = 0; i < levels; i++){
            std::string lod = "lod" + std::to_string(i);
            std::string filename = data.at(lod);
            params.data.lod_filenames.push_back(filename);
        }
    }

    // memory
    {
        auto memory = may_have("memory", j);
        if(ok){

        }
    }

}

inline void SetupVolumeIO(GridVolume::GridVolumeCreateInfo& volInfo,
                   const VolRendererConfigParams::Data& data){
    if(data.lod_filenames.empty()){
        throw std::runtime_error("Empty lod filenames for loading volume data");
    }

    int lod = 0;
    for(auto& filename : data.lod_filenames){
        volInfo.lod_vol_file_io[lod++] = Handle<VolumeIOInterface>(ResourceType::Object,
                                                                   CreateVolumeFileByFileName(filename));
    }
    volInfo.levels = lod;
}