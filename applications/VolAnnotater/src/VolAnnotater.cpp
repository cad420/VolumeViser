#include "Common.hpp"
#include "VolAnnotater.hpp"
#include "SWCRenderer.hpp"
#include "NeuronRenderer.hpp"
#include "VolAnnotaterGUI.hpp"
#undef max
#undef min
#undef far
#undef near

//标注系统的窗口绘制任务交给OpenGL，如果有多个显卡，其余的显卡可以用于网格重建任务
#define FOV 40.f

class VolAnnotaterPrivate{
public:

    VolAnnotaterCreateInfo create_info;

    std::unique_ptr<VolAnnotaterGUI> gui;

};

VolAnnotater::VolAnnotater(const VolAnnotaterCreateInfo &info) {
    _ = std::make_unique<VolAnnotaterPrivate>();
    _->create_info = info;

    if(info.log_level == LOG_LEVEL_DEBUG){
        SET_LOG_LEVEL_DEBUG
    }
    else if(info.log_level == LOG_LEVEL_INFO){
        SET_LOG_LEVEL_INFO
    }
    else if(info.log_level == LOG_LEVEL_ERROR){
        SET_LOG_LEVEL_ERROR
    }
    else{
        SET_LOG_LEVEL_CRITICAL
    }

    SET_LOG_LEVEL_TRACE

    try{
        AppSettings::Initialize(info);

        _->gui = std::make_unique<VolAnnotaterGUI>(window_desc_t{
                .size = {info.window_width, info.window_height},
                .title = "VolAnnotater"
        });

        _->gui->Initialize();
    }
    catch (const std::exception& err) {
        LOG_ERROR("VolAnnotater create failed : {}", err.what());
        throw std::runtime_error(std::string("Create VolAnnotater failed : ") + err.what());
    }
}

VolAnnotater::~VolAnnotater() {

}

void VolAnnotater::run() {

    try{
        _->gui->run();
    }
    catch(const std::exception& err){
        LOG_ERROR("VolAnnotater run error : {}", err.what());
        throw std::runtime_error(std::string("Run VolAnnotater failed : ") + err.what());
    }

}
