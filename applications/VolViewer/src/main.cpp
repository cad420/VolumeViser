#include "VolViewer.hpp"

#include <console/cmdline.hpp>

void ParseFromJsonFile(const std::string& filename, VolViewerCreateInfo& info){

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