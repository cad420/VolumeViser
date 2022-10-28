#include "VolAnnotater.hpp"

#include <console/cmdline.hpp>

int main(int argc, char** argv){
    try {
        cmdline::parser cmd;


        cmd.parse_check(argc, argv);



        VolAnnotater::VolAnnotaterCreateInfo info{};
        VolAnnotater(info).run();
    }
    catch (const std::exception& err) {
        std::cerr << "Process exit with error : " << err.what() << std::endl;
    }
}