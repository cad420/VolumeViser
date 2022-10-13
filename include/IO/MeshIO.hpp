#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN


    class MeshFile : public MeshIOInterface{
    public:
        MeshFile(std::string_view filename);

        ~MeshFile();





    };


VISER_END
