#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN

class SWCFile : public SWCIOInterface{
public:
    struct SWCPoint{
        int id;
        int tag;
        float x;
        float y;
        float z;
        float radius;
        int pid;
        int pad = 0;
    };
    static_assert(sizeof(SWCPoint) == 32, "");


};

VISER_END