#pragma once

#include <Common/Common.hpp>

VISER_BEGIN

class VolumeIOInterface{
public:
    virtual ~VolumeIOInterface() = default;

    virtual void ReadVolumeRegion() = 0;

    virtual void WriteVolumeRegion() = 0;
};

class SWCIOInterface{
public:

};

class MeshIOInterface{
public:

};


VISER_END

