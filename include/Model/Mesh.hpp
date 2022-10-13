#pragma once

#include <IO/MeshIO.hpp>

VISER_BEGIN

class MeshPrivate;
class Mesh{
public:


protected:
    std::unique_ptr<MeshPrivate> _;
};


VISER_END
