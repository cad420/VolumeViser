#include <Core/ResourceMgr.hpp>

VISER_BEGIN


    ResourceMgr &ResourceMgr::GetInstance() {
        static ResourceMgr ins;

        return ins;
    }

    ResourceMgr::~ResourceMgr() {

    }


VISER_END


